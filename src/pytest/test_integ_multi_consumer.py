"""Integration test 6: Producer With Multiple Consumers.

One writer produces data in rounds.  After each round, multiple reader
subprocesses independently query the dataset and all get consistent results.
Also exercises true concurrent access: readers hold open handles and query
while a writer is actively mutating the dataset.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest

from sketch2_wrapper import Sketch2

from integ_helpers import IntegTestBase, format_process_error, subprocess_env

NUM_READERS = 4
QUERIES = [
    "0.0, 0.0, 0.0, 0.0",
    "50.0, 50.0, 50.0, 50.0",
    "100.0, 100.0, 100.0, 100.0",
    "250.0, 250.0, 250.0, 250.0",
    "499.0, 499.0, 499.0, 499.0",
]


class MultiConsumerTest(IntegTestBase):
    _tmpdir_prefix = "sketch2_integ_multi_"

    def _spawn_reader(self, queries: list[str], k: int) -> subprocess.Popen:
        """Spawn a reader subprocess that runs KNN queries and prints JSON results."""
        queries_json = json.dumps(queries)
        script = f"""
import json
from sketch2_wrapper import Sketch2
queries = json.loads({queries_json!r})
with Sketch2({str(self.root)!r}) as ps:
    ps.open({self.dataset_name!r})
    results = []
    for q in queries:
        ids = ps.knn(q, {k})
        results.append(ids)
    ps.close({self.dataset_name!r})
    print(json.dumps(results))
"""
        return subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=subprocess_env(),
        )

    def _collect_reader_results(self, procs: list[subprocess.Popen]) -> list[list[list[int]]]:
        results = []
        for idx, proc in enumerate(procs):
            stdout, stderr = proc.communicate(timeout=30)
            self.assertEqual(0, proc.returncode,
                format_process_error(proc.returncode, stdout, stderr, f"reader {idx}"))
            results.append(json.loads(stdout.strip()))
        return results

    def test_all_readers_agree_after_initial_load(self) -> None:
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dim=4, range_size=1000, dist_func="l1")
            ps.generate(count=500, start_id=0, pattern=0)
            ps.close(self.dataset_name)

        # Spawn readers concurrently.
        procs = [self._spawn_reader(QUERIES, k=5) for _ in range(NUM_READERS)]
        results = self._collect_reader_results(procs)

        # All readers must return identical results.
        for i in range(1, NUM_READERS):
            self.assertEqual(results[0], results[i],
                f"Reader 0 and reader {i} disagree")

    def test_all_readers_agree_after_update(self) -> None:
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dim=4, range_size=1000, dist_func="l1")
            ps.generate(count=500, start_id=0, pattern=0)
            ps.close(self.dataset_name)

        # Get baseline results.
        procs = [self._spawn_reader(QUERIES, k=5) for _ in range(NUM_READERS)]
        baseline = self._collect_reader_results(procs)
        for i in range(1, NUM_READERS):
            self.assertEqual(baseline[0], baseline[i])

        # Writer updates some vectors.
        with Sketch2(self.root) as ps:
            ps.open(self.dataset_name)
            for i in range(100):
                ps.upsert(i, f"{i + 0.5}, {i + 0.5}, {i + 0.5}, {i + 0.5}")
            ps.merge_accumulator()
            ps.close(self.dataset_name)

        # All readers must see updated data and agree.
        procs = [self._spawn_reader(QUERIES, k=5) for _ in range(NUM_READERS)]
        updated = self._collect_reader_results(procs)
        for i in range(1, NUM_READERS):
            self.assertEqual(updated[0], updated[i],
                f"Reader 0 and reader {i} disagree after update")

    def test_readers_run_concurrently_without_corruption(self) -> None:
        """Multiple readers and sequential writes over several rounds."""
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dim=4, range_size=1000, dist_func="l1")
            ps.close(self.dataset_name)

        for round_idx in range(5):
            # Writer adds a batch.
            with Sketch2(self.root) as ps:
                ps.open(self.dataset_name)
                ps.generate(count=100, start_id=round_idx * 100, pattern=0)
                ps.close(self.dataset_name)

            # Readers query concurrently.
            procs = [self._spawn_reader(QUERIES[:2], k=3) for _ in range(NUM_READERS)]
            results = self._collect_reader_results(procs)
            for i in range(1, NUM_READERS):
                self.assertEqual(results[0], results[i],
                    f"Round {round_idx}: reader disagreement")


    def test_concurrent_reader_writer(self) -> None:
        """Readers query with open handles while a writer is actively mutating."""
        # Seed the dataset so readers always have data to query.
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dim=4, range_size=1000, dist_func="l1")
            ps.generate(count=200, start_id=0, pattern=0)
            ps.close(self.dataset_name)

        ready_flag = self.root / "_writer_ready"
        done_flag = self.root / "_writer_done"

        # Writer: signal readiness, then upsert across many rounds so that
        # readers have time to issue queries against the live dataset.
        writer_script = f"""
from pathlib import Path
from sketch2_wrapper import Sketch2
with Sketch2(Path({str(self.root)!r})) as ps:
    ps.open({self.dataset_name!r})
    Path({str(ready_flag)!r}).touch()
    for r in range(20):
        for i in range(10):
            vid = 1000 + r * 10 + i
            val = str(float(vid))
            ps.upsert(vid, f"{{val}}, {{val}}, {{val}}, {{val}}")
        ps.merge_accumulator()
    ps.close({self.dataset_name!r})
Path({str(done_flag)!r}).touch()
"""

        # Reader: wait for the writer to be active, then query in a tight
        # loop until the writer signals completion.  Each query must return
        # the right number of results and must not raise.
        reader_script = f"""
import json, time
from pathlib import Path
from sketch2_wrapper import Sketch2
ready = Path({str(ready_flag)!r})
done = Path({str(done_flag)!r})
while not ready.exists():
    time.sleep(0.005)
query_count = 0
errors = []
with Sketch2(Path({str(self.root)!r})) as ps:
    ps.open({self.dataset_name!r})
    while not done.exists():
        try:
            ids = ps.knn("100.0, 100.0, 100.0, 100.0", 5)
            if len(ids) != 5:
                errors.append(f"query {{query_count}}: expected 5 results, got {{len(ids)}}")
        except Exception as e:
            errors.append(f"query {{query_count}}: {{e}}")
        query_count += 1
    ps.close({self.dataset_name!r})
print(json.dumps({{"query_count": query_count, "errors": errors}}))
"""

        # Launch writer and readers at the same time.
        writer = subprocess.Popen(
            [sys.executable, "-c", writer_script],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=subprocess_env(),
        )
        readers = [
            subprocess.Popen(
                [sys.executable, "-c", reader_script],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=subprocess_env(),
            )
            for _ in range(NUM_READERS)
        ]

        # Collect writer.
        w_out, w_err = writer.communicate(timeout=60)
        self.assertEqual(0, writer.returncode,
            format_process_error(writer.returncode, w_out, w_err, "writer"))

        # Collect readers — each must have completed successfully and
        # executed at least one query while the writer was active.
        for idx, proc in enumerate(readers):
            stdout, stderr = proc.communicate(timeout=60)
            self.assertEqual(0, proc.returncode,
                format_process_error(proc.returncode, stdout, stderr, f"reader {idx}"))
            data = json.loads(stdout.strip())
            self.assertGreater(data["query_count"], 0,
                f"Reader {idx} ran zero queries — no concurrency exercised")
            self.assertEqual([], data["errors"],
                f"Reader {idx} errors: {data['errors']}")


if __name__ == "__main__":
    unittest.main()
