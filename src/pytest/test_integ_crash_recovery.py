"""Integration test 3: Crash Recovery via WAL.

A writer adds vectors to the accumulator (WAL records them) but the
process terminates abruptly via os._exit() — no destructors or cleanup
code run.  A new process opens the dataset, WAL replays, and the
vectors are recoverable.
"""

from __future__ import annotations

import json
import unittest

from sketch2_wrapper import Sketch2

from integ_helpers import IntegTestBase, run_subprocess


class CrashRecoveryTest(IntegTestBase):
    _tmpdir_prefix = "sketch2_integ_crash_"

    def test_wal_replay_recovers_unflushed_vectors(self) -> None:
        # Step 1: Create dataset and persist some base data.
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dim=4, range_size=1000, dist_func="l1")
            ps.generate(count=10, start_id=0, pattern=0)
            ps.close(self.dataset_name)
        self.progress("Base data persisted (10 vectors)")

        # Step 2: Child process opens dataset, upserts vectors into accumulator
        # (written to WAL), then terminates abruptly via os._exit() so that
        # no Python/C++ destructors run — simulating a real crash.
        crash_script = f"""
import os
from sketch2_wrapper import Sketch2
ps = Sketch2({str(self.root)!r})
ps.open({self.dataset_name!r})
for i in range(100, 110):
    val = str(float(i))
    ps.upsert(i, f"{{val}}, {{val}}, {{val}}, {{val}}")
os._exit(0)
"""
        result = run_subprocess(crash_script)
        self.assert_subprocess_ok(result, "crash script")
        self.progress("Crash simulated (os._exit after 10 unflushed upserts)")

        # Step 3: New process opens dataset. WAL replays accumulator.
        recovery_script = f"""
import json
from sketch2_wrapper import Sketch2
with Sketch2({str(self.root)!r}) as ps:
    ps.open({self.dataset_name!r})
    results = {{}}
    for i in range(10):
        vec = ps.get(i)
        results[i] = vec
    for i in range(100, 110):
        vec = ps.get(i)
        results[i] = vec
    print(json.dumps(results))
    ps.close({self.dataset_name!r})
"""
        result = run_subprocess(recovery_script)
        self.assert_subprocess_ok(result, "recovery script")
        recovered = json.loads(result.stdout.strip())
        for i in range(10):
            self.assertIn(str(i), recovered, f"Missing base id {i}")
        for i in range(100, 110):
            self.assertIn(str(i), recovered, f"Missing WAL id {i}")
        self.progress(f"WAL replay recovered {len(recovered)} vectors (10 base + 10 WAL)")

    def test_wal_recovered_data_persists_after_merge(self) -> None:
        # Create dataset and persist base data.
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dim=4, range_size=1000, dist_func="l1")
            ps.generate(count=10, start_id=0, pattern=0)
            ps.close(self.dataset_name)
        self.progress("Base data persisted (10 vectors)")

        # Child: upsert without flush, then terminate abruptly (no destructors).
        crash_script = f"""
import os
from sketch2_wrapper import Sketch2
ps = Sketch2({str(self.root)!r})
ps.open({self.dataset_name!r})
for i in range(50, 55):
    val = str(float(i))
    ps.upsert(i, f"{{val}}, {{val}}, {{val}}, {{val}}")
os._exit(0)
"""
        result = run_subprocess(crash_script)
        self.assert_subprocess_ok(result, "crash script")
        self.progress("Crash simulated (os._exit after 5 unflushed upserts)")

        # New handle: open, verify WAL data, merge, close.
        with Sketch2(self.root) as ps:
            ps.open(self.dataset_name)
            vec = ps.get(50)
            self.assertTrue(len(vec) > 0)
            ps.merge_accumulator()
            ps.close(self.dataset_name)
        self.progress("WAL replayed, merge_accumulator persisted recovered data")

        # Another handle: verify everything is in persisted files.
        with Sketch2(self.root) as ps:
            ps.open(self.dataset_name)
            for i in range(10):
                self.assertTrue(len(ps.get(i)) > 0, f"Missing base id {i}")
            for i in range(50, 55):
                self.assertTrue(len(ps.get(i)) > 0, f"Missing recovered id {i}")
            ps.close(self.dataset_name)
        self.progress("All 15 vectors verified after reopen")


if __name__ == "__main__":
    unittest.main()
