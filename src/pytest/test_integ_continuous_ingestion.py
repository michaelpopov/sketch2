"""Integration test 1: Continuous Ingestion Pipeline.

A writer ingests vectors in repeated batches, flushing the accumulator
each time.  A reader subprocess queries between batches and must always
see the latest persisted data.
"""

from __future__ import annotations

import json
import unittest

from sketch2_wrapper import Sketch2

from integ_helpers import IntegTestBase, run_subprocess

ROUNDS = 10
BATCH_SIZE = 200


class ContinuousIngestionTest(IntegTestBase):
    _tmpdir_prefix = "sketch2_integ_ingest_"

    def test_reader_sees_latest_data_after_each_batch(self) -> None:
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dim=4, range_size=1000, dist_func="l1")
            ps.close(self.dataset_name)

        for round_idx in range(ROUNDS):
            start_id = round_idx * BATCH_SIZE
            # Writer: insert a batch with a unique "anchor" vector at start_id.
            with Sketch2(self.root) as ps:
                ps.open(self.dataset_name)
                ps.generate(count=BATCH_SIZE, start_id=start_id, pattern=0)
                ps.close(self.dataset_name)

            # Reader (subprocess): query for the anchor and verify it's found.
            anchor_val = f"{start_id + 0.1}"
            query_vec = f"{anchor_val}, {anchor_val}, {anchor_val}, {anchor_val}"
            script = f"""
import json
from sketch2_wrapper import Sketch2
with Sketch2({str(self.root)!r}) as ps:
    ps.open({self.dataset_name!r})
    ids = ps.knn({query_vec!r}, 1)
    ps.close({self.dataset_name!r})
    print(json.dumps(ids))
"""
            result = run_subprocess(script)
            self.assert_subprocess_ok(result, f"round {round_idx} reader")
            ids = json.loads(result.stdout.strip())
            self.assertEqual([start_id], ids,
                f"Round {round_idx}: expected [{start_id}], got {ids}")
            self.progress(f"Round {round_idx}/{ROUNDS}: ingested ids {start_id}..{start_id + BATCH_SIZE - 1}, reader found anchor {start_id}")

    def test_growing_dataset_knn_correctness(self) -> None:
        """After all rounds, KNN across the entire dataset returns correct order."""
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dim=4, range_size=1000, dist_func="l1")
            for round_idx in range(ROUNDS):
                ps.generate(count=BATCH_SIZE, start_id=round_idx * BATCH_SIZE, pattern=0)
            self.progress(f"Loaded {ROUNDS * BATCH_SIZE} vectors across {ROUNDS} rounds")

            # Query for vector closest to origin.
            ids = ps.knn("0.0, 0.0, 0.0, 0.0", 3)
            self.assertEqual([0, 1, 2], ids)
            self.progress(f"KNN near origin: {ids}")

            ps.close(self.dataset_name)


if __name__ == "__main__":
    unittest.main()
