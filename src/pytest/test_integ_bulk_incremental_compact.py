"""Integration test 2: Bulk Load, Incremental Updates, and Full Compaction.

Initial bulk load, then rounds of upserts + deletes creating deltas, then
full compaction.  Query results must be stable across compaction.
"""

from __future__ import annotations

import unittest

from sketch2_wrapper import Sketch2, Sketch2Error

from integ_helpers import IntegTestBase

BULK_COUNT = 5000
RANGE_SIZE = 1000
UPDATE_ROUNDS = 5
UPSERTS_PER_ROUND = 200
UPDATES_PER_ROUND = 100
DELETES_PER_ROUND = 50


class BulkIncrementalCompactTest(IntegTestBase):
    _tmpdir_prefix = "sketch2_integ_compact_"

    def _reference_queries(self, ps: Sketch2) -> list[list[int]]:
        """Run a fixed set of KNN queries and return the results."""
        queries = [
            "0.0, 0.0, 0.0, 0.0",
            "100.0, 100.0, 100.0, 100.0",
            "2500.0, 2500.0, 2500.0, 2500.0",
            "4999.0, 4999.0, 4999.0, 4999.0",
        ]
        return [ps.knn(q, 5) for q in queries]

    def test_bulk_incremental_compact_lifecycle(self) -> None:
        deleted_ids: set[int] = set()
        next_new_id = BULK_COUNT

        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dim=4, range_size=RANGE_SIZE, dist_func="l1")

            # --- Bulk load ---
            ps.generate(count=BULK_COUNT, start_id=0, pattern=0)
            self.assertGreater(self.count_files(".data"), 0)
            self.progress(f"Bulk loaded {BULK_COUNT} vectors, {self.count_files('.data')} data files")

            ref_before_updates = self._reference_queries(ps)

            # --- Incremental update rounds ---
            for round_idx in range(UPDATE_ROUNDS):
                # Insert new vectors.
                for i in range(UPSERTS_PER_ROUND):
                    vid = next_new_id + i
                    val = f"{vid + 0.1}"
                    ps.upsert(vid, f"{val}, {val}, {val}, {val}")
                next_new_id += UPSERTS_PER_ROUND

                # Update existing vectors (overwrite with new values).
                for i in range(UPDATES_PER_ROUND):
                    vid = round_idx * UPDATES_PER_ROUND + i
                    if vid in deleted_ids:
                        continue
                    new_val = f"{vid + 0.5}"
                    ps.upsert(vid, f"{new_val}, {new_val}, {new_val}, {new_val}")

                # Delete some vectors.
                for i in range(DELETES_PER_ROUND):
                    vid = BULK_COUNT - 1 - round_idx * DELETES_PER_ROUND - i
                    if vid < 0 or vid in deleted_ids:
                        continue
                    ps.delete(vid)
                    deleted_ids.add(vid)

                ps.merge_accumulator()
                self.progress(f"Round {round_idx}: +{UPSERTS_PER_ROUND} new, ~{UPDATES_PER_ROUND} updated, -{DELETES_PER_ROUND} deleted")

            # Some delta files should exist after incremental updates.
            delta_count_before = self.count_files(".delta")
            self.progress(f"Incremental phase done: {delta_count_before} delta files, {len(deleted_ids)} total deleted")

            # Capture pre-compaction query results.
            ref_before_compact = self._reference_queries(ps)

            # --- Full compaction ---
            ps.merge_delta()

            # No delta files should remain.
            self.assertEqual(0, self.count_files(".delta"))
            self.progress("Compaction complete, 0 delta files remain")

            # Query results must be identical after compaction.
            ref_after_compact = self._reference_queries(ps)
            self.assertEqual(ref_before_compact, ref_after_compact)
            self.progress("Post-compaction KNN matches pre-compaction")

            # Deleted vectors must not be retrievable.
            for vid in list(deleted_ids)[:10]:
                with self.assertRaises(Sketch2Error):
                    ps.get(vid)
            self.progress(f"Verified {min(10, len(deleted_ids))} deleted vectors are gone")

            # Surviving vectors must be retrievable.
            for vid in [0, 1, 2, 3, next_new_id - 1]:
                if vid not in deleted_ids:
                    vec = ps.get(vid)
                    self.assertTrue(len(vec) > 0)
            self.progress("Surviving vectors verified")

            ps.close(self.dataset_name)


if __name__ == "__main__":
    unittest.main()
