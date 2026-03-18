"""Integration test 7: Delete-Heavy Lifecycle.

Insert a large dataset, then delete the majority of vectors in stages,
verifying that deleted vectors vanish from query results and that
surviving vectors remain correctly queryable.  Uses both the C API
wrapper and the SQLite virtual table.
"""

from __future__ import annotations

import sqlite3
import unittest

from sketch2_wrapper import Sketch2, Sketch2Error

from integ_helpers import IntegTestBase, lib_path

TOTAL_VECTORS = 2000
RANGE_SIZE = 1000
DIM = 4


class DeleteHeavyTest(IntegTestBase):
    _tmpdir_prefix = "sketch2_integ_delete_"

    def test_progressive_deletes_remove_vectors_from_knn(self) -> None:
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dim=DIM, range_size=RANGE_SIZE, dist_func="l1")
            ps.generate(count=TOTAL_VECTORS, start_id=0, pattern=0)
            self.progress(f"Generated {TOTAL_VECTORS} vectors")

            # --- Round 1: delete ids 0..499 ---
            deleted: set[int] = set()
            for i in range(500):
                ps.delete(i)
                deleted.add(i)
            ps.merge_accumulator()

            ids = ps.knn("0.0, 0.0, 0.0, 0.0", 10)
            for vid in ids:
                self.assertNotIn(vid, deleted,
                    f"Deleted id {vid} appeared in KNN after round 1")
            self.progress(f"Round 1: deleted 500, KNN near origin: {ids}")

            # --- Round 2: delete ids 500..999 ---
            for i in range(500, 1000):
                ps.delete(i)
                deleted.add(i)
            ps.merge_accumulator()

            ids = ps.knn("500.0, 500.0, 500.0, 500.0", 10)
            for vid in ids:
                self.assertNotIn(vid, deleted,
                    f"Deleted id {vid} appeared in KNN after round 2")
            self.progress(f"Round 2: deleted 500 more, KNN near 500: {ids}")

            # --- Compact ---
            ps.merge_delta()

            ids = ps.knn("0.0, 0.0, 0.0, 0.0", 10)
            for vid in ids:
                self.assertNotIn(vid, deleted,
                    f"Deleted id {vid} appeared in KNN after compaction")
            self.progress(f"Post-compaction KNN near origin: {ids}")

            ps.close(self.dataset_name)

    def test_get_fails_for_deleted_vectors(self) -> None:
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dim=DIM, range_size=RANGE_SIZE, dist_func="l1")
            ps.generate(count=TOTAL_VECTORS, start_id=0, pattern=0)

            to_delete = list(range(0, 200))
            for vid in to_delete:
                ps.delete(vid)
            ps.merge_accumulator()
            self.progress(f"Deleted {len(to_delete)} vectors")

            for vid in to_delete[:20]:
                with self.assertRaises(Sketch2Error,
                        msg=f"get({vid}) should fail after deletion"):
                    ps.get(vid)
            self.progress("Confirmed get() raises for 20 deleted ids")

            for vid in [200, 500, 1000, 1999]:
                vec = ps.get(vid)
                self.assertTrue(len(vec) > 0, f"get({vid}) returned empty")
            self.progress("Surviving vectors still accessible")

            ps.close(self.dataset_name)

    def test_delete_majority_leaves_correct_survivors(self) -> None:
        """Delete 90% of vectors, verify the remaining 10% are queryable."""
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dim=DIM, range_size=RANGE_SIZE, dist_func="l1")
            ps.generate(count=TOTAL_VECTORS, start_id=0, pattern=0)

            survivors = set(range(0, TOTAL_VECTORS, 10))
            for i in range(TOTAL_VECTORS):
                if i not in survivors:
                    ps.delete(i)
            ps.merge_accumulator()
            ps.merge_delta()
            self.progress(f"Deleted {TOTAL_VECTORS - len(survivors)}/{TOTAL_VECTORS} vectors, {len(survivors)} survivors")

            ids = ps.knn("0.0, 0.0, 0.0, 0.0", 20)
            for vid in ids:
                self.assertIn(vid, survivors,
                    f"Non-survivor {vid} appeared in KNN results")
            self.progress(f"KNN returns only survivors: {ids[:5]}...")

            for vid in sorted(survivors)[:50]:
                vec = ps.get(vid)
                self.assertTrue(len(vec) > 0, f"Survivor {vid} not retrievable")
            self.progress("First 50 survivors verified retrievable")

            ps.close(self.dataset_name)

    def test_delete_via_sqlite_sees_no_deleted_ids(self) -> None:
        ini_path = self.ini_path()

        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dim=DIM, range_size=RANGE_SIZE, dist_func="l1")
            ps.generate(count=500, start_id=0, pattern=0)

            deleted = set(range(0, 100))
            for vid in deleted:
                ps.delete(vid)
            ps.merge_accumulator()
            ps.close(self.dataset_name)
        self.progress("Deleted 100/500 vectors, querying via SQLite...")

        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        conn.load_extension(lib_path())
        conn.execute(f"CREATE VIRTUAL TABLE nn USING vlite('{ini_path}')")

        rows = conn.execute(
            "SELECT id FROM nn "
            "WHERE query = '0.0, 0.0, 0.0, 0.0' AND k = 20 "
            "ORDER BY distance"
        ).fetchall()
        conn.close()

        ids = [row[0] for row in rows]
        for row in rows:
            self.assertNotIn(row[0], deleted,
                f"Deleted id {row[0]} appeared in SQLite KNN")
        self.progress(f"SQLite KNN returned {len(ids)} results, no deleted ids: {ids[:5]}...")


if __name__ == "__main__":
    unittest.main()
