"""Integration test 4: Multi-Range Distribution and Correctness.

Dataset with small range_size spanning many file ranges.  Verifies correct
file distribution, selective delta creation, and cross-range KNN.  Uses
both the C API wrapper and SQLite virtual table.
"""

from __future__ import annotations

import sqlite3
import unittest

from sketch2_wrapper import Sketch2

from integ_helpers import IntegTestBase, lib_path

RANGE_SIZE = 100
VECTOR_COUNT = 5000  # 50 ranges


class MultiRangeTest(IntegTestBase):
    _tmpdir_prefix = "sketch2_integ_mrange_"

    def test_file_distribution_across_ranges(self) -> None:
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dim=4, range_size=RANGE_SIZE, dist_func="l1")
            ps.generate(count=VECTOR_COUNT, start_id=0, pattern=0)

            # Should have 50 .data files (ids 0-99 -> file 0, 100-199 -> file 1, etc.)
            expected_files = VECTOR_COUNT // RANGE_SIZE
            self.assertEqual(expected_files, self.count_files(".data"))
            self.progress(f"Generated {VECTOR_COUNT} vectors, verified {expected_files} data files")

            ps.close(self.dataset_name)

    def test_selective_delta_creation(self) -> None:
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dim=4, range_size=RANGE_SIZE, dist_func="l1")
            ps.generate(count=VECTOR_COUNT, start_id=0, pattern=0)
            self.progress(f"Seeded {VECTOR_COUNT} vectors across {VECTOR_COUNT // RANGE_SIZE} ranges")

            # Update vectors in only 3 specific ranges.
            touched_ranges = {0, 25, 49}  # file ids
            for file_id in touched_ranges:
                vid = file_id * RANGE_SIZE + 5  # pick one vector per range
                ps.upsert(vid, "99.0, 99.0, 99.0, 99.0")

            ps.merge_accumulator()

            # Delta files should only exist for touched ranges.
            delta_files = set()
            for p in self.dataset_dir.glob("*.delta"):
                delta_files.add(int(p.stem))
            self.assertEqual(touched_ranges, delta_files)
            self.progress(f"Updated 3 vectors, verified exactly 3 delta files: {delta_files}")

            ps.close(self.dataset_name)

    def test_cross_range_knn(self) -> None:
        """KNN query whose best results come from different file ranges."""
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dim=4, range_size=RANGE_SIZE, dist_func="l1")
            ps.generate(count=VECTOR_COUNT, start_id=0, pattern=0)

            # Query near the boundary of range 0 and range 1.
            # id=99 has vector ~[99.1, ...], id=100 has vector ~[100.1, ...]
            # Query at 99.5 should return ids from both ranges.
            ids = ps.knn("99.5, 99.5, 99.5, 99.5", 4)
            # Expected: 99 (dist=4*0.4=1.6) and 100 (dist=4*0.6=2.4) are closest.
            self.assertIn(99, ids[:2])
            self.assertIn(100, ids[:2])
            self.progress(f"Cross-range KNN verified: results {ids} span multiple files")

            ps.close(self.dataset_name)

    def test_merge_delta_removes_all_deltas(self) -> None:
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dim=4, range_size=RANGE_SIZE, dist_func="l1")
            ps.generate(count=VECTOR_COUNT, start_id=0, pattern=0)

            # Create deltas in multiple ranges.
            for file_id in range(0, 50, 5):
                vid = file_id * RANGE_SIZE
                ps.upsert(vid, "77.0, 77.0, 77.0, 77.0")
            ps.merge_accumulator()
            self.assertGreater(self.count_files(".delta"), 0)
            self.progress(f"Created {self.count_files('.delta')} delta files")

            # Capture pre-merge KNN results.
            pre_merge = ps.knn("77.0, 77.0, 77.0, 77.0", 5)

            ps.merge_delta()
            self.assertEqual(0, self.count_files(".delta"))
            self.progress("Compaction complete, 0 delta files remain")

            # Post-merge results must match.
            post_merge = ps.knn("77.0, 77.0, 77.0, 77.0", 5)
            self.assertEqual(pre_merge, post_merge)
            self.progress("Post-compaction KNN matches pre-compaction")

            ps.close(self.dataset_name)

    def test_cross_range_knn_via_sqlite(self) -> None:
        """Same cross-range test using the SQLite virtual table."""
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dim=4, range_size=RANGE_SIZE, dist_func="l1")
            ps.generate(count=VECTOR_COUNT, start_id=0, pattern=0)
            ps.close(self.dataset_name)
        self.progress(f"Dataset with {VECTOR_COUNT} vectors ready for SQLite")

        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        conn.load_extension(lib_path())

        conn.execute(
            f"CREATE VIRTUAL TABLE nn USING vlite('{self.ini_path()}')"
        )

        rows = conn.execute(
            "SELECT id, distance FROM nn "
            "WHERE query = '99.5, 99.5, 99.5, 99.5' AND k = 4 "
            "ORDER BY distance"
        ).fetchall()
        conn.close()

        ids = [row[0] for row in rows]
        self.assertEqual(4, len(ids))
        self.assertIn(99, ids[:2])
        self.assertIn(100, ids[:2])
        self.progress(f"SQLite cross-range KNN verified: {ids}")



if __name__ == "__main__":
    unittest.main()
