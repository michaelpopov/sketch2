"""Integration test 8: Accumulator Pressure (Auto-Flush).

Configure a small accumulator size.  Rapidly insert vectors exceeding the
accumulator capacity, triggering automatic flushes.  Verify no data loss
and correct query results.
"""

from __future__ import annotations

import unittest

from sketch2_wrapper import Sketch2

from integ_helpers import IntegTestBase

DIM = 4
# 16 bytes per f32 vector (4 * 4).  A 256-byte accumulator can hold roughly
# 6-8 vectors after alignment/overhead, so 200 upserts trigger many flushes.
SMALL_ACCUMULATOR_SIZE = 256
VECTOR_COUNT = 200
RANGE_SIZE = 10000
class AccumulatorPressureTest(IntegTestBase):
    _tmpdir_prefix = "sketch2_integ_accpres_"

    def _create_dataset_with_small_accumulator(self, ps: Sketch2) -> None:
        """Create dataset and patch the ini to use a small accumulator."""
        ps.create(self.dataset_name, dim=DIM, range_size=RANGE_SIZE, dist_func="l1")
        ps.close(self.dataset_name)

        # Patch the ini file to add accumulator_size.
        ini_path = self.root / f"{self.dataset_name}.ini"
        content = ini_path.read_text()
        content += f"accumulator_size={SMALL_ACCUMULATOR_SIZE}\n"
        ini_path.write_text(content)

        ps.open(self.dataset_name)

    def test_rapid_upserts_trigger_auto_flush(self) -> None:
        with Sketch2(self.root) as ps:
            self._create_dataset_with_small_accumulator(ps)
            self.progress(f"Upserting {VECTOR_COUNT} vectors with small accumulator...")

            for i in range(VECTOR_COUNT):
                val = f"{float(i)}"
                ps.upsert(i, f"{val}, {val}, {val}, {val}")

            # Auto-flush should have created data/delta files.
            file_count = self.count_files(".data") + self.count_files(".delta")
            self.assertGreater(file_count, 0,
                "Expected auto-flush to create files")
            self.progress(f"Auto-flush created {file_count} files")

            ps.close(self.dataset_name)

    def test_all_vectors_retrievable_after_pressure(self) -> None:
        with Sketch2(self.root) as ps:
            self._create_dataset_with_small_accumulator(ps)
            self.progress(f"Upserting {VECTOR_COUNT} vectors under pressure...")

            for i in range(VECTOR_COUNT):
                val = f"{float(i)}"
                ps.upsert(i, f"{val}, {val}, {val}, {val}")

            # Flush any remaining accumulator data.
            ps.merge_accumulator()
            self.progress("Merge complete, verifying all vectors retrievable...")

            # Every vector must be retrievable.
            for i in range(VECTOR_COUNT):
                vec = ps.get(i)
                self.assertTrue(len(vec) > 0, f"Vector {i} missing after pressure")
            self.progress(f"All {VECTOR_COUNT} vectors verified")

            ps.close(self.dataset_name)

    def test_knn_correct_after_pressure(self) -> None:
        with Sketch2(self.root) as ps:
            self._create_dataset_with_small_accumulator(ps)
            self.progress(f"Upserting {VECTOR_COUNT} vectors under pressure...")

            for i in range(VECTOR_COUNT):
                val = f"{float(i)}"
                ps.upsert(i, f"{val}, {val}, {val}, {val}")
            ps.merge_accumulator()

            ids = ps.knn("0.0, 0.0, 0.0, 0.0", 5)
            self.assertEqual([0, 1, 2, 3, 4], ids)
            self.progress(f"KNN near origin: {ids}")

            ids = ps.knn("199.0, 199.0, 199.0, 199.0", 3)
            self.assertEqual([199, 198, 197], ids)
            self.progress(f"KNN near 199: {ids}")

            ps.close(self.dataset_name)

    def test_compaction_after_pressure(self) -> None:
        with Sketch2(self.root) as ps:
            self._create_dataset_with_small_accumulator(ps)
            self.progress(f"Upserting {VECTOR_COUNT} vectors under pressure...")

            for i in range(VECTOR_COUNT):
                val = f"{float(i)}"
                ps.upsert(i, f"{val}, {val}, {val}, {val}")
            ps.merge_accumulator()

            pre_compact_ids = ps.knn("50.0, 50.0, 50.0, 50.0", 10)
            self.progress(f"Pre-compaction KNN: {pre_compact_ids}")

            ps.merge_delta()
            self.assertEqual(0, self.count_files(".delta"))
            self.progress("Compaction complete, 0 delta files remain")

            post_compact_ids = ps.knn("50.0, 50.0, 50.0, 50.0", 10)
            self.assertEqual(pre_compact_ids, post_compact_ids)
            self.progress("Post-compaction KNN matches pre-compaction")

            # Verify all vectors survived compaction.
            for i in range(VECTOR_COUNT):
                vec = ps.get(i)
                self.assertTrue(len(vec) > 0, f"Vector {i} missing after compaction")
            self.progress(f"All {VECTOR_COUNT} vectors survived compaction")

            ps.close(self.dataset_name)


if __name__ == "__main__":
    unittest.main()
