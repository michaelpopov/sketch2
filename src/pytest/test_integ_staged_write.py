from __future__ import annotations

import unittest

from integ_helpers import IntegTestBase, lib_path
from sketch2_wrapper import Sketch2


class Sketch2StagedWriteIntegTest(IntegTestBase):
    def test_staged_write_then_knn(self) -> None:
        self.progress("creating dataset and writing staged vectors")
        with Sketch2(self.root, lib_path=lib_path()) as ps:
            ps.create(self.dataset_name, type_name="f32", dim=4, range_size=1000, dist_func="l1")
            ps.start_writing()
            ps.write_vector(10, "0.0, 0.0, 0.0, 0.0")
            ps.write_vector(20, "5.0, 5.0, 5.0, 5.0")
            ps.write_vector(30, "10.0, 10.0, 10.0, 10.0")
            ps.complete_writing()

            self.progress("querying nearest neighbors from staged data")
            self.assertEqual([10, 20], ps.knn("1.0, 1.0, 1.0, 1.0", 2))
            self.assertEqual("[ 5.00, 5.00, 5.00, 5.00 ]", ps.get(20))

    def test_staged_delete_updates_knn_visibility(self) -> None:
        self.progress("writing initial vectors and deleting one via staged API")
        with Sketch2(self.root, lib_path=lib_path()) as ps:
            ps.create(self.dataset_name, type_name="f32", dim=4, range_size=1000, dist_func="l1")
            ps.start_writing()
            ps.write_vector(10, "0.0, 0.0, 0.0, 0.0")
            ps.write_vector(20, "5.0, 5.0, 5.0, 5.0")
            ps.complete_writing()

            ps.start_writing()
            ps.write_deleted(10)
            ps.complete_writing()

            self.progress("verifying deleted item no longer appears in queries")
            self.assertEqual([20], ps.knn("0.0, 0.0, 0.0, 0.0", 1))
            with self.assertRaisesRegex(RuntimeError, "sk_get failed"):
                ps.get(10)

    def test_staged_abort_discards_session_and_allows_restart(self) -> None:
        self.progress("writing staged data and aborting the session")
        with Sketch2(self.root, lib_path=lib_path()) as ps:
            ps.create(self.dataset_name, type_name="f32", dim=4, range_size=1000, dist_func="l1")
            ps.start_writing()
            ps.write_vector(10, "0.0, 0.0, 0.0, 0.0")
            ps.abort_writing()

            self.progress("restarting staged write and verifying only new data is visible")
            ps.start_writing()
            ps.write_vector(20, "5.0, 5.0, 5.0, 5.0")
            ps.complete_writing()

            self.assertEqual([20], ps.knn("5.0, 5.0, 5.0, 5.0", 1))
            with self.assertRaisesRegex(RuntimeError, "sk_get failed"):
                ps.get(10)
            self.assertEqual("[ 5.00, 5.00, 5.00, 5.00 ]", ps.get(20))


if __name__ == "__main__":
    unittest.main()
