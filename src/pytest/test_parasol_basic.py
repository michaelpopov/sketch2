from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from parasol_wrapper import Parasol


class ParasolBasicTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="sketch2_pyparasol_"))
        self.dataset_name = "dataset"
        self.dataset_dir = self.root / self.dataset_name

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_create_open_upsert_merge_accumulator_and_knn(self) -> None:
        with Parasol(self.root) as ps:
            ps.create(self.dataset_name, dist_func="l1")

            ps.upsert(1, "0.0, 0.0, 0.0, 0.0")
            ps.upsert(2, "10.0, 10.0, 10.0, 10.0")
            ps.upsert(3, "1.0, 1.0, 1.0, 1.0")
            ps.merge_accumulator()

            ids = ps.knn("0.0, 0.0, 0.0, 0.0", 2)
            self.assertEqual([1, 3], ids)

            self.assertTrue((self.dataset_dir / "0.data").exists())

            ps.close(self.dataset_name)
            ps.drop(self.dataset_name)

        self.assertFalse(self.dataset_dir.exists())

    def test_load_accepts_unsorted_ids(self) -> None:
        with Parasol(self.root) as ps:
            ps.create(self.dataset_name, dist_func="l1")

            ps.upsert(10, "1.0, 2.0, 3.0, 4.0")
            ps.upsert(5, "1.0, 2.0, 3.0, 4.0")
            ps.merge_accumulator()

            self.assertTrue((self.dataset_dir / "0.data").exists())
            ps.close(self.dataset_name)
            ps.drop(self.dataset_name)

    def test_get_returns_vector_text(self) -> None:
        with Parasol(self.root) as ps:
            ps.create(self.dataset_name, dist_func="l1")

            ps.upsert(42, "1.0, 2.0, 3.0, 4.0")
            ps.merge_accumulator()

            vec = ps.get(42)
            self.assertEqual("[ 1, 2, 3, 4 ]", vec)

            ps.close(self.dataset_name)
            ps.drop(self.dataset_name)

    def test_generate_sequential(self) -> None:
        with Parasol(self.root) as ps:
            ps.create(self.dataset_name, dist_func="l1")

            ps.generate(count=5, start_id=10, pattern=0)

            vec = ps.get(10)
            self.assertIn("[ 10.1", vec)
            self.assertTrue((self.dataset_dir / "0.data").exists())
            ps.close(self.dataset_name)
            ps.drop(self.dataset_name)

    def test_generate_detailed(self) -> None:
        with Parasol(self.root) as ps:
            ps.create(self.dataset_name, dist_func="l1")

            ps.generate(count=3, start_id=20, pattern=1)

            vec = ps.get(20)
            self.assertEqual("[ 0, 0, 0, 0 ]", vec)
            ps.close(self.dataset_name)
            ps.drop(self.dataset_name)

    def test_load_file_bulk_imports_vectors(self) -> None:
        input_path = self.root / "input.txt"
        input_path.write_text(
            "f32,4\n"
            "10 : [ 1.00, 2.00, 3.00, 4.00 ]\n"
            "20 : [ 5.00, 6.00, 7.00, 8.00 ]\n",
            encoding="utf-8",
        )

        with Parasol(self.root) as ps:
            ps.create(self.dataset_name, dist_func="l1")

            ps.load_file(input_path)
            ps.merge_accumulator()

            self.assertEqual("[ 1, 2, 3, 4 ]", ps.get(10))
            self.assertEqual("[ 5, 6, 7, 8 ]", ps.get(20))

            ps.close(self.dataset_name)
            ps.drop(self.dataset_name)

    def test_create_with_l2_writes_distance_function_to_ini(self) -> None:
        with Parasol(self.root) as ps:
            ps.create(self.dataset_name, dist_func="l2")

            ini = (self.root / f"{self.dataset_name}.ini").read_text()
            self.assertIn("dist_func=l2\n", ini)

            ps.close(self.dataset_name)
            ps.drop(self.dataset_name)

    def test_create_with_cos_supports_knn(self) -> None:
        with Parasol(self.root) as ps:
            ps.create(self.dataset_name, dist_func="cos")

            ps.upsert(10, "100.0, 1.0, 0.0, 0.0")
            ps.upsert(20, "1.0, 1.0, 0.0, 0.0")
            ps.upsert(30, "-1.0, 0.0, 0.0, 0.0")
            ps.merge_accumulator()

            ini = (self.root / f"{self.dataset_name}.ini").read_text()
            self.assertIn("dist_func=cos\n", ini)
            self.assertEqual([10, 20, 30], ps.knn("1.0, 0.0, 0.0, 0.0", 3))

            ps.close(self.dataset_name)
            ps.drop(self.dataset_name)


if __name__ == "__main__":
    unittest.main()
