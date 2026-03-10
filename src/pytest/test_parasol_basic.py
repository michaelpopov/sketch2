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
            ps.create(self.dataset_name)
            ps.open(self.dataset_name)

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
            ps.create(self.dataset_name)
            ps.open(self.dataset_name)

            ps.upsert(10, "1.0, 2.0, 3.0, 4.0")
            ps.upsert(5, "1.0, 2.0, 3.0, 4.0")
            ps.merge_accumulator()

            self.assertTrue((self.dataset_dir / "0.data").exists())
            ps.close(self.dataset_name)
            ps.drop(self.dataset_name)

    def test_get_returns_vector_text(self) -> None:
        with Parasol(self.root) as ps:
            ps.create(self.dataset_name)
            ps.open(self.dataset_name)

            ps.upsert(42, "1.0, 2.0, 3.0, 4.0")
            ps.merge_accumulator()

            vec = ps.get(42)
            self.assertEqual("[ 1, 2, 3, 4 ]", vec)

            ps.close(self.dataset_name)
            ps.drop(self.dataset_name)

    def test_generate_sequential(self) -> None:
        with Parasol(self.root) as ps:
            ps.create(self.dataset_name)
            ps.open(self.dataset_name)

            ps.generate(count=5, start_id=10, pattern=0)

            vec = ps.get(10)
            self.assertIn("[ 10.1", vec)
            self.assertTrue((self.dataset_dir / "0.data").exists())
            ps.close(self.dataset_name)
            ps.drop(self.dataset_name)

    def test_generate_detailed(self) -> None:
        with Parasol(self.root) as ps:
            ps.create(self.dataset_name)
            ps.open(self.dataset_name)

            ps.generate(count=3, start_id=20, pattern=1)

            vec = ps.get(20)
            self.assertEqual("[ 0, 0, 0, 0 ]", vec)
            ps.close(self.dataset_name)
            ps.drop(self.dataset_name)


if __name__ == "__main__":
    unittest.main()
