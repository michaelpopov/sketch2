from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from parasol_wrapper import Parasol


class ParasolBasicTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="sketch2_pyparasol_"))
        self.dataset_dir = self.root / "dataset"

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_create_open_add_load_and_knn(self) -> None:
        with Parasol() as ps:
            ps.create(self.dataset_dir)
            ps.open(self.dataset_dir)

            ps.add(1, "0.0, 0.0, 0.0, 0.0")
            ps.add(2, "10.0, 10.0, 10.0, 10.0")
            ps.add(3, "1.0, 1.0, 1.0, 1.0")
            ps.load()

            ids = ps.knn("0.0, 0.0, 0.0, 0.0", 2)
            self.assertEqual([1, 3], ids)

            self.assertTrue((self.dataset_dir / "0.data").exists())

            ps.drop()

        self.assertFalse(self.dataset_dir.exists())

    def test_load_accepts_unsorted_ids(self) -> None:
        with Parasol() as ps:
            ps.create(self.dataset_dir)
            ps.open(self.dataset_dir)

            ps.add(10, "1.0, 2.0, 3.0, 4.0")
            ps.add(5, "1.0, 2.0, 3.0, 4.0")
            ps.load()

            self.assertTrue((self.dataset_dir / "0.data").exists())
            ps.drop()


if __name__ == "__main__":
    unittest.main()
