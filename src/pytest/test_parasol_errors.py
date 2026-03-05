from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from parasol_wrapper import Parasol, ParasolError


class ParasolErrorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="sketch2_pyparasol_err_"))
        self.dataset_dir = self.root / "dataset"

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_open_fails_without_metadata(self) -> None:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        with Parasol() as ps:
            with self.assertRaises(ParasolError):
                ps.open(self.dataset_dir)

    def test_load_fails_without_input(self) -> None:
        with Parasol() as ps:
            ps.create(self.dataset_dir)
            ps.open(self.dataset_dir)
            with self.assertRaises(ParasolError):
                ps.load()
            ps.drop()

    def test_load_twice_second_fails(self) -> None:
        with Parasol() as ps:
            ps.create(self.dataset_dir)
            ps.open(self.dataset_dir)
            ps.add(1, "1.0, 2.0, 3.0, 4.0")
            ps.load()
            with self.assertRaises(ParasolError):
                ps.load()
            ps.drop()


if __name__ == "__main__":
    unittest.main()
