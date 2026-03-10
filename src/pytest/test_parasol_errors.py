from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from parasol_wrapper import Parasol, ParasolError


class ParasolErrorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="sketch2_pyparasol_err_"))
        self.dataset_name = "dataset"
        self.dataset_dir = self.root / self.dataset_name

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_open_fails_without_metadata(self) -> None:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        with Parasol(self.root) as ps:
            with self.assertRaises(ParasolError):
                ps.open(self.dataset_name)

    def test_merge_delta_without_input_is_allowed(self) -> None:
        with Parasol(self.root) as ps:
            ps.create(self.dataset_name)
            ps.open(self.dataset_name)
            ps.merge_delta()
            ps.close(self.dataset_name)
            ps.drop(self.dataset_name)

    def test_open_fails_when_dataset_is_already_open(self) -> None:
        with Parasol(self.root) as ps:
            ps.create(self.dataset_name)
            ps.open(self.dataset_name)
            with self.assertRaises(ParasolError):
                ps.open(self.dataset_name)
            ps.close(self.dataset_name)
            ps.drop(self.dataset_name)

    def test_get_fails_for_missing_id(self) -> None:
        with Parasol(self.root) as ps:
            ps.create(self.dataset_name)
            ps.open(self.dataset_name)
            ps.upsert(1, "1.0, 2.0, 3.0, 4.0")
            ps.merge_accumulator()

            with self.assertRaises(ParasolError):
                ps.get(999)

            ps.close(self.dataset_name)
            ps.drop(self.dataset_name)

    def test_generate_fails_on_invalid_pattern(self) -> None:
        with Parasol(self.root) as ps:
            ps.create(self.dataset_name)
            ps.open(self.dataset_name)

            with self.assertRaises(ParasolError):
                ps.generate(count=10, start_id=0, pattern=7)

            ps.close(self.dataset_name)
            ps.drop(self.dataset_name)

    def test_generate_fails_on_zero_count(self) -> None:
        with Parasol(self.root) as ps:
            ps.create(self.dataset_name)
            ps.open(self.dataset_name)

            with self.assertRaises(ParasolError):
                ps.generate(count=0, start_id=0, pattern=0)

            ps.close(self.dataset_name)
            ps.drop(self.dataset_name)


if __name__ == "__main__":
    unittest.main()
