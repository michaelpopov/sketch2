from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from sketch2_wrapper import Sketch2, Sketch2Error


class Sketch2ErrorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="sketch2_py_errors_"))
        self.dataset_name = "dataset"
        self.dataset_dir = self.root / self.dataset_name

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_open_fails_without_metadata(self) -> None:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        with Sketch2(self.root) as ps:
            with self.assertRaises(Sketch2Error):
                ps.open(self.dataset_name)

    def test_merge_delta_without_input_is_allowed(self) -> None:
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dist_func="l1")
            ps.merge_delta()
            ps.close(self.dataset_name)
            ps.drop(self.dataset_name)

    def test_open_fails_when_dataset_is_already_open(self) -> None:
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dist_func="l1")
            with self.assertRaises(Sketch2Error):
                ps.open(self.dataset_name)
            ps.close(self.dataset_name)
            ps.drop(self.dataset_name)

    def test_get_fails_for_missing_id(self) -> None:
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dist_func="l1")
            ps.upsert(1, "1.0, 2.0, 3.0, 4.0")
            ps.merge_accumulator()

            with self.assertRaises(Sketch2Error):
                ps.get(999)

            ps.close(self.dataset_name)
            ps.drop(self.dataset_name)

    def test_get_rejects_removed_buf_size_argument(self) -> None:
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dist_func="l1")
            ps.upsert(1, "1.0, 2.0, 3.0, 4.0")
            ps.merge_accumulator()

            with self.assertRaises(TypeError):
                ps.get(1, buf_size=128)  # type: ignore[call-arg]

            ps.close(self.dataset_name)
            ps.drop(self.dataset_name)

    def test_generate_fails_on_invalid_pattern(self) -> None:
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dist_func="l1")

            with self.assertRaises(Sketch2Error):
                ps.generate(count=10, start_id=0, pattern=7)

            ps.close(self.dataset_name)
            ps.drop(self.dataset_name)

    def test_generate_fails_on_zero_count(self) -> None:
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dist_func="l1")

            with self.assertRaises(Sketch2Error):
                ps.generate(count=0, start_id=0, pattern=0)

            ps.close(self.dataset_name)
            ps.drop(self.dataset_name)

    def test_create_fails_on_invalid_distance_function(self) -> None:
        with Sketch2(self.root) as ps:
            with self.assertRaises(Sketch2Error):
                ps.create(self.dataset_name, dist_func="cosine")


if __name__ == "__main__":
    unittest.main()
