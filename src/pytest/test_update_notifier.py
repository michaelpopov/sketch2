"""Integration tests for UpdateNotifier via libsketch2.so.

Verifies that write operations (generate, upsert+merge_accumulator,
merge_delta) increment the file-backed counter in the owner lock file,
and that a sequential close/reopen cycle sees updated data.
"""

from __future__ import annotations

import struct
import unittest
from pathlib import Path

from sketch2_wrapper import Sketch2

from integ_helpers import IntegTestBase, run_subprocess

OWNER_LOCK_FILE = "sketch2.owner.lock"
COUNTER_FORMAT = "<Q"  # little-endian uint64


def read_counter(dataset_dir: Path) -> int:
    """Read the 8-byte update counter from the owner lock file."""
    lock_path = dataset_dir / OWNER_LOCK_FILE
    if not lock_path.exists():
        return -1
    data = lock_path.read_bytes()
    if len(data) < 8:
        return -1
    return struct.unpack(COUNTER_FORMAT, data[:8])[0]


class UpdateNotifierTest(IntegTestBase):
    _tmpdir_prefix = "sketch2_py_notifier_"

    def test_generate_increments_counter(self) -> None:
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name)
            # generate calls store() internally
            ps.generate(count=5, start_id=0, pattern=0)
            self.assertEqual(1, read_counter(self.dataset_dir))

            ps.generate(count=5, start_id=100, pattern=0)
            self.assertEqual(2, read_counter(self.dataset_dir))

            ps.close(self.dataset_name)

    def test_upsert_merge_accumulator_increments_counter(self) -> None:
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name)

            ps.upsert(1, "1.0, 2.0, 3.0, 4.0")
            ps.upsert(2, "5.0, 6.0, 7.0, 8.0")
            ps.merge_accumulator()
            self.assertEqual(1, read_counter(self.dataset_dir))

            ps.upsert(3, "9.0, 10.0, 11.0, 12.0")
            ps.merge_accumulator()
            self.assertEqual(2, read_counter(self.dataset_dir))

            ps.close(self.dataset_name)

    def test_merge_delta_increments_counter(self) -> None:
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name)

            # Generate enough base data, then a small update to create a delta.
            ps.generate(count=50, start_id=0, pattern=0)  # counter = 1
            ps.upsert(0, "99.0, 99.0, 99.0, 99.0")
            ps.merge_accumulator()  # counter = 2, creates delta
            self.assertTrue((self.dataset_dir / "0.delta").exists())

            ps.merge_delta()  # counter = 3
            self.assertEqual(3, read_counter(self.dataset_dir))

            ps.close(self.dataset_name)

    def test_counter_persists_across_close_reopen(self) -> None:
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name)
            ps.generate(count=5, start_id=0, pattern=0)
            self.assertEqual(1, read_counter(self.dataset_dir))

            ps.close(self.dataset_name)

            # Reopen and write more data — counter should continue from 1.
            ps.open(self.dataset_name)
            ps.generate(count=5, start_id=100, pattern=0)
            self.assertEqual(2, read_counter(self.dataset_dir))

            ps.close(self.dataset_name)

    def test_sequential_writer_reader_sees_updated_data(self) -> None:
        """Writer and reader alternate access (no concurrent handles)."""
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name)
            ps.upsert(1, "1.0, 1.0, 1.0, 1.0")
            ps.upsert(2, "10.0, 10.0, 10.0, 10.0")
            ps.merge_accumulator()
            ps.close(self.dataset_name)

        # Reader opens, queries, closes.
        with Sketch2(self.root) as ps:
            ps.open(self.dataset_name)
            ids = ps.knn("0.0, 0.0, 0.0, 0.0", 2)
            self.assertEqual([1, 2], ids)
            ps.close(self.dataset_name)

        # Writer reopens, adds a closer vector, closes.
        with Sketch2(self.root) as ps:
            ps.open(self.dataset_name)
            ps.upsert(3, "0.1, 0.1, 0.1, 0.1")
            ps.merge_accumulator()
            ps.close(self.dataset_name)

        # Reader reopens — should see the new vector as nearest.
        with Sketch2(self.root) as ps:
            ps.open(self.dataset_name)
            ids = ps.knn("0.0, 0.0, 0.0, 0.0", 2)
            self.assertEqual([3, 1], ids)
            ps.close(self.dataset_name)

    def test_no_counter_file_before_first_write(self) -> None:
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name)
            # Counter file should not contain a valid counter yet
            # (the file exists as the owner lock but may be empty).
            counter = read_counter(self.dataset_dir)
            self.assertIn(counter, (-1, 0))

            ps.close(self.dataset_name)


class UpdateNotifierCrossProcessTest(IntegTestBase):
    """Tests that the notifier counter is visible across OS processes."""

    _tmpdir_prefix = "sketch2_py_xproc_"

    def test_child_process_sees_counter_after_parent_writes(self) -> None:
        # Parent creates dataset and writes data.
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name)
            ps.generate(count=5, start_id=0, pattern=0)
            ps.close(self.dataset_name)

        # Child process reads the counter.
        script = f"""
import struct
from pathlib import Path
data = Path({str(self.dataset_dir)!r}, {OWNER_LOCK_FILE!r}).read_bytes()
counter = struct.unpack('<Q', data[:8])[0]
assert counter == 1, f"expected 1, got {{counter}}"
"""
        result = run_subprocess(script)
        self.assert_subprocess_ok(result, "child counter check")

    def test_child_process_reads_updated_data_after_parent_writes(self) -> None:
        # Parent: create dataset with initial data.
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name)
            ps.upsert(1, "1.0, 1.0, 1.0, 1.0")
            ps.upsert(2, "10.0, 10.0, 10.0, 10.0")
            ps.merge_accumulator()
            ps.close(self.dataset_name)

        # Child: open dataset and run KNN.
        script = f"""
from sketch2_wrapper import Sketch2
with Sketch2({str(self.root)!r}) as ps:
    ps.open({self.dataset_name!r})
    ids = ps.knn("0.0, 0.0, 0.0, 0.0", 2)
    assert ids == [1, 2], f"expected [1, 2], got {{ids}}"
    ps.close({self.dataset_name!r})
"""
        result = run_subprocess(script)
        self.assert_subprocess_ok(result, "child KNN query")

        # Parent: add a vector closer to the query.
        with Sketch2(self.root) as ps:
            ps.open(self.dataset_name)
            ps.upsert(3, "0.1, 0.1, 0.1, 0.1")
            ps.merge_accumulator()
            ps.close(self.dataset_name)

        # Child: reopen and verify updated results.
        script2 = f"""
from sketch2_wrapper import Sketch2
with Sketch2({str(self.root)!r}) as ps:
    ps.open({self.dataset_name!r})
    ids = ps.knn("0.0, 0.0, 0.0, 0.0", 2)
    assert ids == [3, 1], f"expected [3, 1], got {{ids}}"
    ps.close({self.dataset_name!r})
"""
        result2 = run_subprocess(script2)
        self.assert_subprocess_ok(result2, "child KNN query after update")

    def test_child_writer_counter_visible_to_parent(self) -> None:
        # Parent creates dataset.
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name)
            ps.close(self.dataset_name)

        # Child writes data.
        script = f"""
from sketch2_wrapper import Sketch2
with Sketch2({str(self.root)!r}) as ps:
    ps.open({self.dataset_name!r})
    ps.upsert(1, "1.0, 2.0, 3.0, 4.0")
    ps.merge_accumulator()
    ps.close({self.dataset_name!r})
"""
        result = run_subprocess(script)
        self.assert_subprocess_ok(result, "child writer")

        # Parent reads counter — should be 1.
        self.assertEqual(1, read_counter(self.dataset_dir))

        # Parent opens and reads the data written by child.
        with Sketch2(self.root) as ps:
            ps.open(self.dataset_name)
            vec = ps.get(1)
            self.assertEqual("[ 1, 2, 3, 4 ]", vec)
            ps.close(self.dataset_name)


if __name__ == "__main__":
    unittest.main()
