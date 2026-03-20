"""Integration test 10: New SQLite interfaces (bitset_agg + allowed_ids).

These tests validate extension-level SQL plumbing:
- aggregate bitset_agg(id)
- optional hidden-column constraint vlite.allowed_ids

Filtering by allowed_ids is intentionally not implemented yet, so result sets
are expected to remain unchanged when a blob is provided.
"""

from __future__ import annotations

import sqlite3
import unittest

from sketch2_wrapper import Sketch2

from integ_helpers import IntegTestBase, lib_path


class VliteInterfacesTest(IntegTestBase):
    _tmpdir_prefix = "sketch2_integ_vlite_if_"

    def _open_sqlite(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        conn.load_extension(lib_path())
        return conn

    def test_bitset_agg_builds_dense_blob(self) -> None:
        conn = self._open_sqlite()
        try:
            row = conn.execute(
                "SELECT hex(bitset_agg(id)) "
                "FROM ("
                "  SELECT 0 AS id UNION ALL "
                "  SELECT 1 UNION ALL "
                "  SELECT NULL UNION ALL "
                "  SELECT 8 UNION ALL "
                "  SELECT 8"
                ")"
            ).fetchone()
        finally:
            conn.close()

        self.assertIsNotNone(row)
        # id=0,1 -> 0b00000011 in byte 0; id=8 -> 0b00000001 in byte 1.
        self.assertEqual("0301", row[0])
        self.progress("bitset_agg dense layout verified (ids 0,1,8 -> 0301)")

    def test_bitset_agg_rejects_invalid_inputs(self) -> None:
        conn = self._open_sqlite()
        try:
            with self.assertRaises(sqlite3.Error):
                conn.execute(
                    "SELECT bitset_agg(id) FROM (SELECT -1 AS id)"
                ).fetchall()
            with self.assertRaises(sqlite3.Error):
                conn.execute(
                    "SELECT bitset_agg(id) FROM (SELECT 'oops' AS id)"
                ).fetchall()
        finally:
            conn.close()

        self.progress("bitset_agg rejects negative and non-integer ids")

    def test_vlite_allowed_ids_is_optional_and_currently_noop(self) -> None:
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dim=4, range_size=1000, dist_func="l1")
            ps.upsert(0, "0.0, 0.0, 0.0, 0.0")
            ps.upsert(1, "1.0, 1.0, 1.0, 1.0")
            ps.upsert(2, "2.0, 2.0, 2.0, 2.0")
            ps.merge_accumulator()
            ps.close(self.dataset_name)

        conn = self._open_sqlite()
        try:
            conn.execute(f"CREATE VIRTUAL TABLE nn USING vlite('{self.ini_path()}')")

            baseline_rows = conn.execute(
                "SELECT id, distance FROM nn "
                "WHERE match_expr MATCH '2.1, 2.1, 2.1, 2.1' AND k = 3 "
                "ORDER BY distance"
            ).fetchall()

            blob_rows = conn.execute(
                "SELECT id, distance FROM nn "
                "WHERE match_expr MATCH '2.1, 2.1, 2.1, 2.1' AND k = 3 "
                "AND allowed_ids = (SELECT bitset_agg(id) FROM (SELECT 0 AS id)) "
                "ORDER BY distance"
            ).fetchall()

            null_rows = conn.execute(
                "SELECT id, distance FROM nn "
                "WHERE match_expr MATCH '2.1, 2.1, 2.1, 2.1' AND k = 3 "
                "AND allowed_ids = CAST(NULL AS BLOB) "
                "ORDER BY distance"
            ).fetchall()
        finally:
            conn.close()

        self.assertEqual(baseline_rows, blob_rows)
        self.assertEqual(baseline_rows, null_rows)
        self.progress("allowed_ids accepted (BLOB/NULL), current behavior remains unfiltered")

    def test_vlite_allowed_ids_rejects_non_blob(self) -> None:
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dim=4, range_size=1000, dist_func="l1")
            ps.upsert(10, "10.0, 10.0, 10.0, 10.0")
            ps.merge_accumulator()
            ps.close(self.dataset_name)

        conn = self._open_sqlite()
        try:
            conn.execute(f"CREATE VIRTUAL TABLE nn USING vlite('{self.ini_path()}')")
            with self.assertRaises(sqlite3.Error):
                conn.execute(
                    "SELECT id FROM nn "
                    "WHERE query = '10.0, 10.0, 10.0, 10.0' AND k = 1 "
                    "AND allowed_ids = 'not_a_blob'"
                ).fetchall()
        finally:
            conn.close()

        self.progress("allowed_ids rejects non-BLOB/non-NULL values")


if __name__ == "__main__":
    unittest.main()
