from __future__ import annotations

import csv
import sqlite3
import unittest
from pathlib import Path

from integ_helpers import IntegTestBase, lib_path
from sketch2_test_vectors import find_library, fmt_typed_vector, l2_distance_sq, native_sequential_vector
from sketch2_wrapper import Sketch2


DIM = 4
COUNT = 20
START_ID = 0


def sqlite_extension_path() -> Path:
    return find_library()


def metadata_values(item_id: int) -> tuple[int, int, int, str]:
    aaa = item_id % 2
    bbb = item_id % 5
    ccc = item_id % 10
    return aaa, bbb, ccc, f"aaa={aaa}, bbb={bbb}, ccc={ccc}"


def query_vector(query_value: float) -> str:
    return fmt_typed_vector([query_value] * DIM, "f32")


def expected_knn_rows(
    query_value: float,
    k: int,
    *,
    allowed_ids: set[int] | None = None,
    start_id: int = START_ID,
    count: int = COUNT,
) -> list[tuple[int, float]]:
    query = [query_value] * DIM
    rows: list[tuple[int, float]] = []
    for item_id in range(start_id, start_id + count):
        if allowed_ids is not None and item_id not in allowed_ids:
            continue
        score = l2_distance_sq(query, native_sequential_vector(item_id, DIM, "f32"))
        rows.append((item_id, score))
    rows.sort(key=lambda row: (row[1], row[0]))
    return rows[:k]


def open_sqlite_with_extension() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.enable_load_extension(True)
    con.load_extension(str(sqlite_extension_path()))
    return con


def create_metadata_table(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE metadata (
            id INTEGER PRIMARY KEY,
            aaa INTEGER NOT NULL,
            bbb INTEGER NOT NULL,
            ccc INTEGER NOT NULL,
            text TEXT NOT NULL
        )
        """
    )


def load_metadata_csv_into_table(con: sqlite3.Connection, csv_path: Path) -> None:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    con.executemany(
        "INSERT INTO metadata(id, aaa, bbb, ccc, text) VALUES (?, ?, ?, ?, ?)",
        [
            (
                int(row["id"]),
                int(row["aaa"]),
                int(row["bbb"]),
                int(row["ccc"]),
                row["text"],
            )
            for row in rows
        ],
    )


def create_vlite_table(con: sqlite3.Connection, root: Path, dataset_name: str) -> None:
    root_sql = str(root).replace("'", "''")
    dataset_name_sql = dataset_name.replace("'", "''")
    con.execute(f"CREATE VIRTUAL TABLE nn USING vlite('{root_sql}', '{dataset_name_sql}')")


def run_basic_knn_query(con: sqlite3.Connection, query_vec: str, k: int) -> list[tuple[int, float]]:
    return [
        (int(row[0]), float(row[1]))
        for row in con.execute(
            "SELECT id, score FROM nn WHERE query = ? AND k = ? ORDER BY score, id",
            (query_vec, k),
        )
    ]


def run_match_knn_query(con: sqlite3.Connection, query_vec: str, k: int) -> list[tuple[int, float]]:
    return [
        (int(row[0]), float(row[1]))
        for row in con.execute(
            "SELECT id, score FROM nn WHERE match_expr MATCH ? AND k = ? ORDER BY score, id",
            (query_vec, k),
        )
    ]


def run_join_query(con: sqlite3.Connection, query_vec: str, k: int) -> list[tuple[int, float, int, int, int, str]]:
    return [
        (int(row[0]), float(row[1]), int(row[2]), int(row[3]), int(row[4]), str(row[5]))
        for row in con.execute(
            """
            SELECT n.id, n.score, m.aaa, m.bbb, m.ccc, m.text
            FROM nn AS n
            JOIN metadata AS m ON m.id = n.id
            WHERE n.query = ? AND n.k = ?
            ORDER BY n.score, n.id
            """,
            (query_vec, k),
        )
    ]


def run_post_filtered_join_query(
    con: sqlite3.Connection,
    query_vec: str,
    k: int,
    aaa: int,
) -> list[tuple[int, float, int, int, int, str]]:
    return [
        (int(row[0]), float(row[1]), int(row[2]), int(row[3]), int(row[4]), str(row[5]))
        for row in con.execute(
            """
            SELECT n.id, n.score, m.aaa, m.bbb, m.ccc, m.text
            FROM nn AS n
            JOIN metadata AS m ON m.id = n.id
            WHERE n.query = ? AND n.k = ? AND m.aaa = ?
            ORDER BY n.score, n.id
            """,
            (query_vec, k, aaa),
        )
    ]


def run_pushdown_query(
    con: sqlite3.Connection,
    query_vec: str,
    k: int,
    metadata_predicate_sql: str,
) -> list[tuple[int, float]]:
    sql = f"""
        SELECT n.id, n.score
        FROM nn AS n
        WHERE n.query = ? AND n.k = ?
          AND n.allowed_ids = (
                SELECT bitset_agg(id)
                FROM (
                    SELECT id
                    FROM metadata
                    WHERE {metadata_predicate_sql}
                    ORDER BY id
                )
          )
        ORDER BY n.score, n.id
    """
    return [(int(row[0]), float(row[1])) for row in con.execute(sql, (query_vec, k))]


def run_pushdown_join_query(
    con: sqlite3.Connection,
    query_vec: str,
    k: int,
    metadata_predicate_sql: str,
) -> list[tuple[int, float, int, int, int, str]]:
    sql = f"""
        SELECT n.id, n.score, m.aaa, m.bbb, m.ccc, m.text
        FROM nn AS n
        JOIN metadata AS m ON m.id = n.id
        WHERE n.query = ? AND n.k = ?
          AND n.allowed_ids = (
                SELECT bitset_agg(id)
                FROM (
                    SELECT id
                    FROM metadata
                    WHERE {metadata_predicate_sql}
                    ORDER BY id
                )
          )
        ORDER BY n.score, n.id
    """
    return [
        (int(row[0]), float(row[1]), int(row[2]), int(row[3]), int(row[4]), str(row[5]))
        for row in con.execute(sql, (query_vec, k))
    ]


class Sketch2SqliteMetadataIntegTest(IntegTestBase):
    def prepare_case(self, *, count: int = COUNT, start_id: int = START_ID) -> tuple[sqlite3.Connection, Path]:
        input_path = self.root / "dataset.input"
        metadata_csv_path = self.root / "metadata.csv"

        with Sketch2(self.root, lib_path=lib_path()) as ps:
            ps.create(self.dataset_name, type_name="f32", dim=DIM, range_size=1000, dist_func="l2")
            ps.generate_test_data(input_path, count=count, start_id=start_id, binary=True)
            ps.generate_test_metadata(metadata_csv_path, count=count, start_id=start_id)

        con = open_sqlite_with_extension()
        create_metadata_table(con)
        load_metadata_csv_into_table(con, metadata_csv_path)
        create_vlite_table(con, self.root, self.dataset_name)

        loaded_count = con.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
        self.assertEqual(count, loaded_count)
        return con, metadata_csv_path

    def assert_knn_rows_equal(
        self,
        actual: list[tuple[int, float]],
        expected: list[tuple[int, float]],
    ) -> None:
        self.assertEqual([item_id for item_id, _ in expected], [item_id for item_id, _ in actual])
        self.assertEqual(len(expected), len(actual))
        for (_, expected_score), (_, actual_score) in zip(expected, actual):
            self.assertAlmostEqual(expected_score, actual_score, places=4)

    def assert_join_rows_match_ids(
        self,
        actual: list[tuple[int, float, int, int, int, str]],
        expected_ids: list[int],
        query_value: float,
    ) -> None:
        self.assertEqual(expected_ids, [row[0] for row in actual])
        expected_scores = {
            item_id: l2_distance_sq([query_value] * DIM, native_sequential_vector(item_id, DIM, "f32"))
            for item_id in expected_ids
        }
        for item_id, score, aaa, bbb, ccc, text in actual:
            expected_aaa, expected_bbb, expected_ccc, expected_text = metadata_values(item_id)
            self.assertAlmostEqual(expected_scores[item_id], score, places=4)
            self.assertEqual(expected_aaa, aaa)
            self.assertEqual(expected_bbb, bbb)
            self.assertEqual(expected_ccc, ccc)
            self.assertEqual(expected_text, text)

    def test_sqlite_knn_basic_query_eq_operator(self) -> None:
        self.progress("creating dataset, metadata CSV, SQLite metadata table, and vlite virtual table")
        con, _ = self.prepare_case()
        try:
            query_value = 7.4
            k = 5
            self.progress("running basic WHERE query = ? KNN query through SQLite")
            actual = run_basic_knn_query(con, query_vector(query_value), k)
            expected = expected_knn_rows(query_value, k)
            self.assert_knn_rows_equal(actual, expected)
        finally:
            con.close()

    def test_sqlite_knn_basic_query_match_operator(self) -> None:
        self.progress("creating fresh dataset and metadata for MATCH-based SQLite query")
        con, _ = self.prepare_case()
        try:
            query_value = 12.35
            k = 4
            self.progress("running WHERE match_expr MATCH ? KNN query through SQLite")
            actual = run_match_knn_query(con, query_vector(query_value), k)
            expected = expected_knn_rows(query_value, k)
            self.assert_knn_rows_equal(actual, expected)
        finally:
            con.close()

    def test_sqlite_join_knn_results_with_metadata(self) -> None:
        self.progress("creating dataset and importing generated metadata CSV into SQLite")
        con, _ = self.prepare_case()
        try:
            query_value = 7.4
            k = 4
            self.progress("joining nearest neighbors with metadata rows on id")
            actual = run_join_query(con, query_vector(query_value), k)
            expected_ids = [item_id for item_id, _ in expected_knn_rows(query_value, k)]
            self.assert_join_rows_match_ids(actual, expected_ids, query_value)
        finally:
            con.close()

    def test_sqlite_join_with_metadata_filters_after_knn(self) -> None:
        self.progress("creating dataset and SQLite metadata state for post-filter join query")
        con, _ = self.prepare_case()
        try:
            query_value = 7.4
            k = 6
            self.progress("running join query that filters metadata after the KNN step")
            actual = run_post_filtered_join_query(con, query_vector(query_value), k, aaa=1)
            baseline_ids = [item_id for item_id, _ in expected_knn_rows(query_value, k)]
            expected_ids = [item_id for item_id in baseline_ids if item_id % 2 == 1]
            self.assert_join_rows_match_ids(actual, expected_ids, query_value)
        finally:
            con.close()

    def test_sqlite_pushdown_allowed_ids_from_metadata_subquery(self) -> None:
        self.progress("creating dataset and SQLite metadata state for allowed_ids pushdown query")
        con, _ = self.prepare_case()
        try:
            query_value = 7.4
            k = 6
            self.progress("running KNN query with allowed_ids produced from metadata WHERE aaa = 1")
            actual = run_pushdown_query(con, query_vector(query_value), k, "aaa = 1")
            allowed_ids = {item_id for item_id in range(START_ID, START_ID + COUNT) if item_id % 2 == 1}
            expected = expected_knn_rows(query_value, k, allowed_ids=allowed_ids)
            self.assert_knn_rows_equal(actual, expected)
        finally:
            con.close()

    def test_sqlite_pushdown_changes_neighbor_set_vs_unfiltered_knn(self) -> None:
        self.progress("creating dataset and SQLite metadata state for pushdown-vs-postfilter comparison")
        con, _ = self.prepare_case()
        try:
            query_value = 7.4
            k = 6
            self.progress("comparing metadata post-filtering with metadata pushdown into allowed_ids")
            post_filtered = run_post_filtered_join_query(con, query_vector(query_value), k, aaa=1)
            pushed_down = run_pushdown_query(con, query_vector(query_value), k, "aaa = 1")

            post_filtered_ids = [row[0] for row in post_filtered]
            pushed_down_ids = [row[0] for row in pushed_down]

            self.assertEqual([7, 9, 5], post_filtered_ids)
            self.assertEqual([7, 9, 5, 11, 3, 13], pushed_down_ids)
            self.assertNotEqual(post_filtered_ids, pushed_down_ids)
        finally:
            con.close()

    def test_sqlite_pushdown_with_different_metadata_predicates(self) -> None:
        self.progress("creating dataset and SQLite metadata state for multiple pushdown predicates")
        con, _ = self.prepare_case()
        try:
            query_value = 12.35
            k = 4
            cases = [
                (
                    "bbb_in_1_3",
                    "bbb IN (1, 3)",
                    {item_id for item_id in range(START_ID, START_ID + COUNT) if item_id % 5 in (1, 3)},
                ),
                (
                    "ccc_between_2_6",
                    "ccc BETWEEN 2 AND 6",
                    {item_id for item_id in range(START_ID, START_ID + COUNT) if 2 <= (item_id % 10) <= 6},
                ),
            ]
            for label, predicate_sql, allowed_ids in cases:
                with self.subTest(predicate=label):
                    actual = run_pushdown_query(con, query_vector(query_value), k, predicate_sql)
                    expected = expected_knn_rows(query_value, k, allowed_ids=allowed_ids)
                    self.assert_knn_rows_equal(actual, expected)
        finally:
            con.close()

    def test_sqlite_pushdown_and_join_return_metadata_columns(self) -> None:
        self.progress("creating dataset and SQLite metadata state for combined pushdown-plus-join query")
        con, _ = self.prepare_case()
        try:
            query_value = 12.35
            k = 4
            predicate_sql = "bbb IN (1, 3)"
            self.progress("running pushdown KNN query and joining surviving neighbors with metadata")
            actual = run_pushdown_join_query(con, query_vector(query_value), k, predicate_sql)
            allowed_ids = {item_id for item_id in range(START_ID, START_ID + COUNT) if item_id % 5 in (1, 3)}
            expected_ids = [item_id for item_id, _ in expected_knn_rows(query_value, k, allowed_ids=allowed_ids)]
            self.assert_join_rows_match_ids(actual, expected_ids, query_value)
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
