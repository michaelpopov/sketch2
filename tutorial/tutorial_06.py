#!/usr/bin/env python3
"""Tutorial showing SQL joins between Sketch2 KNN results and metadata."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from sketch2_utils import get_db_path, get_lib_paths, load_sketch2_types


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SQLite queries that join Sketch2 KNN results with metadata."
    )
    parser.add_argument(
        "dataset",
        help="Name of the dataset to open inside the database root",
    )
    return parser.parse_args()


def create_dataset(sketch2, db_path: Path, dataset_name: str) -> None:
    print(f"Opened Sketch2 database in {db_path}")
    dataset_dir = db_path / dataset_name

    dirs_count = 1
    dirs: list[Path] = []
    for index in range(dirs_count):
        dir_path = dataset_dir / f"part_{index:02d}"
        dirs.append(dir_path)

    sketch2.create(
        dataset_name,
        dirs=dirs,
        type_name="f32",
        dim=8,
        range_size=10000,
        dist_func="l2",
    )

    print(f"Created dataset '{dataset_name}'")
    sketch2.close()


def insert_test_data(sketch2) -> None:
    sketch2.start_writing()

    test_vectors = {
        10: 0.00,
        20: 0.50,
        30: 1.00,
        40: 1.50,
        50: 2.00,
        60: 3.00,
        70: 6.00,
        80: 9.00,
    }

    for item_id, value in test_vectors.items():
        vector = ", ".join([f"{value:.2f}"] * 8)
        sketch2.write_vector(item_id, vector)
        print(f"Inserted vector id={item_id}: {vector}")

    sketch2.complete_writing()
    print("Completed writing test vectors")


def dataset_ini_path(db_path: Path, dataset_name: str) -> Path:
    return db_path / dataset_name / f"{dataset_name}.ini"


def create_metadata_table(con: sqlite3.Connection) -> None:
    create_sql = """
        CREATE TABLE metadata (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            author TEXT NOT NULL
        )
    """
    print("Executing SQL:")
    print(create_sql.strip())
    con.execute(create_sql)

    rows = [
        (10, "Intro to Vectors", "books", "alice"),
        (20, "Nearest Neighbor Notes", "blog", "bob"),
        (30, "Vector Search Handbook", "books", "carol"),
        (40, "Embeddings in Practice", "talks", "dave"),
        (50, "ANN Benchmarks", "books", "erin"),
        (60, "Similarity Metrics", "papers", "frank"),
        (70, "Large-Scale Retrieval", "papers", "grace"),
        (80, "Audio Similarity Primer", "music", "heidi"),
    ]
    insert_sql = "INSERT INTO metadata(id, title, category, author) VALUES (?, ?, ?, ?)"
    print("Executing SQL:")
    print(insert_sql)
    con.executemany(insert_sql, rows)
    print(f"Inserted {len(rows)} metadata rows")


def print_join_rows(title: str, rows: list[tuple]) -> None:
    print("\n")
    print(title)
    if not rows:
        print("  (no rows)")
        return

    for rank, row in enumerate(rows, start=1):
        item_id, item_title, category, author, score = row
        print(f"  #{rank:02d} id={item_id:>3} score={float(score):.6f}")
        print(f"      title   : {item_title}")
        print(f"      category: {category}")
        print(f"      author  : {author}")

    print("\n")


def run_sql_join_queries(dataset_ini: Path, extension_lib: Path) -> None:
    print(f"Opening SQLite in-memory database and loading extension {extension_lib}")
    con = sqlite3.connect(":memory:")
    try:
        con.enable_load_extension(True)
        con.load_extension(str(extension_lib))

        ini_sql = str(dataset_ini).replace("'", "''")
        create_vtab_sql = f"CREATE VIRTUAL TABLE nn USING vlite('{ini_sql}')"
        print("Executing SQL:")
        print(create_vtab_sql)
        con.execute(create_vtab_sql)

        create_metadata_table(con)

        query = "1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10"

        join_sql = """
            SELECT m.id, m.title, m.category, m.author, n.score
            FROM nn AS n
            JOIN metadata AS m ON m.id = n.id
            WHERE n.query = ? AND n.k = ?
            ORDER BY n.score
        """
        print("Executing SQL:")
        print(join_sql.strip())
        rows = con.execute(join_sql, (query, 5)).fetchall()
        ids = [int(row[0]) for row in rows]
        expected = [30, 40, 20, 50, 10]
        print_join_rows("Joined KNN rows (no metadata filter):", rows)
        if ids != expected:
            raise RuntimeError(
                f"Joined SQL KNN mismatch (no filter): expected {expected}, got {ids}"
            )

        filtered_join_sql = """
            SELECT m.id, m.title, m.category, m.author, n.score
            FROM nn AS n
            JOIN metadata AS m ON m.id = n.id
            WHERE n.query = ? AND n.k = ? AND m.category = ?
            ORDER BY n.score
        """
        print("Executing SQL:")
        print(filtered_join_sql.strip())
        rows = con.execute(filtered_join_sql, (query, 8, "books")).fetchall()
        ids = [int(row[0]) for row in rows]
        expected = [30, 50, 10]
        print_join_rows("Joined KNN rows (metadata filter category='books'):", rows)
        if ids != expected:
            raise RuntimeError(
                f"Joined SQL KNN mismatch (category='books'): expected {expected}, got {ids}"
            )

        print("Completed validating SQL joins between KNN results and metadata")
    finally:
        con.close()


def main() -> None:
    lib_dir, lib_path = get_lib_paths()
    Sketch2, Sketch2Error = load_sketch2_types(lib_dir)

    args = parse_args()
    db_path = get_db_path()
    dataset_name = args.dataset
    dataset_ini = dataset_ini_path(db_path, dataset_name)

    try:
        with Sketch2(db_path, lib_path=lib_path) as sketch2:
            create_dataset(sketch2, db_path, dataset_name)

            sketch2.open(dataset_name)
            print(f"Opened dataset '{dataset_name}' successfully")

            insert_test_data(sketch2)

            # SQLite queries persisted dataset files, so close the writer first.
            sketch2.close()
            print(f"Closed dataset '{dataset_name}' successfully")

            run_sql_join_queries(dataset_ini, lib_path)

            sketch2.drop(dataset_name)
            print(f"Dropped dataset '{dataset_name}' successfully")

    except Sketch2Error as exc:
        raise SystemExit(exc) from exc


if __name__ == "__main__":
    main()
