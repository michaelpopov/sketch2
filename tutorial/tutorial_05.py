#!/usr/bin/env python3
"""Tutorial showing how to query Sketch2 KNN results through SQLite."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from sketch2_utils import get_db_path, get_lib_paths, load_sketch2_types


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Sketch2 KNN queries through SQLite.")
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
        20: 1.00,
        30: 2.00,
        40: 9.00,
    }

    for item_id, value in test_vectors.items():
        vector = ", ".join([f"{value:.2f}"] * 8)
        sketch2.write_vector(item_id, vector)
        print(f"Inserted vector id={item_id}: {vector}")

    sketch2.complete_writing()
    print("Completed writing test vectors")


def dataset_ini_path(db_path: Path, dataset_name: str) -> Path:
    return db_path / dataset_name / f"{dataset_name}.ini"


def print_knn_rows(title: str, query: str, k: int, rows: list[tuple]) -> None:
    print("")
    print(title)
    print(f"  query: [{query}]")
    print(f"  k    : {k}")
    if not rows:
        print("  (no rows)")
        return
    for rank, row in enumerate(rows, start=1):
        item_id, score = row
        print(f"  #{rank:02d} id={int(item_id):>3} score={float(score):.6f}")


def run_sql_queries(dataset_ini: Path, extension_lib: Path) -> None:
    queries = [
        ("1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10", 3, [20, 30, 10]),
        ("8.50, 8.50, 8.50, 8.50, 8.50, 8.50, 8.50, 8.50", 2, [40, 30]),
        ("0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00", 1, [10]),
    ]

    print(f"Opening SQLite in-memory database and loading extension {extension_lib}")
    con = sqlite3.connect(":memory:")
    try:
        con.enable_load_extension(True)
        con.load_extension(str(extension_lib))

        dataset_name = dataset_ini.stem
        db_path = dataset_ini.parent.parent
        db_path_sql = str(db_path).replace("'", "''")
        dataset_name_sql = dataset_name.replace("'", "''")
        create_sql = f"CREATE VIRTUAL TABLE nn USING vlite('{db_path_sql}', '{dataset_name_sql}')"
        print("Executing SQL:")
        print(create_sql)
        con.execute(create_sql)

        query_sql = "SELECT id, score FROM nn WHERE query = ? AND k = ? ORDER BY score"
        print("Executing SQL:")
        print(query_sql)

        for query, count, expected in queries:
            rows = con.execute(query_sql, (query, count)).fetchall()
            ids = [int(row[0]) for row in rows]
            print_knn_rows("KNN rows:", query, count, rows)
            if ids != expected:
                raise RuntimeError(
                    f"SQL KNN mismatch for query '{query}': expected {expected}, got {ids}"
                )

        print("")
        print("Completed validating SQL KNN results")
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

            run_sql_queries(dataset_ini, lib_path)

            sketch2.drop(dataset_name)
            print(f"Dropped dataset '{dataset_name}' successfully")

    except Sketch2Error as exc:
        raise SystemExit(exc) from exc


if __name__ == "__main__":
    main()
