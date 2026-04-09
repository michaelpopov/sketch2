#!/usr/bin/env python3
"""Tutorial showing how to run KNN search on a Sketch2 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from sketch2_utils import get_db_path, get_lib_paths, load_sketch2_types


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KNN search on a Sketch2 dataset via the Python wrapper.")
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


def print_knn_ids(title: str, query: str, k: int, ids: list[int]) -> None:
    print("")
    print(title)
    print(f"  query: [{query}]")
    print(f"  k    : {k}")
    if not ids:
        print("  (no rows)")
        return
    for rank, item_id in enumerate(ids, start=1):
        print(f"  #{rank:02d} id={int(item_id):>3}")


def run_knn_queries(sketch2) -> None:
    queries = [
        #  Quert vector         K (count)          Expected Ids
        ("1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10", 3, [20, 30, 10]),
        ("8.50, 8.50, 8.50, 8.50, 8.50, 8.50, 8.50, 8.50", 2, [40, 30]),
        ("0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00", 1, [10]),
    ]

    for query, count, expected in queries:
        ids = sketch2.knn(query, count)
        print_knn_ids("KNN rows:", query, count, ids)
        if ids != expected:
            raise RuntimeError(
                f"KNN mismatch for query '{query}': expected {expected}, got {ids}"
            )

    print("")
    print("Completed validating KNN search results")


def main() -> None:
    lib_dir, lib_path = get_lib_paths()
    Sketch2, Sketch2Error = load_sketch2_types(lib_dir)

    args = parse_args()
    db_path = get_db_path()
    dataset_name = args.dataset

    try:
        with Sketch2(db_path, lib_path=lib_path) as sketch2:
            create_dataset(sketch2, db_path, dataset_name)

            sketch2.open(dataset_name)
            print(f"Opened dataset '{dataset_name}' successfully")

            insert_test_data(sketch2)
            run_knn_queries(sketch2)

            sketch2.close()
            print(f"Closed dataset '{dataset_name}' successfully")

            sketch2.drop(dataset_name)
            print(f"Dropped dataset '{dataset_name}' successfully")

    except Sketch2Error as exc:
        raise SystemExit(exc) from exc


if __name__ == "__main__":
    main()
