#!/usr/bin/env python3
"""Minimal tutorial showing how to open, create and close a Sketch2 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from sketch2_utils import get_db_path, get_lib_paths, load_sketch2_types


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open a Sketch2 dataset via the Python wrapper.")
    parser.add_argument(
        "dataset",
        help="Name of the dataset to open inside the database root",
    )
    return parser.parse_args()


def create_dataset(sketch2, db_path: Path, dataset_name: str) -> None:
    dataset_dir = db_path / dataset_name

    dirs_count = 1
    dirs: list[Path] = []
    for index in range(dirs_count):
        dir_path = dataset_dir / f"part_{index:02d}"
        dirs.append(dir_path)

    sketch2.create(
        dataset_name,
        dirs=dirs,
        type_name="f16",
        dim=8,
        range_size=10000,
        dist_func="l1",
    )

def main() -> None:
    lib_dir, lib_path = get_lib_paths()
    Sketch2, Sketch2Error = load_sketch2_types(lib_dir)

    args = parse_args()
    db_path = get_db_path()
    dataset_name = args.dataset

    try:
        with Sketch2(db_path, lib_path=lib_path) as sketch2:
            print(f"Opened Sketch2 database in {db_path}")

            try:
                sketch2.open(dataset_name)
                print(f"Opened dataset '{dataset_name}' successfully")
            except Sketch2Error:
                create_dataset(sketch2, db_path, dataset_name)
                print(f"Created dataset '{dataset_name}'")

            sketch2.close()
            print(f"Closed dataset '{dataset_name}' successfully")

            sketch2.drop(dataset_name)
            print(f"Dropped dataset '{dataset_name}' successfully")

    except Sketch2Error as exc:
        raise SystemExit(exc) from exc


if __name__ == "__main__":
    main()
