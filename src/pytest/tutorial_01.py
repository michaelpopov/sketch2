#!/usr/bin/env python3
"""Minimal tutorial showing how to open, create and close a Sketch2 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from sketch2_utils import get_connect_path, get_lib_paths, load_sketch2_types


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open a Sketch2 dataset via the Python wrapper.")
    parser.add_argument(
        "dataset",
        help="Name of the dataset to open inside the database root",
    )
    return parser.parse_args()


def create_dataset(sketch2, connect_path: Path, dataset_name: str) -> None:
    dataset_dir = connect_path / dataset_name

    dirs_count = 4
    dirs: list[Path] = []
    for index in range(dirs_count):
        dir = dataset_dir / f"part_{index:02d}"
        dirs.append(dir)

    sketch2.create(
        dataset_name,
        dirs=dirs,
        type_name="f16",
        dim=1536,
        range_size=10000,
        dist_func="l1",
    )

def main() -> None:
    args = parse_args()
    lib_dir, lib_path = get_lib_paths()
    connect_path = get_connect_path()

    dataset_name = args.dataset

    Sketch2, Sketch2Error = load_sketch2_types(lib_dir)

    try:
        with Sketch2(lib_path=lib_path) as sketch2:
            sketch2.connect(connect_path)
            print(f"Connected Sketch2 in {connect_path}")

            try:
                sketch2.open(dataset_name)
                print(f"Opened dataset '{dataset_name}' successfully")
            except Sketch2Error:
                create_dataset(sketch2, connect_path, dataset_name)
                print(f"Created dataset '{dataset_name}'")

            sketch2.close(dataset_name)
            print(f"Closed dataset '{dataset_name}' successfully")

            sketch2.drop(dataset_name)
            print(f"Dropped dataset '{dataset_name}' successfully")

            sketch2.disconnect()
            print(f"Disconnected Sketch2 in {connect_path}")
    except Sketch2Error as exc:
        raise SystemExit(exc) from exc


if __name__ == "__main__":
    main()
