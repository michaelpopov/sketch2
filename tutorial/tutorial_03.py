#!/usr/bin/env python3
"""Tutorial showing evolution of storage files in Sketch2 dataset as data added and modified."""

from __future__ import annotations

import argparse
from pathlib import Path
from sys import exit

from sketch2_utils import get_db_path, get_lib_paths, load_sketch2_types

# Loaded once in main(); declared global for helper functions.
Sketch2 = None
Sketch2Error = None

range_size = 10000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open a Sketch2 dataset via the Python wrapper.")
    parser.add_argument(
        "dataset",
        help="Name of the dataset to open inside the database root",
    )
    return parser.parse_args()

# This create_dataset creates a dataset with 2 data directories. All data and delta files
# storing data for this dataset will be distributed in these two directories.
def create_dataset(sketch2, db_path: Path, dataset_name: str) -> None:
    print(f"Opened Sketch2 database in {db_path}")
    dataset_dir = db_path / dataset_name

    dirs_count = 2
    dirs: list[Path] = []
    for index in range(dirs_count):
        dir_path = dataset_dir / f"part_{index:02d}"
        dirs.append(dir_path)

    sketch2.create(
        dataset_name,
        dirs=dirs,
        type_name="f32",
        dim=8,
        range_size=range_size,
        dist_func="l2",
    )

    print(f"Created dataset '{dataset_name}'")
    sketch2.close()

# Development of Sketch2 requires loading sets of vector embeddings. Sketch2 includes
# functionality for generating test data that can be used for loading and testing it.
# The test data is written into a local file that can be loaded into Sketch2 dataset
# after that. There are two modes for input data: text and binary. Text mode is convinient
# because it allows examining what is actually loaded into dataset. Binary mode allows
# generating large sets of vectors in much shorter time than it can be done with text
# mode.
# This function creates a path to a local file for storing generated test data.
def make_test_file_path(db_path) -> Path:
    # db_path is expected to be a config file; if a directory is passed, use it directly.
    resolved = Path(db_path).resolve()
    base_dir = resolved.parent if resolved.is_file() else resolved
    test_dir = base_dir / "temp"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir

# Delete a subset of vectors in a dataset.
def delete_vectors(sketch2, vectors_count, from_index, step) -> None:
    sketch2.start_writing()

    for idx in range(from_index, vectors_count, step):
        sketch2.write_deleted(idx)

    sketch2.complete_writing()

# Sanity check to confirm that a vector was deleted.
def check_deleted_vector(sketch2, id) -> None:
    found = True
    try:
        sketch2.get(id)
    except Sketch2Error as exc:
        found = False
    if found:
        raise RuntimeError(f"Deleted vector found for id={id}")

# Update a subset of vectors in a dataset.
def update_vectors(sketch2, vectors_count, from_index, step) -> None:
    sketch2.start_writing()

    for idx in range(from_index, vectors_count, step):
        value = idx / 10.0
        vector = ", ".join([f"{value:.2f}"] * 8)
        sketch2.write_vector(idx, vector)

    sketch2.complete_writing()

# Sanity check to confirm that a vector was updated.
def check_updated_vector(sketch2, id) -> None:
    value = id / 10.0
    expected = f"[ {', '.join([f'{value:.2f}'] * 8)} ]"
    actual = sketch2.get(id)
    if actual != expected:
        raise RuntimeError(f"Updated vector mismatch for id={id}: expected '{expected}', got '{actual}'")

# Operations in this function create and modify data and delta files of the Sketch2 dataset.
# They produce deterministic predictable results that can be observed and controlled.
def main() -> None:
    lib_dir, lib_path = get_lib_paths()
    global Sketch2, Sketch2Error
    Sketch2, Sketch2Error = load_sketch2_types(lib_dir)

    args = parse_args()
    db_path = get_db_path()
    dataset_name = args.dataset
    test_data_path = make_test_file_path(db_path)
    test_data_file = test_data_path / "test.input"
    vectors_count = int(range_size * 2.5)
    modify_step = int(vectors_count / 1000)

    try:
        with Sketch2(db_path, lib_path=lib_path) as sketch2:
            create_dataset(sketch2, db_path, dataset_name)

            sketch2.open(dataset_name)
            print(f"Opened dataset '{dataset_name}' successfully")

            print("---------- Dataset info after initial input ----------------------\n")
            sketch2.generate_test_data(test_data_file, vectors_count, from_index = 0, binary=True)
            sketch2.load_file(test_data_file)
            sketch2.stats()
            # At this point there are 3 data files: two data files in directory part_00 and one
            # data file in directory part_01.

            print("\n\n---------- Dataset info after deleting some vectors ----------------------\n")
            delete_vectors(sketch2, vectors_count, 0, modify_step)
            check_deleted_vector(sketch2, id = modify_step)
            sketch2.stats()
            # At this point there are 3 additional delta files that contain information about deleted
            # vectors. Data files were not touched by these updates.

            print("\n\n---------- Dataset info after updating some vectors ----------------------\n")
            update_vectors(sketch2, vectors_count, 1, modify_step)
            check_updated_vector(sketch2, id = modify_step + 1)
            sketch2.stats()
            # At this point there are 3 same delta files that contain information about deleted
            # and updated vectors. Data files were not touched by these updates.

            print("\n\n---------- Dataset info after adding more vectors ----------------------\n")
            sketch2.generate_test_data(test_data_file, vectors_count, from_index = vectors_count, binary=True)
            sketch2.load_file(test_data_file)
            sketch2.stats()
            # At this point there are 2 delta files that were not touched by this operation because they cover
            # ranges not overlapping with ids of newly generated test vectors.
            # One of the delta files that covers the range of ids included in a set of newly generated vectors
            # was modified to absorb part of new vectors. It grew in size over threshold that requires merging
            # it with corresponding data file. This delta file was merged into the data file and deleted.
            # The other two original data files were not changed.
            # There are two new data files in id ranges that were not covered by the original data files.
            # They contain part of newly added vectors.

            print("\n\n---------- Dataset info after merging delta files ----------------------\n")
            sketch2.merge_delta()
            sketch2.stats()
            # At this point there are no more delta files. They were merged into data files.

            sketch2.close()
            print(f"Closed dataset '{dataset_name}' successfully")

            sketch2.drop(dataset_name)
            print(f"Dropped dataset '{dataset_name}' successfully")

    except Sketch2Error as exc:
        raise SystemExit(exc) from exc


if __name__ == "__main__":
    main()
