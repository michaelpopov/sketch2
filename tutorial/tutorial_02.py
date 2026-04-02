#!/usr/bin/env python3
"""Minimal tutorial showing how to implement basic writing and reading from a Sketch2 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from sys import exit

from sketch2_utils import get_db_path, get_lib_paths, load_sketch2_types

# Loaded once in main(); declared global for helper functions.
Sketch2 = None
Sketch2Error = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open a Sketch2 dataset via the Python wrapper.")
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
        dir = dataset_dir / f"part_{index:02d}"
        dirs.append(dir)

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

# Example of writing to the dataset.
def write_dataset(sketch2, vectors_count) -> None:
    vectors_count = 100

    # Before writing actual data, function start_writing() must be called.
    # It prepares a local file <dataset_name>.input and accumulates all 
    # the following data writes before storing them into the data files.
    sketch2.start_writing()

    # After calling start_writing() the script can actually start writing
    # data into the currently open dataset.
    for idx in range(vectors_count):
        value = idx / 10.0
        vector = ", ".join([f"{value:.2f}"] * 8)
        sketch2.write_vector(idx, vector)

    # Before complete_writing() is called the new data is NOT VISIBLE to
    # the Sketch2 readers because all vectors are accumulated in the input
    # file and stored in the data files.
    sketch2.complete_writing()
    print(f"Generated {vectors_count} vectors")

    # After complete_writing() is done all the data from the temporary input
    # file is stored in the data (or delta) file and the temporary file is
    # removed. From this point the data becomes visible for Sketch2 readers.
    # At this point there is part_00/0.data file that contains all the data.
    # Sketch2 readers can access this file and get all the data.

def read_dataset(sketch2, vectors_count) -> None:
    # This loop retrieves each vector that we just written in write_dataset()
    # function and compares it to the corresponding control string.
    # NOTE: retrieving a string with vector is intended only for test scenarios!
    # In regular production workflows there is no point of getting these strings.
    for idx in range(vectors_count):
        value = idx / 10.0
        expected = f"[ {', '.join([f'{value:.2f}'] * 8)} ]"
        actual = sketch2.get(idx)
        if actual != expected:
            raise RuntimeError(f"Vector mismatch for id={idx}: expected '{expected}', got '{actual}'")
    print(f"Validated {vectors_count} vectors")


def update_dataset(sketch2) -> None:
    # Calling start_writing() puts a dataset into "writing mode". If a dataset
    # is already in "writing mode" calling start_writing() will fail.
    # Dataset can exit from "writing mode" either by calling complete_writing()
    # or calling abort_writing().
    sketch2.start_writing()

    # Adding one more vector with id 111
    id = 111
    value = 111.11
    vector = ", ".join([f"{value:.2f}"] * 8)
    sketch2.write_vector(id, vector)
    print(f"Added vector with id {id}")

    # Updating existing vector with id 22
    id = 22
    value = 222.22
    vector = ", ".join([f"{value:.2f}"] * 8)
    sketch2.write_vector(id, vector)
    print(f"Updated vector with id {id}")

    # Deleting existing vector with id 33
    id = 33
    sketch2.write_deleted(33)
    print(f"Deleted vector with id {id}")

    # All the updates are accumulated in the input file and not visible to
    # Sketch2 readers at this stage.

    # Calling complete_writing() forces dataset to store data in data/delta files.
    sketch2.complete_writing()
    print("Completed updating the dataset")

    # After complete_writing() is executed there is a new file part_00/0.delta
    # that contains updates. Sketch2 does not update 0.data file at this stage
    # to avoid too much of write amplification.

def read_updated_dataset(sketch2) -> None:
    # Sketch2 reader opens both 0.data and 0.delta files and establishes in-memory
    # index of modified and deleted values. Then it can read correct values stored
    # in both of these files

    id = 111
    value = 111.11
    expected = f"[ {', '.join([f'{value:.2f}'] * 8)} ]"
    actual = sketch2.get(id)
    if actual != expected:
        raise RuntimeError(f"Added vector mismatch for id={id}: expected '{expected}', got '{actual}'")

    id = 22
    value = 222.22
    expected = f"[ {', '.join([f'{value:.2f}'] * 8)} ]"
    actual = sketch2.get(id)
    if actual != expected:
        raise RuntimeError(f"Updated vector mismatch for id={id}: expected '{expected}', got '{actual}'")

    id = 33
    found = True
    try:
        actual = sketch2.get(id)
    except Sketch2Error as exc:
        found = False
    if found:
        raise RuntimeError(f"Deleted vector found for id={id}")
    
    print("Completed validating updates in the dataset")

def main() -> None:
    lib_dir, lib_path = get_lib_paths()
    global Sketch2, Sketch2Error
    Sketch2, Sketch2Error = load_sketch2_types(lib_dir)

    args = parse_args()
    db_path = get_db_path()
    dataset_name = args.dataset

    try:
        with Sketch2(db_path, lib_path=lib_path) as sketch2:
            create_dataset(sketch2, db_path, dataset_name)

            # Object sketch2 can have only one dataset open at any moment of time.
            # Attempt to call open() again when a dataset is already opened will fail.
            # The dataset must be closed using close() call before opening another one.
            sketch2.open(dataset_name)
            print(f"Opened dataset '{dataset_name}' successfully")

            write_dataset(sketch2, vectors_count = 100)
            read_dataset(sketch2, vectors_count = 100)

            update_dataset(sketch2)
            read_updated_dataset(sketch2)

            # After dataset was updated there are .data and .delta files.
            # It is possible to force Sketch2 to merge these files into a single
            # .data file.
            sketch2.merge_delta()
            print("Merged delta file into data file in the dataset")

            # After calling merge_data(), there is a single file part_00/0.data
            # All the reads will be done from this file now.
            read_updated_dataset(sketch2)

            sketch2.close()
            print(f"Closed dataset '{dataset_name}' successfully")

            sketch2.drop(dataset_name)
            print(f"Dropped dataset '{dataset_name}' successfully")

    except Sketch2Error as exc:
        raise SystemExit(exc) from exc


if __name__ == "__main__":
    main()
