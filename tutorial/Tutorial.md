# Sketch2 Tutorials

This guide walks through the starter scripts:
- `tutorial_00.py`: prepares a runnable Sketch2 environment on your filesystem.
- `tutorial_01.py`: exercises the basic open/create/close/drop workflow against that environment.
- `tutorial_02.py`: demonstrates staged writes, reads, updates, and delta merging.
- `tutorial_03.py`: shows how dataset files evolve as data is loaded, deleted, updated, and merged.

## 0. Environment Prep (`tutorial_00.py`)

Run:
`python3 tutorial_00.py --root /tmp/sketch2`

What it does:
- Creates the directory layout that tutorial scripts expect.
- Copies the shared library and Python wrapper helpers into place.
- Writes a minimal `config.ini`.
- Prints the `export` commands for `SKETCH2_LIB` and `SKETCH2_CONFIG`.

Resulting layout (root name is configurable via `--root`):
```
sketch2/
  db/
    config.ini
  lib/
    libsketch2.so
    sketch2_wrapper.py
    sketch2_utils.py
```

Notes:
- You can keep multiple configs (for writer vs reader processes) by changing the config filename.
- The `lib` directory must contain both the shared library and the Python helper modules.

## 1. Basic Dataset Flow (`tutorial_01.py`)

Prereqs:
- Run `tutorial_00.py` (or provide an equivalent layout).
- Export the environment variables printed by `tutorial_00.py`:
  - `SKETCH2_LIB` points to the `lib` directory with `libsketch2.so` and helpers.
  - `SKETCH2_CONFIG` points to the desired `config.ini`.

Run:
`python3 tutorial_01.py demods`

What it demonstrates:
- Initialize a `Sketch2` handle.
- Try to open a dataset.
- If it is missing, create it.
- Close the dataset.
- Drop the dataset (cleanup for the next tutorial).

Dataset artifacts created for dataset name `demods`:
```
sketch2/
  db/
    config.ini
    demods/
      demods.ini
      part_00/    # data files live here
```

Only one `part_00` directory is created for simplicity, but multiple parts can be spread across disks to improve I/O parallelism in real deployments.

## 2. Basic Writes, Reads and Updates (`tutorial_02.py`)

Prereqs:
- Run `tutorial_00.py` to prepare the environment.
- Make sure `SKETCH2_LIB` and `SKETCH2_CONFIG` are exported.
- Start from a clean environment or let the script create and later drop the dataset.

Run:
`python3 tutorial_02.py demods`

What it demonstrates:
- Create a dataset with `f32` vectors of dimension 8.
- Open the dataset and start a staged write session.
- Write 100 vectors with ids `0..99` and values `0.00`, `0.10`, `0.20`, and so on.
- Finalize the staged write so the temporary input file is stored into persisted dataset files.
- Read all written vectors back with `get(id)` and validate them against control strings.
- Start another staged write session to:
  - add a new vector with id `111`
  - update existing vector `22`
  - delete existing vector `33`
- Finalize the update session and verify that reads now reflect both `.data` and `.delta` contents.
- Call `merge_delta()` to merge the delta file back into the main data file.
- Re-run the validation after merge, then close and drop the dataset.

Write-path notes:
- `start_writing()` creates a temporary `<dataset_name>.input` file.
- `write_vector()` and `write_deleted()` append staged changes to that file.
- Before `complete_writing()`, staged changes are not visible to readers.
- After `complete_writing()`, the temporary file is removed and the changes become visible.

Persisted files after the first write:
```
sketch2/
  db/
    config.ini
    demods/
      demods.ini
      demods.lock
      part_00/
        demods.input --- temporary file used for accumulating writes before storing them
                         in .data or .delta file
        0.data
```

Persisted files after the update step:
```
sketch2/
  db/
    config.ini
    demods/
      demods.ini
      demods.lock
      part_00/
        0.data
        0.delta    --- .delta file accumulates updates to prevent write amplification related
                       to changes in main .data file. When .delta file grows to certain percentage
                       of .data file, it is merged into .data file and a new .delta file is started.
```

After `merge_delta()`:
- `0.delta` is folded back into `0.data`.
- Reads are served from the merged data file.

## 3. File Evolution and Multi-Part Layout (`tutorial_03.py`)

Prereqs:
- Run `tutorial_00.py` to prepare the environment.
- Make sure `SKETCH2_LIB` and `SKETCH2_CONFIG` are exported.
- Start from a clean environment or let the script create and later drop the dataset.

Run:
`python3 tutorial_03.py demods`

What it demonstrates:
- Create a dataset with two storage directories: `part_00` and `part_01`.
- Generate test input into a local temporary file and bulk-load it into the dataset.
- Print dataset stats after each major step to observe how `.data` and `.delta` files change.
- Delete a regular subset of vectors through staged writes and confirm one deleted vector is no longer readable.
- Update another regular subset of vectors and confirm one updated vector returns the expected value.
- Generate and load another batch of vectors with ids beyond the original range.
- Merge delte files and inspect the final file layout after pending deltas are folded into data files.

Workflow notes:
- The script uses `generate_test_data()` to build a deterministic input file.
- Initial bulk loads use `load_file()` to store generated vectors into dataset files.
- Deletes and updates are performed through staged write sessions.
- Because the dataset spans two `part_*` directories, file creation and growth can be observed across multiple storage locations.

Expected observations:
- After the first load, several `.data` files appear across `part_00` and `part_01`.
- After the delete step, matching `.delta` files appear while the original `.data` files remain unchanged.
- After the update step, those `.delta` files continue to accumulate both deletions and replacements.
- After loading more vectors, new files appear for new id ranges and one delta file is merged automatically because it become large enough.
- After force merge operation, remaining `.delta` files are folded into `.data` files.
