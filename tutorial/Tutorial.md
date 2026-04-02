# Sketch2 Tutorials

This guide walks through the two starter scripts:
- `tutorial_00.py`: prepares a runnable Sketch2 environment on your filesystem.
- `tutorial_01.py`: exercises the basic open/create/close/drop workflow against that environment.

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
