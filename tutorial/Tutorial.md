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
