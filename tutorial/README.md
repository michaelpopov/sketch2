# Sketch2 Tutorials

This guide walks through the starter scripts:
- `tutorial_00.py`: prepares a runnable Sketch2 environment on your filesystem.
- `tutorial_01.py`: exercises the basic open/create/close/drop workflow against that environment.
- `tutorial_02.py`: demonstrates staged writes, reads, updates, and delta merging.
- `tutorial_03.py`: shows how dataset files evolve as data is loaded, deleted, updated, and merged.
- `tutorial_04.py`: demonstrates KNN search on a dataset using the Python API.
- `tutorial_05.py`: demonstrates querying SQLite with SQL to retrieve KNN nearest-neighbor ids.
- `tutorial_06.py`: demonstrates SQL joins between KNN results and metadata conditions.
- `tutorial_07.py`: demonstrates pushing metadata filters into KNN search with SQL.

File `output.txt` contains output of the session running all the tutorials in the terminal.

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

## 3. Storage Layout (`tutorial_03.py`)

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
- Merge delta files and inspect the final file layout after pending deltas are folded into data files.

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

## 4. KNN Search Using Python API (`tutorial_04.py`)

Run:
`python3 tutorial_04.py demods`

This tutorial demonstrates how to run K Nearest Neighbors search on a dataset with vector embeddings.
The previous tutorials showed how to insert data into Sketch2 storage. They used get() function
to retrieve vectors to validate write operation. Retrieving vectors with get() function is intended
only for testing purposes. The real usage of the vector storage engine consists in finding "nearest"
vectors to the query vector. Sketch2 provides function knn() for this purpose.

The tutorial script demonstrates how to:
- Create a new dataset.
- Inserts test data into the dataset.
- Run KNN search on the dataset.

## 5. Query database with SQL to get KNN (`tutorial_05.py`)

Run:
`python3 tutorial_05.py demods`

The key feature of Sketch2 storage engine is integration into existing databases that provides
users a convenient way to query data using regular SQL statements. This tutorial demonstrates
how to run find nearest neighbors of a vector by querying SQLite database.

The tutorial script demonstrates how to:
- Create a new dataset.
- Insert test data into the dataset.
- Run SQL statement on SQLite database to retrieve ids of nearest neighbors of a query vector.

```
CREATE VIRTUAL TABLE nn USING vlite('/mnt/nvme/sketch2/db/demods/demods.ini');
SELECT id, score FROM nn WHERE query = ? AND k = ? ORDER BY score;
```

## 6. Database query that joins KNN search and metadata conditions  (`tutorial_06.py`)

Run:
`python3 tutorial_06.py demods`

Integrating Sketch2 into database allows joining results of KNN search and other data
in the database.

The tutorial script demonstrates how to:
- Create a new dataset.
- Insert test data into the dataset.
- Insert metadata into SQLite table.
- Run SQL statement on SQLite database that joins results of KNN search and
  data from a "metadata" table.

```
CREATE VIRTUAL TABLE nn USING vlite('/mnt/nvme/sketch2/db/demods/demods.ini');

CREATE TABLE metadata (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            author TEXT NOT NULL
        );

INSERT INTO metadata(id, title, category, author) VALUES (?, ?, ?, ?);

SELECT m.id, m.title, m.category, m.author, n.score
            FROM nn AS n
            JOIN metadata AS m ON m.id = n.id
            WHERE n.query = ? AND n.k = ?
            ORDER BY n.score;
```


## 7. Pushing filter into KNN search  (`tutorial_07.py`)

Run:
`python3 tutorial_07.py demods`

In some cases it is more efficient to limit the subset of vectors that is used during
the search of Nearest Neighbors. Integration with databases allows pushing this filter
into Sketch2. This tutorial demonstrates how to generate a list of "valid ids" using
SQL query, pass it into Sketch2 and get vectors that match the condition.

The tutorial script demonstrates how to:
- Create a new dataset.
- Insert test data into the dataset.
- Insert metadata into SQLite table.
- Run SQL statement on SQLite database that generates a list of valid ids and
  passes it to Sketch2 KNN search so only vectors with these ids are checked.


```
CREATE VIRTUAL TABLE nn USING vlite('/mnt/nvme/sketch2/db/demods/demods.ini');

CREATE TABLE metadata (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            author TEXT NOT NULL
        );

INSERT INTO metadata(id, title, category, author) VALUES (?, ?, ?, ?);

SELECT m.id, m.title, m.category, m.author, n.score
            FROM nn AS n
            JOIN metadata AS m ON m.id = n.id
            WHERE n.query = ?
              AND n.k = ?
              AND n.allowed_ids = (
                    SELECT bitset_agg(id)
                    FROM (
                        SELECT id
                        FROM metadata
                        WHERE category = ?
                        ORDER BY id
                    )
                  )
            ORDER BY n.score;
```
