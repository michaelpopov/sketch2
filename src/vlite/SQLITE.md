# Using Sketch2 As A SQLite Virtual Table

Sketch2 provides a SQLite virtual table extension called `vlite`. It lets you run
K-nearest-neighbor (KNN) vector searches using standard SQL. The virtual table is
read-only: you point it at an existing Sketch2 dataset, then query nearest
neighbors with `SELECT` statements.

## Table Of Contents

- [Building The Extension](#building-the-extension)
- [SQLITE_OMIT_LOAD_EXTENSION](#sqlite_omit_load_extension)
- [Loading The Extension](#loading-the-extension)
- [Creating A Virtual Table](#creating-a-virtual-table)
- [Virtual Table Schema](#virtual-table-schema)
- [Dataset INI Parameters](#dataset-ini-parameters)
- [Environment Variables](#environment-variables)
- [Query Vector Format](#query-vector-format)
- [SQL Examples](#sql-examples)
  - [Basic Nearest Neighbor Search](#basic-nearest-neighbor-search)
  - [Controlling The Number Of Results With k](#controlling-the-number-of-results-with-k)
  - [Using MATCH Instead Of Equals](#using-match-instead-of-equals)
  - [LIMIT And OFFSET](#limit-and-offset)
  - [Ordering By Distance](#ordering-by-distance)
  - [Reading Hidden Columns](#reading-hidden-columns)
  - [Querying Multiple Datasets](#querying-multiple-datasets)
- [Joins With Other Tables](#joins-with-other-tables)
- [Distance Functions](#distance-functions)
- [Supported Data Types](#supported-data-types)
- [Error Cases](#error-cases)
- [Limitations](#limitations)

## Building The Extension

The extension is compiled into `libsketch2.so`, which also contains the Parasol C
API. Both APIs share a single process-wide runtime (logger, thread pool,
configuration).

## SQLITE_OMIT_LOAD_EXTENSION

Some SQLite builds are compiled with `SQLITE_OMIT_LOAD_EXTENSION`, which removes
runtime extension loading entirely. When this flag is set, `.load` does not exist
and `sqlite3_load_extension()` always returns an error. The `vlite` extension
cannot be used with such a build regardless of how `libsketch2.so` is configured.

This is common with system-provided SQLite packages on Linux distributions, which
often strip extension loading to reduce attack surface. If you get an error like
`no such module: vlite` or `.load` is not recognized, check whether your SQLite
was built with this flag:

```bash
sqlite3 ':memory:' 'SELECT sqlite_compileoption_used("OMIT_LOAD_EXTENSION")'
```

A result of `1` means extension loading is disabled. Use the SQLite shell from the
Sketch2 build tree (`bin/sqlite3` or `bin-dbg/sqlite3`), which is built without
this flag.

## Loading The Extension

From the bundled SQLite shell:

```sql
.load /absolute/path/to/libsketch2.so
```

If environment-based configuration is needed, set it before launching SQLite:

```bash
export SKETCH2_LOG_LEVEL=DEBUG
export SKETCH2_THREAD_POOL_SIZE=8
export SKETCH2_LOG_FILE=/tmp/sketch2.log

sqlite3
```

Then inside the shell:

```sql
.load /absolute/path/to/libsketch2.so
```

## Creating A Virtual Table

Each virtual table is bound to exactly one Sketch2 dataset INI file. The dataset
must already exist (created through the Sketch2 APIs or tools before using `vlite`).

```sql
CREATE VIRTUAL TABLE nn
USING vlite('/absolute/path/to/my_dataset.ini');
```

The argument is the path to the dataset INI file. Use an absolute path. The
extension opens the dataset in read-only guest mode.

## Virtual Table Schema

The virtual table declares this schema internally:

```sql
CREATE TABLE x(
    query    TEXT HIDDEN,
    k        INTEGER HIDDEN,
    id       INTEGER,
    distance REAL
)
```

| Column     | Type    | Hidden | Direction | Description                                        |
|------------|---------|--------|-----------|----------------------------------------------------|
| `query`    | TEXT    | Yes    | Input     | Query vector: comma- or space-separated values,    | 
|            |         |        |           | or `@path` to load from a file                     |
| `k`        | INTEGER | Yes    | Input     | Number of nearest neighbors to return (default 10) |
| `id`       | INTEGER | No     | Output    | Sketch2 vector ID of a matching neighbor           |
| `distance` | REAL    | No     | Output    | Distance from the query vector to this neighbor    |

**Hidden columns** (`query` and `k`) are inputs. They do not appear in
`SELECT *` output. They are used in `WHERE` clauses to drive the search.

**Visible columns** (`id` and `distance`) are outputs. They appear in result rows.

The `distance` value is computed using whichever distance function was configured
when the dataset was created (L1, L2, or cosine). The distance function cannot be
changed from SQL.

## Dataset INI Parameters

The INI file passed to `USING vlite(...)` describes the dataset. Example:

```ini
[dataset]
dirs=/data/vectors/my_dataset
range_size=10000
dim=128
type=f32
dist_func=l2
```

| Parameter          | Required | Default | Description                                          |
|--------------------|----------|---------|------------------------------------------------------|
| `dirs`             | Yes      | —       | One or more directories containing data files.       |
|                    |          |         | Comma-separated for multiple directories.            |
| `range_size`       | Yes      | —       | Vectors are distributed across files by ID range.    |
|                    |          |         | Each file holds up to this many vector slots.        |
| `dim`              | Yes      | —       | Vector dimension. Must be between 4 and 4096.        |
| `type`             | Yes      | —       | Element data type: `f32`, `f16`, or `i16`.           |
| `dist_func`        | Yes      | —       | Distance function: `l1`, `l2`, or `cos`.             |
| `accumulator_size` | No       | 65536   | In-memory buffer size in bytes for the accumulator.  |
| `data_merge_ratio` | No       | 2       | Merge threshold ratio for delta files.               |

## Environment Variables

These environment variables configure the Sketch2 runtime. Set them before loading
the extension. Once the runtime initializes, changes have no effect.

| Variable                   | Description                                 |
|----------------------------|---------------------------------------------|
| `SKETCH2_CONFIG`           | Path to an INI config file for runtime      |
|                            | settings                                    |
| `SKETCH2_LOG_LEVEL`        | Log level: `DEBUG`, `TRACE`, `INFO`,        |
|                            | `WARN`, `ERROR`                             |
| `SKETCH2_THREAD_POOL_SIZE` | Number of threads in the compute pool       |
| `SKETCH2_LOG_FILE`         | File path for log output                    |

## Query Vector Format

The `query` value can be provided in three forms.

**Comma-separated values:**

```sql
WHERE query = '1.0, 2.0, 3.0, 4.0'
```

**Space-separated values:**

```sql
WHERE query = '1.0 2.0 3.0 4.0'
```

The delimiter is detected automatically: if the string contains a comma, the
comma parser is used; otherwise the space parser is used.

**File path (prefixed with `@`):**

```sql
WHERE query = '@/absolute/path/to/query.txt'
```

When the value starts with `@`, the rest of the string is treated as a file
path. The file is read and its content is used as the query vector text. The
file may contain comma- or space-separated values and may have a trailing
newline, which is stripped automatically.

Requirements:

- Same number of elements as the dataset dimension
- Compatible with the dataset element type (`f32`, `f16`, or `i16`)
- All values must be finite numbers
- For `i16` datasets, use integer values: `'10 20 30 40'` or `'10, 20, 30, 40'`

## SQL Examples

### Basic Nearest Neighbor Search

Find the 5 nearest neighbors to a query vector:

```sql
SELECT id, distance
FROM nn
WHERE query = '1.0, 0.0, 0.0, 0.0' AND k = 5
ORDER BY distance;
```

This returns up to 5 rows, each with a vector `id` and its `distance` from the
query point. Results are sorted nearest-first.

### Controlling The Number Of Results With k

`k` sets how many nearest neighbors the search returns. If omitted, it defaults
to 10.

```sql
-- Explicit k = 3
SELECT id, distance
FROM nn
WHERE query = '15.2, 15.2, 15.2, 15.2' AND k = 3
ORDER BY distance;

-- Default k = 10
SELECT id, distance
FROM nn
WHERE query = '15.2, 15.2, 15.2, 15.2'
ORDER BY distance;
```

If `k` is larger than the number of vectors in the dataset, all vectors are
returned.

### Using MATCH Instead Of Equals

The `MATCH` operator works identically to `=` for the `query` column. It is not
full-text search. It is just an alternative SQL syntax for passing the query vector.

```sql
SELECT id, distance
FROM nn
WHERE query MATCH '1.0, 0.0, 0.0, 0.0' AND k = 5
ORDER BY distance;
```

### LIMIT And OFFSET

`vlite` pushes `LIMIT` and `OFFSET` down into the KNN computation for efficiency.

```sql
-- Get neighbors 11 through 15 (skip first 10, take next 5)
SELECT id, distance
FROM nn
WHERE query MATCH '15.2, 15.2, 15.2, 15.2'
ORDER BY distance
LIMIT 5 OFFSET 10;
```

How pushdown works:

- If `k` is omitted, the effective KNN count is `LIMIT + OFFSET`.
- If `k` is present, the effective count is `min(k, LIMIT + OFFSET)`.
- If neither `k` nor `LIMIT` is given, the default k of 10 is used.

This means `LIMIT 5 OFFSET 10` without explicit `k` computes 15 nearest neighbors
internally, then SQLite applies the offset and limit to the result.

### Ordering By Distance

The virtual table produces results in nearest-first order and advertises this to
SQLite, so `ORDER BY distance` does not cause a re-sort:

```sql
SELECT id, distance
FROM nn
WHERE query = '0.0, 0.0, 0.0, 0.0' AND k = 10
ORDER BY distance;
```

Always include `ORDER BY distance` when you want guaranteed ordering.

### Reading Hidden Columns

The hidden `query` and `k` columns can be selected. This is useful for debugging:

```sql
SELECT id, k, distance
FROM nn
WHERE query MATCH '15.2, 15.2, 15.2, 15.2'
ORDER BY distance
LIMIT 3;
```

The `k` column reports the SQL-facing value (either the explicit `k` from `WHERE`
or the default 10), not the internal pushdown-adjusted count.

### Querying Multiple Datasets

Create separate virtual tables for different datasets:

```sql
CREATE VIRTUAL TABLE image_vectors USING vlite('/data/images.ini');
CREATE VIRTUAL TABLE audio_vectors USING vlite('/data/audio.ini');

-- Search images
SELECT id, distance FROM image_vectors
WHERE query = '0.1, 0.2, 0.3, 0.4' AND k = 5
ORDER BY distance;

-- Search audio
SELECT id, distance FROM audio_vectors
WHERE query = '1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0' AND k = 5
ORDER BY distance;
```

Each virtual table is independent. They can have different dimensions, data types,
and distance functions.

## Joins With Other Tables

The `id` column returned by `vlite` is a Sketch2 vector ID. In most applications
you will have a regular SQLite table that maps these IDs to metadata. Join on `id`
to combine vector similarity search with relational data.

```sql
CREATE TABLE images (
    id       INTEGER PRIMARY KEY,
    filename TEXT NOT NULL,
    label    TEXT
);

SELECT i.filename, i.label, nn.distance
FROM nn
JOIN images AS i ON i.id = nn.id
WHERE nn.query = '0.12, 0.45, 0.78, 0.33' AND nn.k = 10
ORDER BY nn.distance;
```

The virtual table runs the KNN search first (returning at most `k` rows), then
SQLite joins those rows against the `images` table by `id`. Since the KNN result
set is small, the join is fast.

## Distance Functions

The distance function is set when the dataset is created and recorded in the INI
file. It cannot be changed from SQL. All queries against that dataset use the
configured function.

| Function | INI Value | Description                                          |
|----------|-----------|------------------------------------------------------|
| L1       | `l1`      | Manhattan distance. Sum of absolute differences.     |
| L2       | `l2`      | Squared Euclidean distance. Sum of squared differences. |
| Cosine   | `cos`     | Cosine distance. `1 - cosine_similarity`. Range [0, 2]. |

For cosine distance:

- Identical directions produce distance 0
- Orthogonal vectors produce distance 1
- Opposite directions produce distance 2
- Zero vectors: `distance(zero, zero) = 0`, `distance(zero, nonzero) = 1`

## Supported Data Types

| Type  | INI Value | Element Size | Description                              |
|-------|-----------|--------------|------------------------------------------|
| f32   | `f32`     | 4 bytes      | 32-bit float. Always supported.          |
| f16   | `f16`     | 2 bytes      | 16-bit float. Platform-dependent.        |
| i16   | `i16`     | 2 bytes      | Signed 16-bit integer. Always supported. |

The query vector text must use values compatible with the dataset type. For `i16`
datasets, use integers (`'10, 20, 30, 40'`). For `f32` and `f16` datasets, use
floating point values (`'1.0, 2.0, 3.0, 4.0'`).

## Error Cases

| Error                                   | Cause                                              |
|-----------------------------------------|----------------------------------------------------|
| `vlite requires WHERE query = ... or query MATCH ...` | No `query` constraint in `WHERE` clause |
| `vlite query must be a non-empty string`| Empty string passed as query vector                |
| `vlite k must be > 0`                   | `k` set to zero or a negative number               |
| `truncated vector payload`              | Query vector has fewer dimensions than the dataset |
| `invalid f32 token` (or f16, i16)       | Non-numeric text in the query vector               |
| `no such module: vlite`                 | Extension not loaded or `.load` path is wrong      |
| `vlite dataset ini path must not be empty` | Empty path in `USING vlite(...)`                |
| `failed to open file`                   | `@path` query used but file does not exist         |
| `vlite id exceeds SQLite INTEGER range` | Dataset contains IDs > 9223372036854775807         |

## Limitations

- **Read-only.** `INSERT`, `UPDATE`, and `DELETE` are not supported through the
  virtual table. Modify data through the Sketch2 APIs (parasol C API or Python
  wrapper), then query results through `vlite`.
- **Query vector required.** Every `SELECT` must include `WHERE query = ...` or
  `WHERE query MATCH ...`. Full table scans are rejected.
- **Dataset path is fixed.** The INI path is set at `CREATE VIRTUAL TABLE` time
  and cannot be changed afterward. Drop and recreate the virtual table to point
  at a different dataset.
- **Distance function from metadata.** The distance function is determined by the
  dataset, not by the SQL query.
- **No vector payloads.** Only `id` and `distance` are returned. To retrieve the
  actual vector values, use the Sketch2 C API or Python wrapper.
- **ID range limit.** IDs must fit in a SQLite `INTEGER` (signed 64-bit), so the
  maximum Sketch2 ID usable through `vlite` is 9223372036854775807.
- **MATCH is not FTS.** The `MATCH` keyword is just an alternative operator for
  the hidden `query` column. It has nothing to do with SQLite full-text search.
- **Live data visibility.** If the dataset is updated outside SQLite (through the
  Sketch2 APIs), `vlite` sees the current on-disk state on each query because the
  extension reads through the storage layer. No restart is needed.
