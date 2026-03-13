# vlite

`vlite` is a SQLite virtual table extension that exposes Sketch2 KNN search as SQL.

It is a read-only module. You point a virtual table at an existing Sketch2 dataset
INI file, then query nearest neighbors using standard SQL.

## What It Exposes

The module declares this virtual table schema:

```sql
CREATE TABLE x(
    query TEXT HIDDEN,
    k INTEGER HIDDEN,
    id INTEGER,
    distance REAL
)
```

Meaning:

- `query`: hidden input column. This is the query vector encoded as text.
- `k`: hidden input column. This is the requested number of nearest neighbors.
- `id`: output column. Sketch2 vector id.
- `distance`: output column. Distance from the query vector.

The distance function is not chosen in SQL. It comes from the Sketch2 dataset
metadata in the referenced INI file.

## Prerequisites

Before using `vlite`, you need an existing Sketch2 dataset created by the
regular Sketch2 APIs or tools. The important input to `vlite` is the dataset
INI file, for example:

```ini
[dataset]
dirs=/tmp/my_dataset
range_size=1000
dim=4
type=f32
dist_func=l1
```

`vlite` opens this dataset in read-only mode through the Sketch2 storage layer.

## Building

In a debug build, the important artifacts are typically:

- SQLite shell: `bin-dbg/sqlite3`
- Extension library: `build-dbg/lib/libvlite.so`

Build them with:

```bash
cmake -S . -B build-dbg -DCMAKE_BUILD_TYPE=Debug
cmake --build build-dbg --target sqlite3_cli vlite
```

## Loading In SQLite

From the bundled SQLite shell:

```sql
.load /absolute/path/to/build-dbg/lib/libvlite.so
```

Then create a virtual table bound to a dataset INI:

```sql
CREATE VIRTUAL TABLE nn
USING vlite('/absolute/path/to/my_dataset.ini');
```

The virtual table keeps that dataset path and a cached initialized Sketch2
`Dataset` object for all later queries.

## Basic Query

Use `WHERE query = ...` or `WHERE query MATCH ...` to pass the query vector.

Example:

```sql
SELECT id, distance
FROM nn
WHERE query = '1.0, 0.0, 0.0, 0.0' AND k = 5
ORDER BY distance;
```

Equivalent `MATCH` form:

```sql
SELECT id, distance
FROM nn
WHERE query MATCH '1.0, 0.0, 0.0, 0.0' AND k = 5
ORDER BY distance;
```

`MATCH` here is not full-text search. It is only an alternative SQL operator for
supplying the hidden `query` vector to the virtual table.

## Query Vector Format

`query` must use the same text vector format accepted by Sketch2 parsing code.

Examples:

```sql
'1.0, 2.0, 3.0, 4.0'
'[ 1.0, 2.0, 3.0, 4.0 ]'
```

The vector must match the dataset:

- same dimension
- compatible element type
- finite numeric values

If parsing fails, SQLite returns an error from the virtual table.

## Choosing k

You can specify `k` explicitly:

```sql
SELECT id, distance
FROM nn
WHERE query MATCH '15.2, 15.2, 15.2, 15.2' AND k = 3
ORDER BY distance;
```

If `k` is omitted, `vlite` defaults to `10`.

The hidden `k` column is also readable, which can be useful for debugging:

```sql
SELECT id, k, distance
FROM nn
WHERE query MATCH '15.2, 15.2, 15.2, 15.2'
ORDER BY distance
LIMIT 3;
```

When you read back the hidden `k` column, it reports the SQL-facing requested
value:

- the explicit `k` from `WHERE ... AND k = ...`, or
- the default `10` when `k` is omitted

It does not report the internal pushdown-adjusted count used for `LIMIT/OFFSET`.

## LIMIT And OFFSET

`vlite` supports SQLite `LIMIT` and `OFFSET`, and pushes them down into the
internal Sketch2 KNN request when possible.

Behavior:

- If `k` is omitted, `vlite` uses `LIMIT + OFFSET` as the effective KNN count.
- If `k` is present, `vlite` uses `min(k, LIMIT + OFFSET)`.
- If neither `k` nor `LIMIT` is given, the default `k` is `10`.

Example:

```sql
SELECT id, distance
FROM nn
WHERE query MATCH '15.2, 15.2, 15.2, 15.2'
ORDER BY distance
LIMIT 5 OFFSET 10;
```

That query asks `vlite` to compute only the first `15` neighbors instead of
defaulting to `10` or scanning an arbitrary larger result set.

## Ordering

The natural result order is nearest-first, and `vlite` advertises support for:

```sql
ORDER BY distance
```

Use explicit ordering in SQL when you care about result order.

Recommended form:

```sql
SELECT id, distance
FROM nn
WHERE query = '...'
ORDER BY distance
LIMIT 10;
```

## End-To-End Example

```sql
.load /home/user/sketch2/build-dbg/lib/libvlite.so

CREATE VIRTUAL TABLE nn
USING vlite('/home/user/data/example.ini');

SELECT id, distance
FROM nn
WHERE query MATCH '0.0, 0.0, 0.0, 0.0' AND k = 3
ORDER BY distance;
```

Example output shape:

```text
1|0.0
3|4.0
7|8.0
```

## Using Multiple Datasets

Each virtual table instance is bound to one dataset INI path.

Example:

```sql
CREATE VIRTUAL TABLE images USING vlite('/data/images.ini');
CREATE VIRTUAL TABLE audio  USING vlite('/data/audio.ini');

SELECT id, distance FROM images
WHERE query = '...'
ORDER BY distance
LIMIT 5;

SELECT id, distance FROM audio
WHERE query = '...'
ORDER BY distance
LIMIT 5;
```

## Limitations

- Read-only only. `INSERT`, `UPDATE`, and `DELETE` are not supported.
- A query vector is required. Full table scans are rejected.
- The dataset path is fixed when the virtual table is created.
- The distance function comes from dataset metadata, not from the SQL query.
- `vlite` returns ids and distances only. It does not expose full vector payloads.
- `id` is exposed as SQLite `INTEGER`, so Sketch2 ids must be `<= 9223372036854775807`.
- `MATCH` is just an input operator for the hidden `query` column. It is not FTS.

## Error Cases

Typical failures:

- missing `query` constraint
- invalid dataset INI path
- malformed query vector text
- wrong vector dimension or type for the dataset
- invalid `k` value such as `k <= 0`

Example of an invalid query:

```sql
SELECT id, distance
FROM nn
WHERE k = 5;
```

This fails because `query` is required.

## Practical Notes

- Prefer absolute paths for both `.load` and the dataset INI path.
- Keep using the same dataset creation and update flow as regular Sketch2.
- If you update the dataset outside SQLite, later `vlite` queries see the current
  on-disk Sketch2 state because the extension opens the dataset through the
  storage layer for each search.
