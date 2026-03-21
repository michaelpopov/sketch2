# SQLite Integration

This document explains how SQLite integration in Sketch2 is implemented, what
functionality it supports, and how to use the SQLite virtual table interface.

## Overview

Sketch2 provides SQLite integration through the `vlite` virtual table module.

`vlite` is exposed from:

- `libsketch2.so`

The module is read-only. It lets SQLite execute KNN queries against an existing
Sketch2 dataset while SQLite continues to handle SQL parsing, joins, relational
filters, and result shaping.

This design keeps responsibilities clear:

- Sketch2 owns vector storage and nearest-neighbor execution
- SQLite owns SQL execution and metadata joins

## How SQLite Integration Works

The SQLite entry point is implemented inside the shared Sketch2 runtime
library. When SQLite loads `libsketch2.so`, the extension registers the
`vlite` module and related helper functions.

After that, SQLite can create a virtual table bound to a Sketch2 dataset ini
file:

```sql
CREATE VIRTUAL TABLE nn
USING vlite('/absolute/path/to/dataset.ini');
```

That virtual table keeps the dataset binding and routes KNN requests through
the Sketch2 storage and compute layers.

## Runtime Artifact

The extension entry points are exported from `libsketch2.so`.

Typical library locations:

- release: `bin/libsketch2.so`
- debug: `bin-dbg/libsketch2.so`

If you use the bundled SQLite shell built by the repository, a typical debug
shell path is:

- `bin-dbg/sqlite3`

## Runtime Initialization

SQLite integration uses the same process-wide Sketch2 runtime initialization as
the C API and Python wrapper.

Important environment variables:

- `SKETCH2_CONFIG`
- `SKETCH2_LOG_LEVEL`
- `SKETCH2_THREAD_POOL_SIZE`
- `SKETCH2_LOG_FILE`

Set them before loading the extension:

```bash
export SKETCH2_LOG_LEVEL=DEBUG
export SKETCH2_THREAD_POOL_SIZE=8
export SKETCH2_LOG_FILE=/tmp/sketch2.log
```

## Prerequisites

Before using `vlite`, you need an existing Sketch2 dataset and its ini file.

Typical dataset metadata:

```ini
[dataset]
dirs=/data/my_dataset
range_size=10000
dim=128
type=f32
dist_func=l2
```

Important fields:

- `dirs`: data directories
- `range_size`: id-range partition size
- `dim`: vector dimension
- `type`: `f32`, `f16`, or `i16`
- `dist_func`: `l1`, `l2`, or `cos`

## Loading The Extension

Load the Sketch2 extension into SQLite:

```sql
.load /absolute/path/to/build/lib/libsketch2.so
```

If your system SQLite binary disables loadable extensions, use the SQLite
binary built by this repository instead.

## Creating A Virtual Table

Each `vlite` virtual table is bound to one Sketch2 dataset ini path.

Example:

```sql
CREATE VIRTUAL TABLE nn
USING vlite('/absolute/path/to/dataset.ini');
```

You can create multiple virtual tables for multiple datasets:

```sql
CREATE VIRTUAL TABLE images USING vlite('/data/images.ini');
CREATE VIRTUAL TABLE audio  USING vlite('/data/audio.ini');
```

## Virtual Table Schema

`vlite` declares this schema:

```sql
CREATE TABLE x(
    query TEXT HIDDEN,
    match_expr TEXT HIDDEN,
    k INTEGER HIDDEN,
    allowed_ids BLOB HIDDEN,
    id INTEGER,
    distance REAL
)
```

Meaning:

- `query`: hidden input query vector
- `match_expr`: alternate hidden input query vector
- `k`: hidden input top-k count
- `allowed_ids`: optional hidden input bitset filter
- `id`: output vector id
- `distance`: output distance

The visible output columns are `id` and `distance`.

## Supported Query Functionality

SQLite integration supports:

- KNN queries through `query = ...` or `match_expr MATCH ...`
- explicit `k`
- `LIMIT` and `OFFSET`
- joins with regular SQLite tables
- optional candidate filtering through `allowed_ids`
- SQL-side bitset generation through `bitset_agg(id)`

The distance function is not selected in SQL. It comes from the Sketch2
dataset metadata.

## Query Vector Format

Query vectors use the normal Sketch2 text format.

Supported forms include:

- comma-delimited values
- space-delimited values
- file-reference input using `@/absolute/path/to/query.txt`

Examples:

```sql
'1.0, 2.0, 3.0, 4.0'
'1.0 2.0 3.0 4.0'
'@/absolute/path/to/query.txt'
```

For `i16` datasets, use integer query values.

## Basic Queries

Use `query`:

```sql
SELECT id, distance
FROM nn
WHERE query = '1.0, 2.0, 3.0, 4.0' AND k = 5
ORDER BY distance;
```

Use `MATCH`:

```sql
SELECT id, distance
FROM nn
WHERE match_expr MATCH '1.0, 2.0, 3.0, 4.0' AND k = 5
ORDER BY distance;
```

`MATCH` here is only an accepted operator for passing the query vector. It is
not full-text search.

## Choosing `k`

If `k` is omitted, `vlite` defaults to `10`.

Example:

```sql
SELECT id, distance
FROM nn
WHERE query = '0.0, 0.0, 0.0, 0.0'
ORDER BY distance
LIMIT 10;
```

`LIMIT` and `OFFSET` are pushed down into the effective internal KNN request
when possible.

## Joining With Metadata Tables

SQLite integration becomes most useful when KNN results are joined with regular
relational metadata.

Create a metadata table:

```sql
CREATE TABLE items_meta (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT,
    author TEXT
);
```

Join KNN results with metadata:

```sql
SELECT m.title, m.category, n.distance
FROM nn AS n
JOIN items_meta AS m ON m.id = n.id
WHERE n.query = '1.0, 2.0, 3.0, 4.0' AND n.k = 5
ORDER BY n.distance;
```

Return ids together with metadata:

```sql
SELECT m.id, m.title, m.author, n.distance
FROM nn AS n
JOIN items_meta AS m ON m.id = n.id
WHERE n.match_expr MATCH '0.1, 0.2, 0.3, 0.4'
  AND n.k = 10
ORDER BY n.distance;
```

Apply relational filtering together with KNN:

```sql
SELECT m.id, m.title, n.distance
FROM nn AS n
JOIN items_meta AS m ON m.id = n.id
WHERE n.match_expr MATCH '0.1, 0.2, 0.3, 0.4'
  AND n.k = 20
  AND m.category = 'books'
ORDER BY n.distance
LIMIT 10;
```

This is the main value of the SQLite integration: Sketch2 computes nearest
neighbors, then SQLite combines those ids with relational data.

## `allowed_ids` Filtering

`allowed_ids` is an optional hidden column that constrains candidate ids before
the final KNN results are returned.

Rules:

- `NULL` means no filtering
- a `BLOB` applies filtering
- non-`BLOB` and non-`NULL` inputs are rejected

The expected BLOB format is the bitset produced by `bitset_agg(id)`.

## `bitset_agg(id)`

`bitset_agg(id)` is an aggregate helper function exported by the extension.
It builds a dense bitset BLOB from integer ids so that SQL can prepare an
`allowed_ids` filter without leaving SQLite.

Example:

```sql
SELECT hex(bitset_agg(id))
FROM (
    SELECT 0 AS id
    UNION ALL
    SELECT 1
    UNION ALL
    SELECT 8
);
```

## `allowed_ids` Query Examples

Filter candidates to a fixed SQL-generated set:

```sql
SELECT n.id, n.distance
FROM nn AS n
WHERE n.match_expr MATCH '2.1, 2.1, 2.1, 2.1'
  AND n.k = 3
  AND n.allowed_ids = (
        SELECT bitset_agg(id)
        FROM (SELECT 0 AS id)
      )
ORDER BY n.distance;
```

Build the bitset from a metadata table:

```sql
SELECT n.id, n.distance
FROM nn AS n
WHERE n.match_expr MATCH '2.1, 2.1, 2.1, 2.1'
  AND n.k = 10
  AND n.allowed_ids = (
        SELECT bitset_agg(id)
        FROM items_meta
        WHERE category = 'books'
      )
ORDER BY n.distance;
```

Keep normal behavior explicitly:

```sql
SELECT n.id, n.distance
FROM nn AS n
WHERE n.query = '10.0, 10.0, 10.0, 10.0'
  AND n.k = 5
  AND n.allowed_ids = CAST(NULL AS BLOB)
ORDER BY n.distance;
```

Combine metadata filtering and result joins:

```sql
SELECT m.id, m.title, n.distance
FROM nn AS n
JOIN items_meta AS m ON m.id = n.id
WHERE n.match_expr MATCH '0.5, 0.5, 0.5, 0.5'
  AND n.k = 20
  AND n.allowed_ids = (
        SELECT bitset_agg(id)
        FROM items_meta
        WHERE author = 'Alice'
      )
ORDER BY n.distance;
```

This pattern is useful when the SQL layer knows a prefiltered candidate set
and wants Sketch2 to run KNN only within that subset.

## Error Handling

Typical failures include:

- missing query constraint
- invalid dataset ini path
- malformed query vector text
- wrong dimension or type for the dataset
- `k <= 0`
- `allowed_ids` value that is neither `BLOB` nor `NULL`

Example of an invalid query:

```sql
SELECT id, distance
FROM nn
WHERE k = 5;
```

Example of an invalid `allowed_ids` value:

```sql
SELECT id
FROM nn
WHERE query = '10.0, 10.0, 10.0, 10.0'
  AND k = 1
  AND allowed_ids = 'not_a_blob';
```

## Limitations

Current limits include:

- read-only virtual table
- query vector is required
- dataset path is fixed at virtual-table creation time
- distance function comes from dataset metadata
- output columns are limited to `id` and `distance`
- ids must fit SQLite `INTEGER`

## Notes

- prefer absolute paths for both `.load` and dataset ini files
- use SQLite joins to combine KNN ids with relational metadata
- use `allowed_ids` when SQL can produce a candidate subset efficiently
- `vlite` is intended to complement SQLite tables, not replace them
