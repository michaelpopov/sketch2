# Sketch2 SQLite (`vlite`) Guide

`vlite` is a read-only SQLite virtual table for KNN search over an existing
Sketch2 dataset.

- Input: query vector (`query` or `match_expr`), optional `k`, optional `allowed_ids`
- Output: `id`, `distance`
- Distance metric comes from dataset metadata (`dist_func`)

## Build And Load

The extension is part of `libsketch2.so`.

```sql
.load /absolute/path/to/libsketch2.so
```

If your SQLite build has `SQLITE_OMIT_LOAD_EXTENSION`, runtime loading is disabled.
Check with:

```bash
sqlite3 ':memory:' 'SELECT sqlite_compileoption_used("OMIT_LOAD_EXTENSION")'
```

If the query returns `1`, use the SQLite binary built in this repository.

## Create A Virtual Table

`vlite` binds one virtual table to one dataset INI path.

```sql
CREATE VIRTUAL TABLE nn
USING vlite('/absolute/path/to/dataset.ini');
```

The dataset must already exist.

## Virtual Table Schema

`vlite` declares:

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

Column notes:

- `query` / `match_expr` (hidden input): query vector text
- `k` (hidden input): top-k size, default `10`
- `allowed_ids` (hidden input): optional bitset BLOB filter
- `id` (output): vector id
- `distance` (output): distance according to dataset metric

`SELECT *` only returns visible output columns (`id`, `distance`).

## Query Formats

Vector text supports:

- Comma-delimited: `'1.0, 2.0, 3.0, 4.0'`
- Space-delimited: `'1.0 2.0 3.0 4.0'`
- File reference: `'@/absolute/path/to/query.txt'`

For `i16` datasets, use integer values.

## Basic Queries

```sql
SELECT id, distance
FROM nn
WHERE query = '1.0, 0.0, 0.0, 0.0' AND k = 5
ORDER BY distance;
```

`MATCH` is also supported:

```sql
SELECT id, distance
FROM nn
WHERE match_expr MATCH '1.0, 0.0, 0.0, 0.0' AND k = 5
ORDER BY distance;
```

`MATCH` here is not FTS; it is just an accepted operator for the hidden query
columns.

## LIMIT / OFFSET Pushdown

`vlite` pushes `LIMIT/OFFSET` into effective KNN count.

- If `k` omitted: internal count is `LIMIT + OFFSET`
- If `k` present: internal count is `min(k, LIMIT + OFFSET)`
- If no `k` and no `LIMIT`: default `k=10`

Example:

```sql
SELECT id, distance
FROM nn
WHERE query = '0.0, 0.0, 0.0, 0.0'
ORDER BY distance
LIMIT 5 OFFSET 10;
```

## `allowed_ids` Filtering

`allowed_ids` is optional.

- `NULL` means no filtering
- `BLOB` applies filtering
- non-`BLOB` and non-`NULL` values are rejected

Example with SQL-side producer:

```sql
SELECT id, distance
FROM nn
WHERE match_expr MATCH '2.1, 2.1, 2.1, 2.1'
  AND k = 3
  AND allowed_ids = (
        SELECT bitset_agg(id)
        FROM (SELECT 0 AS id)
      )
ORDER BY distance;
```

This returns only neighbors whose ids are present in the bitset.

Use `bitset_agg(id)` to build the BLOB. For format details, see `src/vlite/BITSET.md`.

## Dataset Metadata (`dataset.ini`)

Typical section:

```ini
[dataset]
dirs=/data/my_dataset
range_size=10000
dim=128
type=f32
dist_func=l2
```

Important keys:

- `dirs`: one or more data directories
- `range_size`: id-range sharding size
- `dim`: vector dimension (`4..4096`)
- `type`: `f32`, `f16`, `i16`
- `dist_func`: `l1`, `l2`, `cos`

## Runtime Environment Variables

Set before loading extension:

- `SKETCH2_CONFIG`
- `SKETCH2_LOG_LEVEL`
- `SKETCH2_THREAD_POOL_SIZE`
- `SKETCH2_LOG_FILE`

## Distance Functions

Distance is fixed by dataset metadata.

- `l1`: Manhattan distance
- `l2`: squared Euclidean distance
- `cos`: cosine distance (`1 - cosine_similarity`)

For cosine:

- same direction -> `0`
- orthogonal -> `1`
- opposite direction -> `2`

## Common Errors

- `vlite requires WHERE query = ... or query MATCH ...`
- `vlite query must be a non-empty string`
- `vlite k must be > 0`
- `vlite allowed_ids must be a BLOB or NULL`
- parse errors like `invalid f32 token` / `truncated vector payload`
- `no such module: vlite` when extension is not loaded

## Limits

- Read-only virtual table (`INSERT/UPDATE/DELETE` not supported)
- Query constraint required (`query` or `match_expr`)
- Result ids must fit SQLite `INTEGER` range
- Distance metric cannot be overridden in SQL
