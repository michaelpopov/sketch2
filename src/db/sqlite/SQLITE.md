# Sketch2 SQLite (`vlite`) Guide

`vlite` is a read-only SQLite virtual table for KNN search over an existing
Sketch2 dataset.

- Input: query vector (`query` or `match_expr`), optional `k`, optional `allowed_ids`
- Output: `id`, `score`
- Score metric comes from dataset metadata (`dist_func`)

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
    score REAL
)
```

Column notes:

- `query` / `match_expr` (hidden input): query vector text
- `k` (hidden input): top-k size, default `10`
- `allowed_ids` (hidden input): optional bitset filter
- `id` (output): vector id
- `score` (output): score according to dataset metric

`SELECT *` only returns visible output columns (`id`, `score`).

## Query Formats

Vector text supports:

- Comma-delimited: `'1.0, 2.0, 3.0, 4.0'`
- Space-delimited: `'1.0 2.0 3.0 4.0'`
- File reference: `'@/absolute/path/to/query.txt'`

For `i16` datasets, use integer values.

## Basic Queries

```sql
SELECT id, score
FROM nn
WHERE query = '1.0, 0.0, 0.0, 0.0' AND k = 5
ORDER BY score;
```

`MATCH` is also supported:

```sql
SELECT id, score
FROM nn
WHERE match_expr MATCH '1.0, 0.0, 0.0, 0.0' AND k = 5
ORDER BY score;
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
SELECT id, score
FROM nn
WHERE query = '0.0, 0.0, 0.0, 0.0'
ORDER BY score
LIMIT 5 OFFSET 10;
```

## `allowed_ids` Filtering

`allowed_ids` is optional.

- `NULL` means no filtering
- the typed pointer returned by `bitset_agg(id[, name])` applies filtering
- `BLOB` applies filtering when SQLite provides a 32-byte-aligned pointer
- non-`BLOB` and non-`NULL` values are rejected

`bitset_agg(id[, name])` accepts ids in any order. The optional `name` parameter
must be a non-empty string when present:

```sql
SELECT bitset_agg(id)
FROM labels
WHERE label = 3;

SELECT bitset_agg(id, 'label_3')
FROM labels
WHERE label = 3;
```

The aggregate returns an API-owned typed pointer that wraps the serialized
chunked-Roaring format documented in `src/sketch2api/BITSET.md`. The opaque
object may be heap-backed or mmap-backed. SQLite calls Sketch2's release
function when that pointer value is destroyed, and the release path handles
either storage kind.

When `name` is provided on a row observed by the aggregate, Sketch2 also
publishes the serialized filter as `<spill_dir>/<name>.bitset`, where
`spill_dir` comes from `bitset_filter.spill_dir` or
`SKETCH2_BITSET_FILTER_SPILL_DIR`. Names may contain only ASCII letters, digits,
and underscores, and empty names are rejected. The published file persists after
the SQL pointer value is released and is replaced atomically when rebuilt with
the same name. All-`NULL` id input can still publish an empty named file.

The first non-`NULL` `name` observed by the step function wins for naming. Later
non-`NULL` names in the same group must pass the same value; a different value is
rejected instead of silently publishing under the first name. A zero-row SQL
aggregate still returns a valid empty filter, but publishes no named file because
SQLite never calls the step function and Sketch2 cannot observe the name
argument.

Raw SQL `BLOB` bitset filters are still accepted only when SQLite provides a
32-byte-aligned caller-owned buffer. Spillover applies to the opaque
`bitset_agg(id[, name])` result, not to arbitrary BLOB values.

`bitset_drop(name)` deletes the persistent file for a named filter:

```sql
SELECT bitset_drop('label_3');
```

It returns `1` when a file was removed and `0` when the named file was already
absent. `NULL`, empty, and otherwise invalid names are rejected.

Mapped spill only avoids allocating the final serialized bitset filter buffer with
`aligned_alloc`. The aggregate still accumulates its `ChunkedBits` /
`RoaringIdsBuilder` working state in memory before serialization.

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
- `SKETCH2_BITSET_FILTER_SPILL_THRESHOLD_BYTES`
- `SKETCH2_BITSET_FILTER_SPILL_DIR`

The bitset filter spill settings apply only to the finalized serialized buffer, not
to the in-memory `bitset_agg(id)` builder state.

## Score Functions

The score function is fixed by dataset metadata.

- `l1`: Manhattan score
- `l2`: squared Euclidean score
- `cos`: cosine score (`1 - cosine_similarity`)

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
- Score metric cannot be overridden in SQL
