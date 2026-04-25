# Sketch2 API Bitset Filter Format

This document owns the binary allowlist bitset contract consumed by the
Sketch2 API query path.

The format is currently used by:

- `sk_knn_items(...)` through `allowed_ids_blob` and `allowed_ids_blob_size`
- `sk_bitset_create(...)`
- `sk_bitset_load(...)`
- `sk_bitset_builder_add(...)` + `sk_bitset_builder_finish(...)`

SQLite's `bitset_agg(id)` now uses a process-local chunked Roaring allowlist
for direct query execution. Persisted allowlist files and callers that pass
`allowed_ids_blob` to the C API still use the dense bitset format documented
here.

## Purpose

The bitset blob represents a compact allowlist of vector ids.

- if `allowed_ids_blob == nullptr && allowed_ids_blob_size == 0`, no filter is applied
- otherwise the blob is interpreted as a binary allowlist
- ids missing from the blob are treated as not allowed

## Binary Layout

For non-empty output, the blob format is:

- bytes `[0..7]`: `base_id` (`uint64_t`, first represented id)
- bytes `[8..]`: packed dense bitset bytes

For an id `id >= base_id`:

- `relative_id = id - base_id`
- `byte_index = relative_id / 8`
- `bit_index = relative_id % 8`
- mask is least-significant-bit first: `(1u << bit_index)`

So:

- id `base_id` is byte `0`, bit `0` (`0x01`)
- id `base_id + 1` is byte `0`, bit `1` (`0x02`)
- id `base_id + 7` is byte `0`, bit `7` (`0x80`)
- id `base_id + 8` is byte `1`, bit `0` (`0x01`)

Example:

- ids: `{10, 11, 18}`
- `base_id`: `10`
- bitset bytes: `[0x03, 0x01]`
- full blob bytes: `0A000000000000000301`

## Validation Rules

`sk_knn_items(...)` applies these rules:

- `nullptr` with size `0` means "filter not present"
- non-`nullptr` with size `< 8` is rejected as too small
- `nullptr` with non-zero size is invalid
- empty represented range after the header means no ids are allowed by the blob itself

Behavior notes:

- ids below `base_id` are not allowed
- ids beyond the represented range are not allowed
- a `NULL` SQL value should be translated by the caller into `nullptr, 0`
- builder-produced blobs currently support ids only up to `100000000`

## SQLite Interop

`src/db/sqlite/vlite.cpp` forwards the hidden `allowed_ids` column to
`sk_knn_items(...)`.

`src/db/sqlite/bitset_agg(id)` accepts ids in any order and produces a
process-local value consumed by the same SQLite statement. It is not a
persistable dense bitset BLOB. To persist an allowlist with
`sk_bitset_create(...)`, build or load the dense format described above.

Example:

```sql
SELECT id, score
FROM nn AS n
WHERE n.query = :query
  AND n.k = :k
  AND n.allowed_ids = (
        SELECT bitset_agg(id)
        FROM labels
        WHERE label = 3
      )
ORDER BY score;
```
