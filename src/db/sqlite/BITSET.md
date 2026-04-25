# SQLite Bitset Notes

This document covers the SQLite-facing pieces of bitset filtering.

The binary allowlist blob format itself is owned by Sketch2 API and is
documented in [src/sketch2api/BITSET.md](/home/mpopov/projects/sketch2/src/sketch2api/BITSET.md).

## SQL Producer

SQLite exposes:

```sql
SELECT bitset_agg(id) FROM some_table;
```

`bitset_agg(id)`:

- ignores `NULL`
- accepts only SQLite `INTEGER`
- rejects negative ids
- accepts ids in any order
- returns an empty `BLOB` for empty input
- emits a process-local SQLite pointer value for non-empty input

The aggregate keeps SQLite-specific type validation in `vlite.cpp`, but it now
delegates construction to Sketch2 API helpers. Internally it uses chunked
Roaring id sets so the memory footprint follows the selected ids rather than
the span between the smallest and largest id.

Recommended shape:

```sql
SELECT bitset_agg(id)
FROM some_table
WHERE ...;
```

## SQL Consumer

`vlite` accepts the produced value through hidden column `allowed_ids`:

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

Accepted SQL values for `allowed_ids`:

- result of `bitset_agg(id)`: apply filtering
- `BLOB`: apply filtering
- `NULL`: no filter

Other non-`BLOB` and non-`NULL` values are rejected with a SQLite error.
The process-local pointer value returned by `bitset_agg(id)` is accepted
internally by `vlite` even though it is not a `BLOB`.
