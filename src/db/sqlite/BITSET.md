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
- requires ids in non-decreasing order
- returns an empty `BLOB` for empty input
- emits a blob in the Sketch2 API bitset format

Recommended shape:

```sql
SELECT bitset_agg(id)
FROM (
    SELECT id
    FROM some_table
    WHERE ...
    ORDER BY id
);
```

## SQL Consumer

`vlite` accepts the produced blob through hidden column `allowed_ids`:

```sql
SELECT id, score
FROM nn AS n
WHERE n.query = :query
  AND n.k = :k
  AND n.allowed_ids = (
        SELECT bitset_agg(id)
        FROM (
            SELECT id
            FROM labels
            WHERE label = 3
            ORDER BY id
        )
      )
ORDER BY score;
```

Accepted SQL values for `allowed_ids`:

- `BLOB`: apply filtering
- `NULL`: no filter

Non-`BLOB` and non-`NULL` values are rejected with a SQLite error.
