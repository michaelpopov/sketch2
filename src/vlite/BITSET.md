# Bitset Filter Format And Processing

This document describes the bitset BLOB produced by `bitset_agg(id)` and how
`vlite` can consume it through the hidden `allowed_ids` column.

## Purpose

`allowed_ids` is an optional SQL-side candidate filter.

- SQL builds a compact bitset of allowed vector IDs.
- `vlite` receives that bitset as a `BLOB` in `xFilter()`.
- Search logic can later use it to skip disallowed IDs.

Current status: filtering is implemented in scanner and applied when
`allowed_ids` is provided.

## SQL Interface

### Aggregate producer

```sql
SELECT bitset_agg(id) FROM some_table;
```

`bitset_agg(id)` behavior:

- ignores `NULL`
- accepts only SQLite `INTEGER` values
- rejects negative ids
- returns a dense bitset as `BLOB`
- returns empty `BLOB` for empty input

### Virtual-table consumer

```sql
SELECT id, distance
FROM vlite AS v
WHERE v.match_expr MATCH :query
  AND v.allowed_ids = (
        SELECT bitset_agg(id)
        FROM labels
        WHERE label = 3
      )
ORDER BY distance;
```

`allowed_ids` contract:

- accepted types: `BLOB` or `NULL`
- `NULL` means "no filter"
- non-`BLOB` and non-`NULL` values are rejected with SQLite error

## Bitset Layout

For an ID `id >= 0`:

- `byte_index = id / 8`
- `bit_index = id % 8`
- mask is least-significant-bit first: `(1u << bit_index)`

So:

- ID `0` is byte `0`, bit `0` (mask `0x01`)
- ID `1` is byte `0`, bit `1` (mask `0x02`)
- ID `7` is byte `0`, bit `7` (mask `0x80`)
- ID `8` is byte `1`, bit `0` (mask `0x01`)

Example:

- IDs: `{0, 1, 8}`
- bytes: `[0x03, 0x01]`
- SQL check:

```sql
SELECT hex(bitset_agg(id))
FROM (SELECT 0 AS id UNION ALL SELECT 1 UNION ALL SELECT 8);
-- 0301
```

## Access In `vlite`

`xFilter()` in `src/vlite/vlite.cpp` extracts:

- `const void* allowed_ids_blob`
- `int allowed_ids_blob_size`
- `bool has_allowed_ids`

These are passed into `Scanner::find_items(...)` as `BitsetFilter`.
Filtering is applied during scan loops, so disallowed ids are skipped before
heap insertion.

## Safe Bit-Test Helpers (Recommended)

No dedicated helper exists yet in the codebase for this BLOB format.
Use small local helpers like these:

```cpp
#include <cstdint>
#include <cstddef>

inline bool bitset_contains_id(const uint8_t* data, size_t size, uint64_t id) {
    if (data == nullptr) {
        return false;
    }
    const uint64_t byte_index = id >> 3;  // id / 8
    if (byte_index >= size) {
        return false;
    }
    const uint8_t mask = static_cast<uint8_t>(1u << (id & 7u));
    return (data[byte_index] & mask) != 0;
}

inline bool allowed_by_filter(const void* blob, int blob_size, uint64_t id) {
    if (blob == nullptr || blob_size <= 0) {
        return false;
    }
    return bitset_contains_id(
        static_cast<const uint8_t*>(blob),
        static_cast<size_t>(blob_size),
        id);
}
```

Behavior notes:

- IDs beyond `blob_size * 8` are treated as not allowed.
- Empty blob means no IDs allowed by the blob itself.
- If SQL passed `NULL`, treat it as "filter not present" and bypass checks.

## Applying During Search

Current decision flow:

1. `has_allowed_ids == false` -> run normal search.
2. `has_allowed_ids == true` -> apply blob membership checks and keep only
   allowed ids.

## Limits And Error Conditions

`bitset_agg` currently enforces size bounds tied to SQLite result limits:

- max bitset bytes ~= `INT_MAX`
- max supported ID ~= `INT_MAX * 8 - 1`

On invalid input or overflow conditions, it returns SQLite errors.
On allocation failure, it returns SQLite OOM (`sqlite3_result_error_nomem`).

## Test Coverage

Covered by:

- `src/vlite/utest_vlite.cpp` (unit tests)
- `src/pytest/test_integ_vlite_interfaces.py` (integration tests)

These tests validate format, validation errors, and active `allowed_ids`
filtering semantics.
