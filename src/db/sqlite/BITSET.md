# SQLite Bitset Notes

SQLite exposes:

```sql
SELECT bitset_agg(id) FROM some_table;
```

`bitset_agg(id)`:

- ignores `NULL`
- accepts only SQLite `INTEGER`
- rejects negative ids
- accepts ids in any order
- returns an API-owned typed pointer to a serialized chunked-Roaring allowlist
- returns a valid empty-filter allowlist for empty input

The underlying serialized format is documented in
[src/sketch2api/BITSET.md](/home/mpopov/projects/sketch2/src/sketch2api/BITSET.md).

`allowed_ids` accepts:

- `NULL`: no filter
- the typed pointer returned by `bitset_agg(id)`: apply filtering
- `BLOB`: apply filtering only when SQLite provides a 32-byte-aligned pointer

SQLite never allocates or frees the `bitset_agg(id)` buffer directly. It stores
the typed pointer temporarily and calls Sketch2's release function when the
value is destroyed.
