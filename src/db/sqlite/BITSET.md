# SQLite Bitset Notes

SQLite exposes:

```sql
SELECT bitset_agg(id) FROM some_table;
SELECT bitset_agg(id, 'name') FROM some_table;
SELECT bitset_drop('name');
```

`bitset_agg(id[, name])`:

- ignores `NULL`
- accepts only SQLite `INTEGER`
- accepts an optional non-empty string `name`
- rejects negative ids
- accepts ids in any order
- returns an API-owned typed pointer to a serialized chunked-Roaring bitset_filter
- returns a valid empty bitset filter for empty input
- when `name` is provided on an observed row, writes
  `<spill_dir>/<name>.bitset` and leaves that file in place after the pointer is
  released, including all-`NULL` id input
- a zero-row SQL aggregate still returns a valid empty filter, but publishes no
  named file because SQLite never calls the aggregate step function
- the first non-`NULL` `name` observed by the step function wins for naming;
  later non-`NULL` names must pass the same `name`, or the aggregate is rejected
- rejects an empty `name`

`bitset_drop(name)`:

- accepts a non-empty string `name`
- deletes `<spill_dir>/<name>.bitset`
- returns `1` when a file was removed
- returns `0` when the named file was already absent
- rejects `NULL`, empty, or invalid names

The underlying serialized format is documented in
[src/sketch2api/BITSET.md](/home/mpopov/projects/sketch2/src/sketch2api/BITSET.md).

`allowed_ids` accepts:

- `NULL`: no filter
- the typed pointer returned by `bitset_agg(id[, name])`: apply filtering
- `BLOB`: apply filtering only when SQLite provides a 32-byte-aligned pointer

SQLite never allocates or frees the `bitset_agg(id[, name])` buffer directly. It stores
the typed pointer temporarily and calls Sketch2's release function when the
value is destroyed. The pointed-to opaque bitset filter object may be heap-backed
or mmap-backed, and `sk_release_bitset_filter()` releases either kind correctly.

Raw `BLOB` bitset filters remain caller-owned 32-byte-aligned buffers. The mapped
spill path is only for opaque bitset filters produced by `bitset_agg(id[, name])` /
`sk_bitset_filter_builder_finish()`. It avoids allocating the final serialized
bitset filter buffer with `aligned_alloc`; the aggregate still accumulates builder
state in memory before that final buffer is created.
