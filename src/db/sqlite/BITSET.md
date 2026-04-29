# SQLite Bitset Notes

SQLite exposes:

```sql
SELECT bitset_agg(id) FROM some_table;
SELECT bitset_agg(id, 'name') FROM some_table;
SELECT bitset_load('name');
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
- evicts the named filter from the process-wide mapped-file cache
- returns `1` when a file was removed
- returns `0` when the named file was already absent
- rejects `NULL`, empty, or invalid names

`bitset_load(name)`:

- accepts a non-empty string `name`
- maps `<spill_dir>/<name>.bitset`
- returns an API-owned typed pointer that can be passed to `allowed_ids`
- reuses a process-wide cache of loaded named filters across prepared
  statements and SQLite connections in the same process
- rejects `NULL`, empty, invalid, missing, or malformed named filters

Named filters created by `bitset_agg(id, name)` and loaded by
`bitset_load(name)` are cached by name after validation. Cache eviction does not
invalidate typed pointers that SQLite has already returned; those pointers keep
their mapped storage alive until SQLite destroys the value. Use
`bitset_drop(name)` to remove both the on-disk file and the process-wide cache
entry.

The underlying serialized format is documented in
[src/sketch2api/BITSET.md](/home/mpopov/projects/sketch2/src/sketch2api/BITSET.md).

`allowed_ids` accepts:

- `NULL`: no filter
- the typed pointer returned by `bitset_agg(id[, name])` or `bitset_load(name)`:
  apply filtering
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
