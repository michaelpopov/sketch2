# Sketch2 API Bitset Filter Blob Format

Sketch2 supports one bitset filter format: a serialized chunked-Roaring BLOB.

`sk_knn_items(...)` and `sk_knn_vector_items(...)` accept:

- `allowed_ids_blob == nullptr && allowed_ids_blob_size == 0`: no filter
- otherwise: a caller-owned 32-byte-aligned chunked-Roaring BLOB

Non-null bitset filter buffers that are not 32-byte aligned are rejected. A header
with `chunk_count = 0` is an empty filter and matches no ids.

## Binary Layout

All integer fields are little-endian.

Header, 16 bytes:

- `magic`: `uint32`, bytes `SKCB`
- `version`: `uint16`, currently `1`
- `chunk_bits`: `uint16`, currently `20`
- `chunk_count`: `uint64`

Directory entry, 24 bytes each:

- `chunk_id`: `uint64`
- `payload_offset`: `uint64`, byte offset from the start of the BLOB
- `payload_size`: `uint64`

The full layout is:

```text
Header
Directory[chunk_count]
padding as needed
Payloads
```

Directory entries are sorted by `chunk_id` ascending and unique. Payload
offsets are 32-byte aligned. Each payload is CRoaring frozen serialization for
one chunk, with base id `chunk_id << 20`.

## SQLite

`bitset_agg(id[, name])` builds this serialized format inside an API-owned opaque
bitset filter object and returns that object to SQLite as a typed pointer. The
object may keep the serialized bytes in heap memory, in an unlinked temporary
mapped file, or in a named mapped file when `name` is provided. SQLite does not
allocate the serialized buffer and releases the object by calling the Sketch2 API
release function registered with the pointer value.

When `name` is provided on any row observed by the aggregate, Sketch2 publishes
the serialized filter as `<spill_dir>/<name>.bitset` and leaves that file in
place after the SQLite pointer is released. This includes groups where all
observed ids are `NULL`, which publish an empty named filter. Names may contain
only ASCII letters, digits, and underscores, and empty names are rejected.

Within one aggregate group, the first non-`NULL` `name` observed by the step
function wins for naming; later non-`NULL` names must pass the same value or the
aggregate is rejected. If a SQL aggregate has zero rows, SQLite never calls the
step function, so Sketch2 cannot observe the name argument and no named file is
published.

`sk_bitset_filter_builder_finish()` uses the same opaque ownership model for
in-process callers. Pass the returned object to `sk_knn_items_bitset_filter()` and
release it with `sk_release_bitset_filter()`, which handles either heap-backed or
mmap-backed storage.

`sk_bitset_filter_load(name)` maps `<spill_dir>/<name>.bitset`, validates the
serialized filter, and returns the same opaque object shape. Pass it to
`sk_knn_items_bitset_filter()` and release it with `sk_release_bitset_filter()`.
Code that shares one opaque object across multiple owners may call
`sk_retain_bitset_filter()` for each additional owner and balance each retain
with `sk_release_bitset_filter()`.

`sk_bitset_filter_drop(name)` deletes `<spill_dir>/<name>.bitset` for a named
filter and reports whether a file was removed. Missing files are not errors.

Mapped spill only avoids allocating the final serialized bitset filter buffer with
`aligned_alloc`. The aggregation builder still accumulates `ChunkedBits` /
`RoaringIdsBuilder` state in memory before serialization, so very large
bitset filters can still run out of memory before spill is reached.

Ordinary SQL `BLOB` values are still accepted by `allowed_ids` only when SQLite
provides a 32-byte-aligned pointer.

SQL `NULL` in `allowed_ids` still means no filter.
