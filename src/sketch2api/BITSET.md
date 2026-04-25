# Sketch2 API Allowlist Blob Format

Sketch2 supports one allowlist format: a serialized chunked-Roaring BLOB.

`sk_knn_items(...)` and `sk_knn_vector_items(...)` accept:

- `allowed_ids_blob == nullptr && allowed_ids_blob_size == 0`: no filter
- otherwise: a 32-byte-aligned chunked-Roaring BLOB

Non-null allowlist buffers that are not 32-byte aligned are rejected. A header
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

`bitset_agg(id)` builds this serialized format inside an API-owned opaque
allowlist object and returns that object to SQLite as a typed pointer. SQLite
does not allocate the serialized buffer and releases the object by calling the
Sketch2 API release function registered with the pointer value.

Ordinary SQL `BLOB` values are still accepted by `allowed_ids` only when SQLite
provides a 32-byte-aligned pointer.

SQL `NULL` in `allowed_ids` still means no filter.
