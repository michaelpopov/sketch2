# CompactIds

`CompactIds` is the storage-layer representation for sorted vector ids and sorted tombstone ids.
It exists to keep id metadata compact on disk and in memory while still supporting fast ordered access.

There are two related types:

- `CompactIdsBuilder`: append-only builder used while writing or merging.
- `CompactIds`: read-only container used after a file has been opened.

Both represent ids as:

- one `uint64_t base`
- one strictly increasing `uint32_t` offset per id, where `id = base + offset`

This saves memory compared with storing every id as `uint64_t`, as long as all ids in one set fit within a `uint32_t` span from the first id.

## Serialized Form

Each serialized `CompactIds` section starts with a 24-byte header:

- `encoding`
- `count`
- `max_offset`
- `payload_size`
- `base`

The payload uses one of two encodings:

- `Offsets32`: raw `uint32_t` offsets
- `Bitset`: one bit per possible offset

`CompactIdsBuilder` and `CompactIds` both choose the smaller canonical payload automatically. Bitset is used only when it is smaller than raw offsets.

At runtime, `CompactIds` is always materialized as ordered `uint32_t` offsets in memory, even if the file stored a bitset payload. This keeps `id(index)`, `lower_bound_index(id)`, and merge-style ordered scans simple and fast.

## File Layout

In a `.data` or `.delta` file, the id metadata lives in the trailer after the vector records and optional cosine inverse norms:

1. vector records
2. optional cosine inverse norms
3. alignment padding to `kIdsAlignment`
4. active ids `CompactIds`
5. deleted ids `CompactIds`

`DataMetadataLayout` in `data_file_layout.h` computes the offsets for these sections.

`DataFileHeader.count` is the number of active ids.
`DataFileHeader.deleted_count` is the number of deleted ids.
`DataFileHeader.min_id` and `DataFileHeader.max_id` describe only the active ids.

## DataWriter

`DataWriter::load()` builds both id sections before writing the trailer.

Procedure:

1. Scan the sorted `InputReaderView`.
2. Split ids into:
   - `active_ids` when the row has vector data
   - `deleted_ids` when `reader.is_no_data(i)` is true
3. Validate that ids are strictly increasing.
4. Build the file header from:
   - active min/max id
   - active count
   - deleted count
5. Stream only live vector records into the data section.
6. Optionally write cosine inverse norms for live rows.
7. Write id-trailer alignment padding.
8. Serialize `active_ids`, then serialize `deleted_ids`.

Important details:

- `CompactIdsBuilder` is used directly, so `DataWriter` does not keep a second full `std::vector<uint64_t>` copy of ids.
- If all rows are deletes, header `min_id` and `max_id` are written as zero.
- Builder append fails if ids are not strictly increasing or if the span from the first id exceeds `uint32_t`.

## DataReader

`DataReader::init()` memory-maps the file, validates the header and layout, and then parses the two `CompactIds` sections from the trailer.

Procedure:

1. `mmap()` the file and validate the header, version, type, dimensions, stride, and flags.
2. Use `compute_data_metadata_layout()` to locate the trailer.
3. Read the first `CompactIds` section into `ids_`.
4. Read the second `CompactIds` section into `deleted_ids_`.
5. Require the parsed trailer to consume the file exactly.
6. Require `ids_.count()` and `deleted_ids_.count()` to match the header counts.
7. Copy cosine inverse norms into a heap buffer when present.

How `CompactIds` is used after load:

- `id(index)` and `id_unchecked(index)` provide ordered id access.
- `lower_bound_index(id)` powers `DataReader::get(id)`.
- `deleted_id(index)` exposes persisted tombstones.
- `check_consistency()` verifies that active ids and deleted ids are each strictly sorted and disjoint.

When a base reader is opened with an attached delta reader, `init_delta()` walks:

- base `ids_`
- delta `ids_`
- delta `deleted_ids_`

and marks base rows hidden when the delta overwrites or deletes the same id.

## DataMerger

`DataMerger` uses `CompactIdsBuilder` while producing the merged file.

### Live ids

`MergeOutputWriter` owns `output_ids_`, a `CompactIdsBuilder` for the active ids of the new file.
Each time a surviving record is written, `write_binary_record()` first appends its id to `output_ids_`, then writes the vector bytes.

This means the final active-id section is built incrementally in output order while the merge stream is produced.

### Delete ids

The merge logic still uses plain sorted `std::vector<uint64_t>` delete lists while deciding which records survive, because the merge loop only needs sequential comparisons.

After the delete set is known:

- data-file merges pass an empty `CompactIdsBuilder`, because the output `.data` file does not preserve tombstones
- delta-file merges build one `CompactIdsBuilder` from the final delete list and reuse it when writing the trailer

The delta path does not rebuild deleted ids inside `write_ids_section()` anymore. The builder is prepared once by the caller and then written directly.

### Merge flow

The high-level merge procedure is:

1. Normalize updater input into sorted live items plus a sorted delete list.
2. Run `merge_records()` over source ids, updater ids, and deletes.
3. For each surviving live row, append the id to `output_ids_` and write the vector record.
4. After all vectors are written, call `write_ids_section()` to write:
   - cosine inverse norms
   - alignment padding
   - active ids from `output_ids_`
   - deleted ids from the provided builder
5. Patch the header with final active `min_id`, `max_id`, and `count`.
6. For delta merges, also patch `deleted_count`.

### Delta-specific delete handling

For delta merges, the final tombstone set is not just the updater delete list.
`build_delta_deletes()`:

- keeps source tombstones that were not resurrected by live updater rows
- unions them with updater tombstones
- removes duplicates

That final sorted delete list is what gets serialized into the deleted-id `CompactIds` section of the merged delta file.

## Why This Design

This design keeps the important operations efficient:

- compact on-disk metadata
- lower in-memory id storage than `uint64_t` arrays
- fast `id(index)` access
- fast binary search by id
- simple linear merge walks over sorted ids
- no extra full-id copy on write and merge paths

In short, `CompactIds` is the sorted-id spine of the storage format: `DataWriter` builds it, `DataReader` uses it for lookup and iteration, and `DataMerger` carries it forward while producing new files.
