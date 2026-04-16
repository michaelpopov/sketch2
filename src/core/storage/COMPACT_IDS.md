# CompactIdsExt

`CompactIdsExt` is the storage-layer representation for sorted active ids and
sorted tombstone ids.
It is the orchestrator for the compact-id format: it chooses the payload
encoding when writing, dispatches by encoding when mapping, and exposes one
uniform sorted-id API to storage code.

The concrete payload backends are:

- `CompactIdsOffsets`: raw strictly increasing `uint32_t` offsets from a base id
- `CompactIdsBitset`: one bit per possible offset in the covered span
- `CompactIdsMisses`: the offsets that are absent inside an otherwise dense span

`CompactIdsAccumulator` is a write-path helper that collects sorted ids as
offsets from the first id. Once accumulation is finished, `CompactIdsExt`
chooses the smallest backend representation and writes it to disk.

All variants represent ids relative to one `uint64_t base`, where the logical id
space is `id = base + offset`.

## Serialized Form

Each serialized `CompactIdsExt` section starts with a 24-byte header:

- `encoding`
- `count`
- `aux_data`
- `payload_size`
- `base`

`aux_data` is format-specific:

- `Offsets32`: `max_offset`
- `Bitset`: `max_offset`
- `Misses32`: `miss_count`

The payload encoding is selected by `CompactIdsExt`:

- `Offsets32`: stores one `uint32_t` per id
- `Bitset`: stores one bit per possible offset in the covered span
- `Misses32`: stores one `uint32_t` per missing offset in the covered span

At load time, `CompactIdsExt::map()` reads `encoding` and forwards to the
matching backend parser. After mapping, callers still use the same high-level
operations such as `id(index)`, `lower_bound_index(id)`, `contains(id)`, and
ordered iteration regardless of which payload was stored on disk.

## File Layout

In a `.data` or `.delta` file, the id metadata lives in the trailer after the
vector records and optional cosine inverse norms:

1. vector records
2. optional cosine inverse norms
3. alignment padding to `kIdsAlignment`
4. active ids `CompactIdsExt`
5. deleted ids `CompactIdsExt`

`DataMetadataLayout` in `data_file_layout.h` computes the offsets for these
sections.

`DataFileHeader.count` is the number of active ids.
`DataFileHeader.deleted_count` is the number of deleted ids.
`DataFileHeader.min_id` and `DataFileHeader.max_id` describe only the active ids.

## DataWriter

`DataWriter::write()` builds both id sections before writing the trailer.

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
8. Materialize `CompactIdsExt` for active ids and deleted ids, then serialize both.

Important details:

- `CompactIdsAccumulator` is used while scanning ids.
- `CompactIdsExt` is materialized once per section right before trailer output.
- If all rows are deletes, header `min_id` and `max_id` are written as zero.
- Accumulation fails if ids are not strictly increasing or if the span from the
  first id exceeds `uint32_t`.

## DataReader

`DataReader::init()` memory-maps the file, validates the header and layout, and
then parses the two `CompactIdsExt` sections from the trailer.

Procedure:

1. `mmap()` the file and validate the header, version, type, dimensions,
   stride, and flags.
2. Use `compute_data_metadata_layout()` to locate the trailer.
3. Read the first `CompactIdsExt` section into `ids_`.
4. Read the second `CompactIdsExt` section into `deleted_ids_`.
5. Require the parsed trailer to consume the file exactly.
6. Require `ids_.count()` and `deleted_ids_.count()` to match the header counts.
7. Copy cosine inverse norms into a heap buffer when present.

How `CompactIdsExt` is used after load:

- `id(index)` and `id_unchecked(index)` provide ordered id access.
- `lower_bound_index(id)` powers `DataReader::get(id)`.
- `deleted_id(index)` exposes persisted tombstones.
- `check_consistency()` verifies that active ids and deleted ids are each
  strictly sorted and disjoint.

When a base reader is opened with an attached delta reader, `init_delta()` walks:

- base `ids_`
- delta `ids_`
- delta `deleted_ids_`

and marks base rows hidden when the delta overwrites or deletes the same id.

## DataMerger

`DataMerger` uses `CompactIdsAccumulator` while producing the merged file.

### Live ids

`MergeOutputWriter` owns `output_ids_`, a `CompactIdsAccumulator` for the active
ids of the new file.
Each time a surviving record is written, `write_binary_record()` first appends
its id to `output_ids_`, then writes the vector bytes.

This means the final active-id section is built incrementally in output order
while the merge stream is produced.

### Delete ids

The merge logic still uses plain sorted `std::vector<uint64_t>` delete lists
while deciding which records survive, because the merge loop only needs
sequential comparisons.

After the delete set is known:

- data-file merges pass an empty `CompactIdsExt`, because the output `.data`
  file does not preserve tombstones
- delta-file merges build one `CompactIdsAccumulator` from the final delete list
  and materialize `CompactIdsExt` when writing the trailer

The delta path does not rebuild deleted ids inside `write_ids_section()`
anymore. The accumulator is prepared once by the caller and then written
directly as `CompactIdsExt`.

### Merge flow

The high-level merge procedure is:

1. Normalize updater input into sorted live items plus a sorted delete list.
2. Run `merge_records()` over source ids, updater ids, and deletes.
3. For each surviving live row, append the id to `output_ids_` and write the
   vector record.
4. After all vectors are written, call `write_ids_section()` to write:
   - cosine inverse norms
   - alignment padding
   - active ids from `output_ids_`, materialized as `CompactIdsExt`
   - deleted ids from the provided `CompactIdsExt`
5. Patch the header with final active `min_id`, `max_id`, and `count`.
6. For delta merges, also patch `deleted_count`.

### Delta-specific delete handling

For delta merges, the final tombstone set is not just the updater delete list.
`build_delta_deletes()`:

- keeps source tombstones that were not resurrected by live updater rows
- unions them with updater tombstones
- removes duplicates

That final sorted delete list is what gets serialized into the deleted-id
`CompactIdsExt` section of the merged delta file.

## Why This Design

This design keeps the important operations efficient:

- compact on-disk metadata
- a format-specific payload chosen to minimize storage
- fast `id(index)` access
- fast binary search by id
- simple linear merge walks over sorted ids
- no extra full-id copy on write and merge paths

In short, `CompactIdsExt` is the sorted-id spine of the storage format:
`DataWriter` builds it, `DataReader` uses it for lookup and iteration, and
`DataMerger` carries it forward while producing new files.
