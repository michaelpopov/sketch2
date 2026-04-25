# ID Handling in Storage (RoaringIds)

This document provides a comprehensive overview of how vector IDs are handled within the storage layer of `sketch2`. It covers the core data structures, the binary storage layout, read/write workflows, and the optimizations used in hot path computations. If you are new to the codebase and want to understand how IDs flow through the storage system, start here.

## 1. Overview of `RoaringIds`

At the heart of ID management are `RoaringIds` and `RoaringIdsBuilder` (`src/core/utils/roaring_ids.h`). They wrap the `CRoaring` library, which provides a highly optimized implementation of Roaring Bitmaps.

Vectors in `sketch2` are identified by a user-provided `uint64_t` ID. While the underlying CRoaring bitmap operates on 32-bit integers natively, `RoaringIds` manages 64-bit IDs by storing them as offsets from a base ID (passed as `base` to the API; persisted as `min_range_id` in the file header). This delta encoding enables the system to support a 64-bit ID space while leveraging the memory and performance efficiency of 32-bit Roaring bitmaps, provided the maximum ID minus the base within a single file fits within a 32-bit range.

A `RoaringIds` instance is a single-bitmap, read-stage container with no notion of active vs. deleted. Each storage file holds **two** `RoaringIds` instances: one for active IDs (`ids_`) and one for tombstones (`deleted_ids_`). New or derived sets are assembled with `RoaringIdsBuilder`, then finalized into `RoaringIds`.

**Key responsibilities of `RoaringIds`:**
- Maintaining a sorted, compact set of `uint64_t` IDs as a single bitmap.
- Providing sequential iterators (`RoaringIds::Iterator`), exact membership/index lookup (`find_index`), and positional lookup (`id(index)`, `id_unchecked(index)`).
- Serializing to and deserializing from memory-mapped regions ("frozen views") via `init_frozen_view`, without copying the bitmap payload during reads.

**Key responsibilities of `RoaringIdsBuilder`:**
- Owning the mutable construction stage (`init`, `add`, `load`, `union_in_place`, `andnot_in_place`).
- Supporting optional buffered `add()` ingestion via `init_buffered`, where buffered IDs are sorted and flushed in batches.
- Compacting the bitmap and moving it into a read-stage `RoaringIds` with `build()`.

## 2. Storage Layout and Serialization

Data files (`.data`) and delta files (`.delta`) store vector records alongside their corresponding IDs. The binary layout is defined in `src/core/storage/data_file.h`.

When a file is written (via `DataWriter` or `DataMerger`), the vector records are written first in a contiguous block. After all vectors are flushed, the IDs are appended as trailers at the end of the file. The exact section ordering is documented as the v13 payload contract in `data_file.h`:

1. Aligned vector records with optional inline norm.
2. Region-alignment padding.
3. Frozen `RoaringIds` for active IDs — omitted when `count` is zero.
4. Region-alignment padding.
5. Frozen `RoaringIds` for deleted IDs — omitted when `deleted_count` is zero.

The `DataFileHeader` tracks these sections via the following ID-related fields:
- `count` & `deleted_count`: Number of active and deleted IDs. A zero value indicates the corresponding trailer is absent.
- `ids_offset` & `ids_bytes`: The location and size of the active IDs frozen view.
- `deleted_ids_offset` & `deleted_ids_bytes`: The location and size of the deleted IDs frozen view (tombstones).
- `min_id` & `max_id`: The smallest and largest active IDs in the file (inclusive bounds).
- `min_range_id`: The base ID used for delta-encoding the 32-bit bitmaps.

During the writing phase, both `DataWriter` and `DataMerger` use `RoaringIdsBuilder` instances for active and deleted IDs. Initial writes do this while scanning an `InputReaderView`; merge writes do this while streaming surviving rows through `MergeOutputWriter`. Once all records are flushed, builders are finalized with `build()`, which compacts the bitmap and moves it into `RoaringIds` for serialization into the trailing sections of the file.

## 3. Read Path and Hot Loop Scanning

### Deserialization
When a `DataReader` opens a file, it validates the header and maps the vector and ID sections. Instead of parsing the ID trailers into new bitmap payloads, it initializes `RoaringIds` directly over the memory-mapped regions using `init_frozen_view`. This zero-copy-payload approach keeps opening large datasets fast and memory-efficient.

### Lookup Modes
`RoaringIds` exposes several access patterns, and the caller should choose based on workload:

- `find_index(id, &index)` is for point lookup by vector ID. `DataReader::get` uses it to avoid a lower-bound/select pair.
- `id(index)` and `id_unchecked(index)` are positional lookups. They use CRoaring `select` internally, so they are convenient but should not be used in tight sequential scan loops.
- `RoaringIds::Iterator` is for ordered scans. It advances through the bitmap and is the preferred primitive when records are visited in increasing position order.

### Hot Path Scans
In computation hot loops (e.g., `src/core/compute/scanner_scan_loops.h`), the scanner must iterate through vector records and their corresponding IDs simultaneously.

Calling `id(index)` or `id_unchecked(index)` on a `RoaringIds` wrapper performs a `select()` operation on the underlying bitmap. `select()` is non-constant time, with cost depending on the container type the position falls in (binary search for array containers, rank-table walk for bitmap containers, linear over runs for run containers). It is far too slow for the inner scan loop.

To avoid that cost, scanner loops use `DataReader::BaseScanCursor`. This cursor maintains an active `RoaringIds::Iterator`. When advancing to the next visible record, the cursor calls `advance_id_iter_to_index`, which calls `next()` on the iterator until it reaches the target index. Across a sequential scan, each ID is advanced at most once, so ID access is amortized constant-time and avoids repeated `select()` calls.

## 4. Updates, Deletes, and Visibility

The storage engine uses an overlay model. A base `.data` file can have an attached `.delta` file containing recent insertions, updates, and deletions.

Each data file (base or delta) loads its own active `ids_` and `deleted_ids_` frozen views when it is opened — that step is not delta-specific. The delta-specific step happens in `DataReader::init_delta`: it walks the **base file's** active `ids_` cursor twice in parallel — first against the **delta's** `deleted_ids_`, then against the **delta's** active `ids_`. For every base-row index whose ID matches an entry on either side (a tombstone or an update), the corresponding bit in the base reader's `DynamicBitset` (`changed_bitset_`) is set.

During iteration (e.g., `DataReader::OrderedIterator` or hot loop scanning), any index where the `changed_bitset_` is `true` is considered "hidden" and skipped. The `next_visible_base_index_unchecked` function uses fast bitwise operations on the `DynamicBitset` to jump over sequences of hidden rows efficiently.

## 5. Merge Path (`DataMerger`)

The `DataMerger` (`src/core/storage/data_merger.cpp`) consolidates base files and deltas or merges new input streams into existing files. Because both the base file and the updater stream are sorted by ID, the merge is an efficient $O(N)$ linear process.

The merger utilizes specialized synchronized cursors to walk the sorted streams simultaneously:
- **`DataReaderLiveRowCursor`**: Walks visible source rows.
- **`InputReaderUpdaterCursor`**: Walks new/updated rows.
- **`DeltaDeleteCursor`**: Produces the merged stream of deleted IDs. It takes three inputs — the source's deleted IDs, the updater's deleted IDs, and the updater's *live* row stream — and uses the third input to perform resurrection (see workflow item 1).

**The Merge Workflow:**
1. **Resurrection Handling**: The `DeltaDeleteCursor` ensures that if an ID was deleted in an older file but is subsequently inserted as a live row in the updater, it is dropped from the merged tombstone stream (resurrected). This is the role of `skip_resurrected_source_deletes`.
2. **Conflict Resolution**: As the cursors advance side-by-side, if the source ID equals the updater ID, the source record is shadowed, and the updater record takes precedence.
3. **Conflict Failure**: If the same ID appears as both a *live* updater row and an updater *delete*, the inputs are contradictory and the merge aborts with an error rather than silently choosing one interpretation.
4. **Filtering**: If the current live ID matches the current ID in the `DeltaDeleteCursor`, the record is skipped (deleted).
5. **Output Generation**: Surviving records are streamed sequentially into a `MergeOutputWriter`, which buffers the IDs in memory. Once all vectors are written, the finalized `RoaringIds` bitmaps are appended to the file.

## 6. Important Invariants

The Roaring-backed layout depends on a few invariants that writers, readers, and tests enforce:

- IDs in one persisted file are represented as `uint32_t` offsets from `min_range_id`; any ID below the base or more than `UINT32_MAX` above it is rejected.
- Active IDs and deleted IDs are stored in separate `RoaringIds` trailers and must be disjoint for a consistent reader.
- Header counts (`count`, `deleted_count`) must match the cardinality of the corresponding Roaring trailers.
- Empty ID sets are represented by zero trailer bytes. A non-zero count with zero bytes, or a zero count with non-zero bytes, is invalid.
- Frozen Roaring trailers must start at 32-byte-aligned addresses. The file layout aligns the ID sections to `kDataRegionAlignment`, which is a multiple of 32.
- Base and delta files attached in one `DataReader`, or merged through the persisted-file merge path, must share the same `min_range_id`.
- Readers validate the current file-format version and reject legacy raw-ID layouts.
