# storage

This directory contains the persisted-storage layer for vectors.

Its job is to:

- ingest text or binary input
- write the project’s binary storage format
- expose persisted files through memory-mapped readers
- buffer recent updates and deletes in memory
- recover buffered updates after a crash
- merge persisted state back into compact files
- present a dataset as one logical collection to the compute/scanner layer

The current implementation is no longer just an early design sketch. It has a
concrete file format, a dataset coordinator, an in-memory accumulator, and a
WAL-backed write path.

## Design Goals

The storage layer is built around a few assumptions.

1. Most vectors arrive through bulk or batched ingest.
2. The bulk of persisted data should remain compact and cheap to scan.
3. Updates and deletes happen, but are small relative to the full dataset.
4. Query code should read a logically current view without rewriting files for
   every mutation.
5. Persisted state should be recoverable after a crash when owner-mode updates
   are still buffered in memory.

That leads to a hybrid model:

- most data lives in immutable `.data` files
- recent changes are persisted in `.delta` files or in the in-memory
  accumulator
- the accumulator is protected by a write-ahead log
- merges periodically fold changes back into compact persisted files

## Main Components

### InputGenerator

`input_generator.{h,cpp}` creates deterministic text or binary input files used
by tests and development flows.

It is not part of the production read path. Its purpose is to generate source
data in the same import format that `InputReader` understands.

### InputReader

`input_reader.{h,cpp}` parses the import format used for bulk loading.

It:

- memory-maps the source file
- reads the dataset header (`type`, `dim`)
- indexes record ids and payload spans
- exposes rows through lightweight views without rescanning the file

`InputReaderView` is a range slice over one `InputReader`. Dataset store code
uses it to process one id range at a time.

### DataWriter

`data_writer.{h,cpp}` converts an `InputReaderView` into the on-disk binary
data-file format.

It writes:

- the file header
- aligned vector records
- optional cosine inverse norms
- sorted active ids
- sorted deleted ids

### DataReader

`data_reader.{h,cpp}` exposes a persisted `.data` or `.delta` file as a
memory-mapped read view.

It:

- validates the binary layout
- caches pointers to the vector, cosine, id, and delete sections
- supports point lookup by id
- supports scanning through iterators
- can layer a delta reader over a base reader

When a delta is attached, `DataReader` hides base rows shadowed by the delta.

### Accumulator

`accumulator.{h,cpp}` stores recent owner-mode updates and deletes in memory.

It keeps:

- active vector ids
- one aligned packed byte buffer for vector payloads
- optional per-vector cosine inverse norms
- a deleted-id set

The accumulator gives the dataset a cheap mutable layer without rewriting
persisted files for every change.

### Accumulator WAL

`accumulator_wal.{h,cpp}` persists accumulator mutations in append-only form.

It exists so that:

- acknowledged in-memory updates survive crashes
- owner-mode datasets can reconstruct buffered state on startup

The WAL stores:

- a WAL header with type and dimension
- checksummed add/delete records

On replay, torn trailing records are truncated and valid records are applied in
order.

### DataMerger

`data_merger.{h,cpp}` combines:

- base data files
- delta files
- accumulator slices

into new persisted files.

It handles both:

- rewriting a compact `.data` file
- building or updating a `.delta` file

### Dataset

`dataset.{h,cpp}` is the main storage coordinator.

It owns:

- dataset metadata
- file-range mapping
- owner/guest mode
- the accumulator and its WAL
- cached dataset items
- cached readers

It exposes the high-level operations used by the rest of the system:

- initialize a dataset
- bulk store input files
- add or delete vectors in owner mode
- flush the accumulator
- merge deltas
- prepare a read-consistent view
- iterate persisted ranges

## Persisted Artifacts

The storage layer uses several file types.

### 1. Data Files

Extension: `.data`

These are the main persisted files. They hold the compact base state for one id
range.

### 2. Delta Files

Extension: `.delta`

These hold persisted changes for one id range:

- replacement vectors
- new vectors in that range
- deleted ids

They are meant to be smaller than the base `.data` file and are eventually
merged back into it.

### 3. Temporary and Merge Files

Extensions:

- `.temp`
- `.merge`

These are internal staging files used while writing or merging a range.

They let the code build the next version safely and then rename it into place.

### 4. Per-Range Lock Files

Extension: `.lock`

Each range operation takes a file lock on `<file_id>.lock` so concurrent writes
to the same range do not corrupt persisted state.

### 5. Dataset Owner Lock

File name: `sketch2.owner.lock`

This lock lives in the first dataset directory and ensures only one owner-mode
dataset instance performs modifications at a time.

### 6. Accumulator WAL

File name: `sketch2.accumulator.wal`

This file lives in the first dataset directory and persists in-memory
accumulator mutations.

## Partitioning by Id Range

The dataset is partitioned by `range_size`.

For any vector id:

```text
file_id = id / range_size
```

That file id determines:

- which logical file range owns the id
- which directory stores the range:

```text
dir = dirs[file_id % dirs.size()]
```

This gives the storage layer a stable mapping:

- a vector id always belongs to one range
- a range always maps to one file id
- a file id always maps to one directory slot

That stability is important for:

- splitting bulk ingest
- targeted merges
- caching readers
- accumulator flushes by range

## Binary Data File Format

The on-disk data-file layout is defined by:

- `data_file.h`
- `data_file_layout.h`

The file layout is:

```text
[DataFileHeader]
[alignment padding]
[aligned vector records]
[optional cosine inverse norms]
[alignment padding for ids]
[active ids]
[deleted ids]
```

The header includes:

- file magic
- file kind/version
- min and max active id
- active vector count
- deleted id count
- data type
- dimension
- vector section offset
- vector stride
- flags

### Vector Storage

Vectors are stored in a fixed aligned stride rather than packed tightly.

That means:

- each persisted vector record begins at a predictable offset
- readers can compute addresses cheaply
- SIMD code sees a stable per-record layout

The stride is:

- at least the true vector byte size
- rounded up to the configured data alignment

### Id Sections

There are two sorted `uint64_t` tables:

- active ids
- deleted ids

The active id position matches:

- the vector position in the vector section
- the cosine inverse norm position, when that section exists

### Cosine Metadata

For cosine datasets, persisted data files store one `float` inverse norm per
active vector.

Each value is:

```text
1 / ||vector||
```

Zero vectors store `0.0`.

This section is optional at the file-format level, but cosine datasets require
it on persisted files because the scanner uses it for the fast cosine path.

## Reading Model

### DataReader as a Persisted View

`DataReader` is the low-level read abstraction over one persisted file.

It provides:

- `get(id)` for point lookup
- `at(index)` for positional access
- `begin()` for the visible row stream
- `base_begin()` and `delta_begin()` for ordered base/delta scans

### Base + Delta Overlay

When a delta reader is attached to a base reader:

- base rows shadowed by delta updates/deletes are hidden
- base iteration skips hidden rows
- delta rows are exposed after base rows in the general iterator
- ordered iterators still let callers walk base and delta streams separately

This split is important for scanner logic, which wants:

- one logical visible view
- but also ordered base and delta streams for skip-list handling

### Consistency Checks

`DataReader` validates:

- file magic, type, and version
- vector stride and offsets
- file size versus computed section layout
- cosine metadata layout
- compatibility between a base reader and its delta reader

This prevents malformed files from silently entering the query path.

## Mutable State: Accumulator

The accumulator is the in-memory mutable layer for owner-mode datasets.

It is sized by `accumulator_size` and tracks current usage. When there is not
enough capacity for another mutation, the dataset flushes it to persisted files.

### What It Stores

The accumulator keeps:

- active vector ids
- a packed aligned vector buffer
- optional cosine inverse norms
- deleted ids

### Mutation Rules

- `add_vector(id, data)` inserts or replaces the active value for `id`
- `delete_vector(id)` removes an active buffered value and records a tombstone
- adding an id removes it from the deleted set if present
- deleting an id removes any buffered active value if present

For cosine datasets, the accumulator recomputes and stores the inverse norm for
every active buffered vector.

### WAL Integration

The accumulator can attach a WAL. Once attached:

- adds append WAL add records
- deletes append WAL delete records
- startup replay rebuilds accumulator state
- successful persisted flushes reset the WAL

That means owner-mode datasets can recover pending buffered state after a crash
instead of losing acknowledged updates.

## Dataset as the Main Coordinator

`Dataset` is the storage subsystem entrypoint used by higher layers.

### Metadata

`DatasetMetadata` includes:

- `dirs`
- `type`
- `dist_func`
- `dim`
- `range_size`
- `data_merge_ratio`
- `accumulator_size`

This metadata can be provided directly or loaded from an ini file.

### Owner and Guest Modes

Datasets run in one of two modes.

#### Owner

Owner mode can:

- bulk store input files
- add vectors
- delete vectors
- flush the accumulator
- merge deltas

It also owns:

- the owner lock
- the accumulator
- the accumulator WAL

#### Guest

Guest mode is read-only.

It rejects:

- `store()`
- `store_accumulator()`
- `merge()`
- `add_vector()`
- `delete_vector()`

`set_guest_mode()` also refuses to switch if there are pending accumulator
updates, because that would discard mutable state.

### Read Preparation

`prepare_read_state()` is the point where owner-mode reads ensure that pending
accumulator/WAL state is attached and visible before query code starts reading.

That is why scanner code can combine:

- persisted dataset readers
- accumulator iteration

and still see a logically fresh dataset view.

### Reader and Item Caches

`Dataset` caches:

- discovered dataset items
- opened readers by file id

This avoids rescanning directories and remapping files on every repeated access.

The caches are invalidated after successful write/merge operations.

## Write and Merge Flows

### Bulk Store

`Dataset::store()` reads one input file, determines which id ranges are touched,
and processes each touched range independently.

Per range, the code:

1. builds a range view over the input
2. writes a temporary persisted file
3. decides whether to:
   - create a new `.data`
   - create/update a `.delta`
   - merge directly into `.data`

If a thread pool is configured, independent range tasks may run in parallel.

### Accumulator Flush

`Dataset::store_accumulator()` groups buffered ids and deletes by range and then
persists each affected range independently.

After all affected ranges are persisted successfully:

- the WAL is reset
- the accumulator is cleared

### Forced Delta Merge

`Dataset::merge()` scans dataset items and folds every existing delta file into
its corresponding data file.

This is the explicit “compact everything” path.

### Merge Heuristics

The storage layer does not always create a delta blindly.

It compares new/update volume to base-file size using `data_merge_ratio`.

That lets it choose between:

- keeping a small delta
- or rewriting the full data file immediately

This helps balance:

- write amplification
- scan cost
- storage fragmentation

## DataMerger Semantics

`DataMerger` merges sorted streams of ids.

The key rules are:

- source rows survive unless shadowed
- updater rows replace same-id source rows
- deleted ids suppress matching rows
- output ids remain sorted and unique

The merger can write:

- a new compact `.data`
- an updated `.delta`

It also preserves cosine inverse norms when the source layout requires them.

## Cosine-Specific Storage Rules

Cosine datasets carry extra persisted metadata.

### Persisted Files

For cosine datasets:

- data files are written with inverse norms
- delta files are written with inverse norms
- merge outputs preserve inverse norms

### Accumulator

The accumulator also stores one inverse norm per active vector.

### Why This Exists

The compute/scanner layer can then evaluate cosine distance using:

```text
1 - dot(a, q) * inv_norm(a) * inv_norm(q)
```

instead of recomputing the stored-vector norm in the hot scan loop.

This is a storage/performance tradeoff:

- a little more persisted metadata
- less per-candidate compute

## Interaction with Query Execution

The compute layer does not read raw files directly. It queries storage through:

- `DataReader`
- `Dataset`

From the compute layer’s perspective:

- `DataReader` is a visible persisted view
- `Dataset` is a logical collection that combines persisted state and
  accumulator state

The scanner relies on this storage model in two important ways.

1. Persisted readers expose base and delta streams in a form suitable for
   shadow-aware top-k scanning.
2. The dataset exposes accumulator iterators and modified-id lists so scanner
   can suppress stale persisted rows and include the freshest in-memory rows.

## Current Safety and Durability Model

The storage layer uses several mechanisms together:

- mmap-based validated reads
- per-range file locks
- a dataset owner lock
- temporary/merge files before rename
- accumulator WAL replay and truncation of torn tails

This is not a full transactional database, but it is designed to avoid the most
obvious failure modes:

- concurrent writers clobbering the same range
- losing buffered owner-mode updates after a crash
- exposing malformed persisted files to query code

## Practical Summary

The current storage design is:

- range-partitioned by vector id
- based on immutable base files plus smaller change files
- buffered by an in-memory accumulator
- protected by a WAL for buffered mutations
- compacted by explicit and heuristic merges
- exposed through validated memory-mapped readers
- coordinated by `Dataset`

That is the model the compute/scanner layer is built on top of today.
