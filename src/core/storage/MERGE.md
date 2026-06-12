# Merge Functionality in Sketch2 Storage

This document provides a detailed technical overview of how data merging works in the `sketch2` storage layer. Merging is a critical background process that ensures storage efficiency and maintains high read performance by compacting updates and deletes into optimized base files.

## Table of Contents
1. [Overview](#1-overview)
2. [High-Level Orchestration: DatasetWriter](#2-high-level-orchestration-datasetwriter)
    * [store_and_merge Strategy](#store_and_merge-strategy)
    * [Merge Heuristics](#merge-heuristics)
3. [Low-Level Execution: DataMerger](#3-low-level-execution-datamerger)
    * [Unified Update Streams](#unified-update-streams)
    * [The Merge Algorithm](#the-merge-algorithm)
4. [Key Classes and Structures](#4-key-classes-and-structures)
5. [Atomic Safety and Durability](#5-atomic-safety-and-durability)
6. [Performance Optimizations](#6-performance-optimizations)

---

## 1. Overview

In `sketch2`, data is organized into **ranges** of IDs, each corresponding to a pair of files:
- **Base File (`.data`)**: A compact, sorted collection of vector records.
- **Delta File (`.delta`)**: An "overlay" containing recent updates and tombstones (deletes) for that range.

**Merging** is the process of:
1. Combining a base file with a delta file to produce a new, compact base file.
2. Combining new incoming data with an existing base or delta file.
3. Reconciling deletes (tombstones) so that deleted records are physically removed from base files.

The goal is to prevent the "overlay cost" (reading multiple files for one range) from growing indefinitely while keeping write operations fast.

## 2. High-Level Orchestration: [DatasetWriter](dataset_writer.cpp)

`DatasetWriter` is the "brain" of the operation. it decides *when* and *what* to merge based on the incoming data volume and existing file states.

### `store_and_merge` Strategy
When new data arrives (via `store()` or `complete_writing()`), it is processed range-by-range in [store_and_merge](dataset_writer.cpp). The logic follows these branching paths:

1.  **No Base File Yet**: The new data becomes the first `.data` file.
2.  **Base Exists, No Delta**:
    *   If the new data is **large** (relative to the base): Merge it directly into the base file (creates a new `.data`).
    *   If the new data is **small**: Save it as the first `.delta` file.
3.  **Both Exist**:
    *   The new data is merged into the existing `.delta` file.
    *   **Compaction Check**: After updating the delta, if the delta's size now exceeds a threshold relative to the base, the delta is folded into the base file.

### Merge Heuristics
The decision to merge into a base file is governed by `metadata_.data_merge_ratio`.
- [check_data_file_merge](dataset_writer.cpp): Returns true if `base_count < update_count * ratio`.
- [check_data_delta_merge](dataset_writer.cpp): Returns true if `base_count < delta_count * ratio`.

A higher ratio means merges happen less frequently (allowing larger deltas), while a lower ratio keeps base files more up-to-date at the cost of more frequent merge IO.

## 3. Low-Level Execution: [DataMerger](data_merger.cpp)

`DataMerger` is the utility class that performs the actual bit-level merging. It is designed to be agnostic of whether the updates come from another file on disk or directly from memory.

### Unified Update Streams
To keep the core logic simple, `DataMerger` presents all update sources through lightweight sorted cursor adapters instead of materializing a separate merge array.
- **From File**: `DataReaderUpdaterCursor` and `DataReaderDeletedCursor` stream live rows and deleted ids directly from a `DataReader`.
- **From Memory**: `InputReaderUpdaterCursor` and `InputReaderDeletedCursor` stream live rows and delete-only rows directly from an `InputReaderView`.
- **Delta Tombstones**: `DeltaDeleteCursor` merges persisted tombstones with incoming deletes while dropping any source delete that is resurrected by a live updater row.

### The Merge Algorithm
The core of the merge is [merge_records](data_merger.cpp). It uses a **two-pointer walk** (similar to the "Merge" step in Merge Sort) across two sorted streams:
1.  **Source Stream**: The existing "Base" or "Delta" data.
2.  **Updater Stream**: The new records being injected.
3.  **Delete Set**: A sorted list of IDs that must be suppressed.

**Logic Flow:**
- If an ID is in the **Delete Set**, it is skipped (dropped from the output).
- If an ID exists in both Source and Updater, the **Updater** version wins (overwrites the old record).
- Records are streamed to the output file in sorted order, ensuring the result is immediately ready for binary search.

## 4. Key Classes and Structures

### Cursor Adapters
The cursor helpers in [data_merger.cpp](data_merger.cpp) give `merge_records` a uniform interface over different update sources:
- `DataReaderUpdaterCursor`: streams sorted persisted updater rows.
- `InputReaderUpdaterCursor`: streams sorted in-memory updater rows and parses text vectors only if the row survives to output.
- `DataReaderDeletedCursor` and `InputReaderDeletedCursor`: stream sorted delete ids.
- `DeltaDeleteCursor`: builds the merged delete stream used by delta-file merges.

### [MergeFile](data_merger.cpp)
An RAII wrapper for the destination file. It handles:
- Creating the file with a `.merge` extension.
- Writing the initial header.
- Finalizing with `fsync` and `fclose`.

### [MergeOutputWriter](data_merger.cpp)
A helper that manages the specific layout of the `sketch2` data format. It ensures that vectors are written first, followed by the optional cosine array, and finally the sorted ID array. It also handles on-demand parsing of text-based updates using a scratch buffer.

### Direct Input Path
When merging from `InputReaderView`, `DataMerger` no longer builds a temporary materialized updater structure. Instead, it borrows raw binary payloads or text slices directly from the input reader and lets the updater cursors feed surviving rows into `MergeOutputWriter`. That keeps memory usage lower while preserving the same sorted merge semantics.

## 5. Atomic Safety and Durability

Merging is designed to be crash-safe:
1.  **Isolated Output**: All merges write to a temporary `.merge` file.
2.  **Durable Flush**: The file is explicitly flushed to disk via `fsync()` before closing.
3.  **Atomic Rename**: Once the merge is complete and durable, `std::filesystem::rename` is used to replace the old file. This ensures that readers never see a partially written or corrupted file.
4.  **Directory Sync**: The parent directory is `fsync()`ed after publishing a renamed file. When compaction removes an old delta after replacing the data file, the directory is synced after the rename and again after the unlink so recovery cannot observe old data with the delta missing.
5.  **Error Cleanup**: If a merge fails at any point, the `.merge` file is deleted.

## 6. Performance Optimizations

- **Fine-Grained Locking**: By splitting the monolithic writer lock into session, file, and state mutexes, `sketch2` allows input staging and dataset merging to happen concurrently. A new write session can start as soon as the previous one has finished writing its temporary file, even if the actual merge into the base data is still in progress.
- **Parallel Processing**: `DatasetWriter` uses a thread pool to merge different ID ranges in parallel, significantly speeding up large ingestion tasks.
- **Direct Input Merging**: If new data is large enough to trigger a base merge, it is merged *directly* from the parsed input memory. This skips the redundant step of writing a temporary file to disk just to read it back for the merge.
- **Zero-Copy Reads**: `DataReader` uses memory-mapping (`mmap`), allowing `DataMerger` to access source records directly from the OS page cache without extra copies.
- **Buffered Writing**: `MergeFile` uses a large internal buffer (`setvbuf`) to minimize the number of `write` system calls.

---
*This document was generated by Gemini to assist in understanding the Sketch2 storage architecture.*
