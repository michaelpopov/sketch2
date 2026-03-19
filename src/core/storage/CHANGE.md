# Dataset → DatasetReader / DatasetWriter Refactor

This document describes the staged refactor that replaced the monolithic `Dataset`
class with a three-class inheritance hierarchy: `Dataset` (base), `DatasetReader`
(read infrastructure), and `DatasetWriter` (write + accumulator infrastructure).

---

## Motivation

The original `Dataset` class mixed two unrelated concerns:

- **Reading** — file cache, update detection, `reader()` / `get()` / `get_vector()`
- **Writing** — accumulator, WAL, owner lock, `store()` / `add_vector()` / merge

It also used a runtime `DatasetMode` flag (`Owner` / `Guest`) to control whether
mutations were allowed.  Callers were responsible for calling `set_guest_mode()`
on instances that should be read-only, and the check was enforced at runtime only.

The refactor separates concerns at the **type level**:

| Class | Role | Consumers |
|-------|------|-----------|
| `Dataset` | Metadata only (`init`, accessors) | base class |
| `DatasetReader` | Read path (cache, notifier, `reader()`, `get()`) | vlite, scanner |
| `DatasetWriter` | Write path (accumulator, WAL, owner lock, `store()`) | parasol |

A `DatasetWriter` IS-A `DatasetReader` IS-A `Dataset`.  Read-only consumers
receive `const DatasetReader&` or hold a `DatasetReader*`, so the compiler
prevents them from calling write operations.

---

## Stage 1 — Rename DatasetReader → DatasetRangeReader

**Motivation**: The name `DatasetReader` was already taken by a file-range
iterator.  Freeing that name required a mechanical rename before the new class
could be introduced.

**Changes**:
- `dataset.h` / `dataset.cpp`: `DatasetReader` → `DatasetRangeReader` everywhere
  (class declaration, `DatasetReaderPtr` typedef → `DatasetRangeReaderPtr`,
  `friend` declaration, `Dataset::reader()` return type and body)

No behavioral change; all tests pass.

---

## Stage 2 — Prepare Dataset base + create empty subclass headers

**Motivation**: Before any code can use `DatasetReader` or `DatasetWriter` as
distinct types, the class hierarchy needs to exist, and `Dataset` needs to be
a proper base class.

**Changes**:

`dataset.h`:
- `virtual ~Dataset()` — makes the destructor virtual so `DatasetWriter`'s
  destructor (which flushes the accumulator) runs correctly when the object is
  held through a base-class pointer
- `metadata_` moved from `private` to `protected` — required by `DatasetReader`
  methods that inspect metadata (e.g., to check `dist_func` in `open_reader_`)
- `item_path_base()` moved from `private` to `protected` — needed by
  `DatasetWriter`'s write methods (e.g., `store_and_merge`)

New files:

`dataset_reader.h` — empty subclass at this point:
```cpp
class DatasetReader : public Dataset {
public:
    using Dataset::Dataset;
    ~DatasetReader() override = default;
};
```

`dataset_writer.h` — empty subclass:
```cpp
class DatasetWriter : public DatasetReader {
public:
    using DatasetReader::DatasetReader;
    ~DatasetWriter() override = default;
};
```

No `.cpp` files yet.  No behavioral change.

---

## Stage 3 — Update all consumers to use new types

**Motivation**: Wire up the new types before moving any implementation.  Because
the subclasses are empty wrappers at this stage, all methods are still inherited
from `Dataset`; the type names are the only change.

### Scanner (`scanner.h` / `scanner.cpp`)

- Forward declaration `class Dataset` → `class DatasetReader`
- All `const Dataset&` parameters → `const DatasetReader&`
- `#include "dataset.h"` → `#include "dataset_reader.h"`
- All internal templates (`scan_dataset_heap_custom`,
  `build_dataset_heap_with_score`, `dispatch_dataset`, `dispatch_dataset_cos`,
  `dispatch_with_backend` Dataset overload) changed to take `const DatasetReader&`

### Parasol (`parasol.cpp`)

- `#include "dataset.h"` → `#include "dataset_reader.h"` + `#include "dataset_writer.h"`
- `sk_handle::ds` type: `Dataset*` → `DatasetWriter*`
- New data member `DatasetReader* ds_reader` added to `sk_handle`; set to `handle->ds`
  at open time (upcast of the same object)
- `sk_open_`: `make_unique<Dataset>()` → `make_unique<DatasetWriter>(); ds_reader = ds`
- Read operations (`sk_knn_`, `sk_gid_`, `sk_print_`) switched to use `ds_reader`
  so they express read-only intent; write operations (`sk_upsert_`, `sk_del_`,
  `sk_macc_`, `sk_mdelta_`) continue to use `ds`

### Vlite (`vlite.cpp`)

- `#include "dataset.h"` → `#include "dataset_reader.h"`
- `unique_ptr<Dataset>` → `unique_ptr<DatasetReader>`
- `make_unique<Dataset>()` → `make_unique<DatasetReader>()`
- `set_guest_mode()` call **removed** — vlite now simply instantiates
  `DatasetReader`, whose design will enforce read-only access at the type level

### Guest mode removal

The `DatasetMode` enum (`Owner` / `Guest`), the `mode_` private field, and the
`set_guest_mode()` method were **removed entirely** from `Dataset`.  Their
functions are replaced by the type hierarchy:

- Read-only access: use `DatasetReader` (will never acquire the owner lock or
  touch the accumulator)
- Read-write access: use `DatasetWriter` (acquires owner lock on first write)

Runtime guest-mode tests in `utest_dataset.cpp` were removed or reworked to use
`DatasetWriter` + `DatasetReader` pairs directly.

### Tests and benchmark

All test and benchmark files updated:
- `Dataset` → `DatasetWriter` wherever both read and write operations are used
- Guest-mode tests reworked so `DatasetWriter owner` / `DatasetReader guest`
  replace `Dataset owner` / `Dataset guest + set_guest_mode()`

---

## Stage 4 — Move read members and methods to DatasetReader

**Motivation**: The big structural split.  Everything that belongs to the read
path leaves `Dataset` and moves into `DatasetReader`, which gets its own
`dataset_reader.cpp`.

### New free functions in `dataset.h`

Two functions are used by both `dataset_reader.cpp` and `dataset_writer.cpp` and
are declared as free functions in `dataset.h`:

- `collect_dataset_items(const DatasetMetadata&, std::vector<DatasetItem>*)` —
  scans the dataset directories and collects `{data, delta}` file pairs grouped
  by numeric file id.  Previously in `dataset.cpp`'s anonymous namespace; now
  defined in `dataset_reader.cpp` (its primary consumer) and visible to
  `dataset_writer.cpp` (needed by `merge_()`).
- `dataset_owner_lock_path(const DatasetMetadata&)` — returns the path of the
  lock file used as both the filesystem exclusive lock and the UpdateNotifier
  counter file.  Also defined in `dataset_reader.cpp`.

### What moved to DatasetReader

**New protected members** (accessible to `DatasetWriter` via friendship):

| Member | Type | Purpose |
|--------|------|---------|
| `cache_lock_` | `mutable sketch::RWLock` | Serialises mutations to the three cache fields below |
| `items_cache_valid_` | `mutable bool` | Whether `items_cache_` reflects on-disk state |
| `items_cache_` | `mutable vector<DatasetItem>` | Cached result of the last directory scan |
| `reader_cache_` | `mutable unordered_map<uint64_t, DataReaderPtr>` | Per-range-id cached `DataReader` objects |

**New private member**:

| Member | Type | Purpose |
|--------|------|---------|
| `update_notifier_` | `mutable unique_ptr<UpdateNotifier>` | Checker-mode notifier; detects when a writer has changed the on-disk files |

`DatasetReader::ensure_update_notifier_()` always initialises in **checker** mode.
The writer's update notifier (updater mode) is a separate instance owned by
`DatasetWriter`.

**Public methods moved** from Dataset: `reader()`, `get()`, `get_vector()`

**Protected method moved**: `invalidate_data_caches_()`

**Private methods moved**: `ensure_update_notifier_()`, `ensure_items_cache_()`,
`find_item_()`, `open_reader_()`, `get_cached_reader_()`

**Friend declarations on DatasetReader**:
- `friend class DatasetWriter` — `DatasetWriter`'s write operations need access
  to `invalidate_data_caches_()` (protected) and `ensure_update_notifier_()`
  (private) to keep caches fresh and notifications correct
- `friend class DatasetRangeReader` — `DatasetRangeReader::next()` and `get()`
  call `get_cached_reader_()` (private) and access `metadata_` (protected)

**DatasetRangeReader** pointer type: `const Dataset*` → `const DatasetReader*`

**DatasetRangeReader, DataReaderPtr, DatasetRangeReaderPtr** moved to
`dataset_reader.h` (previously in `dataset.h`).

**AccumulatorIterator type alias** in DatasetReader (`using AccumulatorIterator =
Dataset::AccumulatorIterator`) removed later when the real implementation moved
to `DatasetWriter`.

### DatasetWriter additions in Stage 4

`DatasetWriter` gains its own `init()` overrides.  Each calls `Dataset::init()`
then `init_writer_()`:

`init_writer_()` — private method on Dataset at this stage:
1. Tries to acquire the owner lock **non-blocking** (`FileLockGuard::try_lock`)
2. If successful (no other owner): calls `init_accumulator_()` to replay any
   pending WAL, then **releases the lock immediately**
3. If the lock is held by another process (active writer): returns without
   acquiring the lock, allowing the caller to proceed as a plain reader of
   persisted files

This design satisfies two competing requirements:
- **Crash recovery**: after the previous owner crashed, the lock is released by
  the OS; `init_writer_()` acquires it and replays the WAL, making unflushed
  vectors visible via `get_vector()` before any write is called
- **Concurrent access**: readers opening a `DatasetWriter` while a writer is
  active do not block; they simply skip WAL replay and query persisted files
  (stale reads are acceptable per the concurrency contract in `DESIGN.md`)

`FileLockGuard::try_lock()` was added to support the non-blocking attempt.

**Owner lock and counter initialisation in `ensure_owner_lock_()`**: when a
`DatasetWriter` first calls a write operation and acquires `owner_lock_`, it also
ensures the lock file contains a valid 8-byte counter value.  This prevents
`DatasetReader::update_notifier_` (checker mode) from hitting the conservative
"short read → treat as updated" path on every subsequent `check_updated()` call,
which would otherwise cause the reader cache to be invalidated unnecessarily on
every `get()` call.

**Write method overrides**: `store()`, `store_accumulator()`, and `merge()` are
overridden in `DatasetWriter` to call `Dataset`'s implementation, then
`invalidate_data_caches_()` (so the same instance sees fresh data on the next
read), then `notify_update_()` (to bump the cross-process counter so other
`DatasetReader` instances detect the change).

**`get_vector()` override**: `DatasetWriter::get_vector()` checks the accumulator
first (pending writes visible without flushing), then falls back to the persisted
read path from `DatasetReader::get_vector()`.

### What Dataset retains after Stage 4

`Dataset` keeps only:
- `init()` overloads + `init_()` private
- Metadata accessors (`type()`, `dim()`, etc.)
- `metadata_` protected field
- `item_path_base()` protected method
- `AccumulatorIterator` nested class (still referenced by accumulator methods
  not yet moved; removed in Stage 5)
- Write and accumulator members/methods (moved in Stage 5)

---

## Stage 5 — Move write members and methods to DatasetWriter

**Motivation**: Complete the split.  Everything that belongs to the write path
leaves `Dataset` and moves to `DatasetWriter`, which gets its own
`dataset_writer.cpp`.

### What moved to DatasetWriter

**New private members**:

| Member | Type | Purpose |
|--------|------|---------|
| `write_mutex_` | `std::mutex` | Serialises concurrent mutation calls on the same instance |
| `owner_lock_` | `mutable unique_ptr<FileLockGuard>` | Exclusive filesystem lock; acquired lazily on first write |
| `accumulator_` | `mutable unique_ptr<Accumulator>` | In-memory write buffer with attached WAL |
| `update_notifier_` | `mutable unique_ptr<UpdateNotifier>` | Updater-mode notifier; increments the cross-process counter after each write |

**Destructor**: `~DatasetWriter()` contains the accumulator flush logic (previously
in `~Dataset()`).  `~Dataset()` is now `= default`.

**Public methods moved** from Dataset: `store()`, `store_accumulator()`, `merge()`,
`add_vector()`, `delete_vector()`

**Private methods moved** from Dataset: `store_()`, `store_accumulator_()`,
`merge_()`, `store_and_merge()`, `store_and_merge_accumulator()`,
`write_accumulator_range_()`, `init_accumulator_()`, `check_data_file_merge()`,
`check_data_delta_merge()`, `merge_data_file()`, `merge_delta_file()`,
`ensure_owner_lock_()`

**`init_writer_()` moved** from Dataset (was `protected`) to DatasetWriter
(`private`).

**`get_vector_from_accumulator_()`** moved from Dataset (`protected`) to
DatasetWriter (`private`).

### AccumulatorIterator moves to DatasetWriter

`Dataset::AccumulatorIterator` is replaced by `DatasetWriter::AccumulatorIterator`.
The key difference: `DatasetWriter::accumulator_begin()` returns a **real iterator**
backed by `accumulator_->begin()`, whereas the old `Dataset::accumulator_begin()`
always returned an empty iterator.

The iterator is used in parasol's `sk_gid_()` (which finds a vector's id by
scanning the accumulator for a byte-identical match) and was updated from
`handle->ds_reader->accumulator_begin()` to `handle->ds->accumulator_begin()`.

### Accumulator is purely write-side

The accumulator is no longer part of the read or query path.  Per-query
accumulator scanning was removed in Stage 5:

- `Dataset` loses `accumulator_modified_ids()`, `has_accumulator()`,
  `accumulator_has_cosine_inv_norms()`, `accumulator_begin()` — all accumulator
  read methods move to `DatasetWriter` (where they remain available for
  administrative use) or are removed from the scanner
- `DatasetReader` has no accumulator-related methods

### Scanner simplification

With the accumulator no longer consulted during queries, the scanner is
significantly simplified:

**Removed**:
- `scan_ordered_reader_scored()` — scanned persisted files while skipping ids
  present in the accumulator (the skip list is no longer needed)
- The `AccumScoreFn` template parameter and `accum_score` argument on
  `scan_dataset_heap_custom`
- The `require_accumulator_cosine_inv_norms` parameter and related assert
- `const std::vector<uint64_t> modified_ids = dataset.accumulator_modified_ids()`
  computation and all skip-list passing to lambdas
- Both `scan_iterator_scored(dataset.accumulator_begin(), ...)` calls (sequential
  and parallel paths)
- `scan_reader_heap_custom()` intermediate wrapper

**Simplified**:
- `scan_data_reader_scored()` now takes no skip-list; calls `scan_iterator_scored`
  directly on `reader.base_begin()` and `reader.delta_begin()`
- `scan_dataset_heap_custom()` template has one fewer parameter; lambdas passed
  to it take `(const DataReader&, size_t, DistHeap*)` instead of additionally
  carrying a skip-ids vector
- `build_reader_heap_with_score()` and `build_reader_heap_with_cos_scores()` are
  now simple one-liners calling `scan_data_reader_scored` directly

### What Dataset retains after Stage 5

After the full refactor, `Dataset` is a lean base that contains only:

- `DatasetMetadata` and `DatasetItem` struct definitions
- `collect_dataset_items()` and `dataset_owner_lock_path()` free function
  declarations (defined in `dataset_reader.cpp`)
- `Dataset` class: `init()` overloads, metadata accessors, `metadata_` protected
  field, `item_path_base()` protected method

`dataset.cpp` contains only the `init()` implementations and `item_path_base()`.

---

## Two UpdateNotifier instances

A subtlety of the final design is that a `DatasetWriter` instance contains
**two** `UpdateNotifier` objects operating on the same lock file:

1. **`DatasetReader::update_notifier_`** (inherited, checker mode) — initialised
   lazily on the first read operation.  Reads the 8-byte counter from the lock
   file; if it has changed since the last check, `ensure_items_cache_()` clears
   the file cache and rescans the directories.

2. **`DatasetWriter::update_notifier_`** (updater mode) — initialised lazily on
   the first write operation.  Increments the counter in the lock file after each
   successful `store()`, `store_accumulator()`, or `merge()`.

When a `DatasetWriter` both writes and reads (the typical parasol usage), its own
write bumps the counter, which its own checker detects on the next read and
triggers a cache refresh.  This is intentional: caches are always fresh after a
write on the same instance.

---

## File layout after the refactor

| File | Contents |
|------|----------|
| `dataset.h` | `DatasetMetadata`, `DatasetItem`, free function declarations, `Dataset` base |
| `dataset.cpp` | `Dataset::init()` overloads, `item_path_base()` |
| `dataset_reader.h` | `DataReaderPtr`, `DatasetRangeReader`, `DatasetRangeReaderPtr`, `DatasetReader` |
| `dataset_reader.cpp` | `collect_dataset_items`, `dataset_owner_lock_path`, `DatasetReader` read methods, `DatasetRangeReader` |
| `dataset_writer.h` | `DatasetWriter` with `AccumulatorIterator`, write methods, accumulator members |
| `dataset_writer.cpp` | All `DatasetWriter` method implementations |
