# Design

Sketch2 is designed as a vector storage and compute engine, not as a general
purpose database. The project is opinionated about where it should compete:
layout of vector data on disk, how that data is moved through memory, and how
distance calculations run on modern CPUs. Metadata management, relational
queries, and application-specific workflows are expected to live in host
systems that integrate Sketch2.

## Scope Boundary

The main architectural choice is to keep Sketch2 focused on vector-heavy work.
That gives the project room to optimize for:

- compact persisted vector storage
- predictable scans over stored vectors
- efficient batched ingestion and merge flows
- runtime SIMD dispatch across supported CPUs
- integration points for other systems instead of a full database surface

This boundary is visible across the repository:

- the core storage layer owns data files, deltas, the in-memory accumulator,
  and crash recovery
- the compute layer owns metric kernels and top-k scanning
- the `parasol` API exposes a shared library interface
- `vlite` exposes read-only SQLite integration
- Python support exists as a thin wrapper used for demos, tests, and scripting

## Hardware-Aware Design

The project treats vector workloads as a storage and compute problem that
should be shaped around the actual hardware:

- persisted vectors are stored in a custom binary format instead of being
  packed into generic database pages
- data is laid out to keep scans sequential and cheap
- alignment is preserved so SIMD kernels can process vectors efficiently
- runtime backend selection chooses the best supported implementation on the
  current CPU without requiring separate binaries

The design goal is to reduce unnecessary translation layers between persisted
bytes, CPU caches, and the hot distance-calculation loop.

## Storage Model

Sketch2 uses a hybrid storage model. Most data lives in compact immutable base
files, while recent updates are staged separately and merged later.

### Base State

Persisted data is partitioned by id range. For a given vector id:

```text
file_id = id / range_size
dir = dirs[file_id % dirs.size()]
```

That mapping is stable and is used for bulk ingest, point lookup, merge
decisions, and cache reuse. The main persisted artifact is a `.data` file for
one id range. A data file stores:

- a header describing type, dimension, and layout
- aligned vector records
- optional cosine inverse norms for cosine datasets
- sorted active ids
- sorted deleted ids

This format is designed for direct memory-mapped reads and cheap sequential
scan access.

The directory mapping is also a scaling mechanism. A dataset can use multiple
data directories, which means different id ranges can be placed on different
disks or storage devices. That gives the system a straightforward way to spread
I/O across independent storage paths and improve parallel access during ingest,
merge, and query-heavy workloads.

### Mutable State

Sketch2 does not rewrite persisted base files on every update. Instead it uses
two mutable layers:

1. an in-memory accumulator for recent owner-mode updates and deletes
2. optional `.delta` files for persisted changes that have not yet been folded
   back into the base `.data` file

This model is based on the assumption that most datasets are dominated by
stable persisted state and that updates/deletes are smaller than the full data
volume. The benefit is lower write amplification and better read locality for
the main body of data.

### Crash Recovery

The accumulator is protected by an append-only write-ahead log,
`sketch2.accumulator.wal`. Acknowledged in-memory changes can therefore be
replayed after a crash instead of being lost. Torn trailing WAL records are
truncated during recovery, and valid records are applied in order.

This gives Sketch2 a mutable front layer without giving up durability for
owner-mode writes.

### Merge Strategy

Changes are reconciled by `DataMerger`. Depending on the state of a range,
Sketch2 can:

- create a new base `.data` file
- create or refresh a `.delta` file
- merge `.delta` content back into a compact `.data` file
- fold accumulator content into persisted state

The design deliberately accepts a visibility gap between write time and compact
persisted state. Queries still see a logically current view, but the physical
layout is repaired asynchronously through merge steps instead of one-write,
one-rewrite behavior.

## Read Path And Query Semantics

The read path is designed around a logically current view of a dataset even
when data exists in multiple physical layers.

- `DataReader` exposes memory-mapped persisted files
- `Dataset` coordinates range mapping, cached readers, owner/guest behavior,
  and accumulator state
- the scanner can search either a `DataReader` or a `Dataset`
- when newer data shadows older persisted rows, the scan logic skips the stale
  base rows instead of materializing a rewritten file first

The result is that queries operate on a current logical dataset while the
storage layer preserves efficient bulk-oriented write behavior.

## Parallel Execution

Sketch2 is designed to use available hardware parallelism when the work can be
split cleanly. The system does not assume that vector storage is a
single-threaded pipeline.

- storage work can be divided by id range
- different ranges can live in different directories and on different disks
- range-oriented tasks can therefore run concurrently without contending on one
  shared file
- the runtime can use multiple threads to improve CPU utilization where the
  operation benefits from parallel execution

This is especially useful for dataset loading, range-level maintenance, and
other operations that naturally decompose into independent units of work. The
design goal is not parallelism for its own sake, but better utilization of the
available CPU cores and storage bandwidth when independent work already exists.

## Concurrency And Ownership

Sketch2 explicitly distinguishes between writer ownership and read-only access.

- owner mode can mutate the dataset
- guest mode is read-only
- per-range lock files protect range-level write operations
- a dataset owner lock ensures that only one owner instance modifies a dataset
  at a time

This matters because the design assumes the storage engine can be embedded into
larger systems. The locking model is intended to keep read behavior simple and
write behavior explicit instead of relying on hidden multi-writer coordination.

## Writer And Reader Model

Sketch2 is designed so that data modification and query execution can run as
separate roles.

- a writer can run in one process and apply dataset updates
- a reader can run in another process and execute queries
- multiple readers can operate concurrently, including readers running on
  different threads

The system is built to let writes proceed without forcing query processing to
stop. Persisted base files, delta files, and the accumulator give the writer a
way to apply changes incrementally, while readers continue to work against a
consistent logical dataset view instead of requiring a full stop-the-world
rewrite.

Writer ownership is intentionally strict. Sketch2 enforces a single dataset
writer in the system through the dataset owner lock, so mutation semantics stay
clear and concurrent writers do not corrupt shared state. Readers use read-only
access patterns and can scale independently from that single writer.

There is also a communication path from writer to reader about applied updates.
The writer increments a file-backed update counter when storage-visible changes
are committed, and readers can check that counter to detect that their cached
view may be stale. This provides a lightweight cross-process notification
mechanism without coupling readers to the writer process directly.

## Synthetic Test Data

Sketch2 includes dedicated functionality for generating synthetic input data.
This is a design choice, not just a test convenience.

- the system can generate large volumes of vectors for development, testing,
  and demos
- generated patterns are deterministic, so the resulting dataset contents are
  predictable
- unit tests and integration tests actively use that predictable content to
  assert clear expected outcomes instead of relying on opaque random fixtures

This matters for a storage and compute engine because correctness needs to be
checked at scale. Deterministic synthetic data makes it practical to exercise
bulk ingest, multi-range layouts, reader/writer coordination, merges, deletes,
and KNN behavior with inputs that are both large and easy to reason about.

## Compute Architecture

Sketch2 keeps compute specialization explicit. Distance computation is not a
generic callback invoked inside the innermost loop. Instead, the scanner
dispatches once per query across three axes:

- backend
- distance function
- vector element type

After that, the search runs on one specialized path.

Supported distance functions today:

- `l1`
- `l2`
- `cos`

Supported vector element types today:

- `f32`
- `f16`
- `i16`

## Runtime SIMD Dispatch

One binary is expected to run on different machines. The project therefore
separates build-time availability of optimized kernels from runtime backend
selection.

Current runtime backends are:

- `scalar`
- `avx512_vnni`
- `avx512f`
- `avx2`
- `neon`

The selected backend is process-wide and detected once during runtime
initialization. On x86, the build keeps the baseline portable and compiles
higher-ISA code through per-function target attributes, which is why GCC or
Clang are required for the x86 runtime-dispatch model. On AArch64, the build
enables the architecture options needed for NEON and fp16 support.

This avoids the two common failure modes of vector libraries:

- shipping one binary that crashes or traps on older CPUs
- shipping multiple architecture-specific binaries and pushing selection to the
  operator

## Integration Strategy

Sketch2 is meant to be used from other environments, so integrations are a
design requirement rather than an afterthought.

### Parasol C API

The primary native integration surface is the `parasol` C API. It exposes
dataset creation, loading, mutation, and query operations through the shared
`libsketch2.so` runtime library.

This layer exists so that host applications can integrate Sketch2 without
binding directly to internal storage and compute classes. It provides a stable
entry point for embedding the engine into other systems while still sharing the
same process-wide runtime state, configuration, logging, and thread-pool
behavior as the rest of Sketch2.

### Python

Python is the lightweight scripting surface. It is useful for:

- data preparation and ingestion pipelines
- demos and local experimentation
- integration tests

The wrapper is intentionally thin and delegates core behavior to
`libsketch2.so`.

### SQLite

SQLite integration provides a familiar entry point for combining vector search
with relational data. The `vlite` module is read-only and binds a virtual table
to an existing Sketch2 dataset. This keeps responsibilities clean:

- Sketch2 owns vector storage and KNN execution
- SQLite owns SQL planning, joins, and relational filtering

That division makes SQLite a practical adoption path without forcing Sketch2 to
become a relational database.

## Platform Constraints

Sketch2 is Linux-only today. That is a deliberate scope decision rather than an
oversight. Supporting multiple operating systems would dilute effort in the
parts of the system that matter most right now: file layout, mmap-based access,
runtime dispatch, and correctness of the core data path.

At the same time, CPU portability matters. The project is intended to run well
on multiple CPU families because vector workloads benefit directly from
architecture-specific SIMD support.

## Current And Future Scope

The current system is centered on brute-force KNN over the stored
representation. That is intentional: the project first establishes a strong
base for storage layout, mutable-state handling, and runtime-specialized
compute.

Planned next steps build on that base:

- ANN indexing such as IVF and PQ
- more host-database integrations
- possible interoperability with other vector ecosystems where it makes sense

The main design principle is to get the fundamentals right first: compact
storage, clear layering, predictable mutation semantics, and a fast scan path
that respects the machine it runs on.
