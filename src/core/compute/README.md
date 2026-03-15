# compute

This directory contains the vector-distance and top-k scan layer used by the
rest of the system. It is responsible for:

- distance kernels for `L1`, `L2`, and cosine distance
- scalar and SIMD implementations of those kernels
- runtime selection of the best available compute backend
- top-k scanning over `DataReader` and `Dataset`
- keeping the hot scan loop specialized while still shipping one portable binary

The current design is centered around a process-wide `ComputeUnit` selected at
runtime and then reused by all queries.

## Why This Design Exists

The earlier model selected the compute backend at compile time. That approach
was simple, but it had two major drawbacks:

1. A binary built with AVX2 globally enabled could not safely run on older CPUs.
2. Adding faster backends meant either shipping separate binaries or raising the
   minimum CPU requirement for everyone.

The new design fixes that by separating:

- build-time availability of backend-specific code
- runtime selection of which backend is active on the current machine

This gives us one binary that can:

- run on a scalar-only host
- use AVX-512F on supported x86 CPUs
- use AVX2 on supported x86 CPUs
- use NEON on AArch64
- keep the hot path specialized instead of routing every vector pair through a
  generic virtual or indirect-call layer

## High-Level Architecture

The compute subsystem now has four layers.

### 1. Backend Detection and Runtime State

The process-wide backend is represented by `ComputeUnit` in
`src/core/utils/compute_unit.{h,cpp}`.

Today it supports:

- `scalar`
- `avx512_vnni`
- `avx512f`
- `avx2`
- `neon`

The tree also now contains x86 AVX-512 build scaffolding:

- `SKETCH_ENABLE_AVX512F`
- `SKETCH_ENABLE_AVX512VNNI`
- `compute_avx512_utils.h`

`avx512f` is a selectable runtime backend for `f32`, `f16`, and full-range-
correct `i16` kernels.
`avx512_vnni` is a selectable runtime backend that reuses the same AVX-512
kernels behind a distinct runtime/backend name on CPUs that advertise VNNI.

`ComputeUnit::detect_best()` selects the best supported backend for the current
process, optionally honoring `SKETCH2_COMPUTE_BACKEND`.

The selected backend is stored in the process-wide `Singleton` in
`src/core/utils/singleton.{h,cpp}`.

Important properties:

- selection happens once for normal execution
- the active backend is process-wide
- the backend can be overridden in tests with
  `Singleton::force_compute_unit_for_testing(...)`
- the override is intentionally not a production control surface for live
  per-query backend switching

### 2. Metric Families

Each metric family exposes:

- scalar fallback kernels
- runtime resolvers
- typed entrypoints used by scanner template dispatch

The main headers are:

- `compute_l1.h`
- `compute_l2.h`
- `compute_cos.h`

Each one follows the same pattern:

- `dist(...)` is the `ICompute` interface entrypoint
- `resolve_*()` chooses the implementation for the active `ComputeUnit`
- typed functions such as `dist_f32`, `dist_f16`, `dist_i16` are the concrete
  scalar entrypoints

Architecture-specific kernels live in:

- `compute_l1_avx2.h`, `compute_l2_avx2.h`, `compute_cos_avx2.h`
- `compute_l1_avx512.h`, `compute_l2_avx512.h`, `compute_cos_avx512.h`
- `compute_l1_neon.h`, `compute_l2_neon.h`, `compute_cos_neon.h`

### 3. Scanner-Oriented Dispatch

`scanner.cpp` is where query-time dispatch actually matters.

The scanner does not call a generic “distance” callback for every vector pair.
Instead, it performs dispatch once per query on three axes:

- backend
- metric
- data type

After that, the scan stays on one fully specialized path.

This is the key design choice that makes runtime backend selection cheap enough
for the hot path.

### 4. Data Source Traversal

The scanner can run against two logical sources:

- `DataReader`
- `Dataset`

`Dataset` adds another concern: newer state may exist in the accumulator, so
persisted readers must skip ids shadowed by pending updates/deletes. The scan
helpers in `scanner.cpp` handle that once and reuse the same score objects
across metrics.

## Runtime Backend Selection

### Selection Rules

On process startup, `ComputeUnit::detect_best()` follows this logic:

1. Read `SKETCH2_COMPUTE_BACKEND`
2. If it contains a known backend name and that backend is supported by both:
   - this build
   - this CPU
   then use it
3. Otherwise, auto-detect the best supported backend
4. Fall back to `scalar` if nothing else is valid

Current environment values are:

- `auto`
- `scalar`
- `avx512_vnni`
- `avx512f`
- `avx2`
- `neon`

If the override is invalid or unsupported, the code quietly falls back to
auto-detection rather than failing process startup.

### Why Selection Lives in `Singleton`

The selected compute backend is process-wide state for the same reason log
configuration and thread-pool configuration are process-wide:

- it should not drift between queries in normal execution
- it should be decided once, not rediscovered in every hot function
- test code still needs a controlled way to force a backend

This is why the `Singleton` stores `ComputeUnit` directly instead of asking
every metric or every scan loop to probe the CPU again.

## Build Model

The build intentionally does not compile the whole project with a high x86 ISA
enabled globally.

### x86

On x86, the baseline build only assumes `-msse2` at the project level. Higher
ISAs are made available through build options and per-function target
attributes, not through global project-wide compiler flags.

Currently:

- `SKETCH_ENABLE_AVX2`
- `SKETCH_ENABLE_AVX512F`
- `SKETCH_ENABLE_AVX512VNNI`
- GCC/Clang per-function target attributes in `compute_avx2_utils.h`
- GCC/Clang per-function target attributes in `compute_avx512_utils.h`

Today AVX2, AVX-512F, and AVX-512 VNNI are all wired through runtime dispatch.

That lets scalar, AVX2, AVX-512F, and AVX-512 VNNI code live in the same
binary.

This is the critical difference from the old model. We are no longer building
the whole compute library as “a high-ISA x86 binary”. We are building:

- baseline code that runs everywhere
- AVX2 code that exists in the binary but is only entered when the CPU supports
  it
- AVX-512F code that exists in the binary but is only entered when the CPU
  supports it
- AVX-512 VNNI code that exists in the binary but is only entered when the CPU
  supports it

Because this design relies on per-function `target(...)` attributes, the x86
runtime-dispatch model is currently supported only with GCC or Clang.

### AArch64

On AArch64, the build enables the architecture options needed for NEON/fp16 and
the runtime backend is simply `neon`.

## Dispatch Boundaries

The current design is intentionally layered so that dispatch happens at the
right cost level.

### Cheap Enough for Setup-Time Only

The following functions read the process-wide backend from the singleton and
should be treated as setup-time dispatch:

- `ComputeL1::resolve_dist(...)`
- `ComputeL2::resolve_dist(...)`
- `ComputeCos::resolve_dist(...)`
- `ComputeCos::resolve_dot(...)`
- `ComputeCos::resolve_squared_norm(...)`
- `ComputeCos::resolve_dist_with_query_norm(...)`

These functions are cheap enough to call once when binding a path, but they are
not intended to sit inside the innermost per-vector loop.

### Hot Path Dispatch

The scanner performs the real query dispatch at the outer entrypoints:

- `Scanner::find_(const Dataset&, ...)`
- `Scanner::find_items_(const Dataset&, ...)`
- `Scanner::find_(const DataReader&, ...)`
- `Scanner::find_items_(const DataReader&, ...)`

Each of these:

1. validates arguments
2. reads the active `ComputeUnit`
3. chooses one backend family
4. chooses one metric family
5. chooses one typed scorer
6. runs the full scan using that specialized scorer

This means the inner scan loop does not branch on backend or data type.

## Scanner Design in More Detail

`scanner.cpp` is structured around a small number of reusable concepts.

### Top-k Heap

The scanner maintains a bounded heap of `DistItem`. `push_result(...)` keeps
only the best `count` results seen so far, and `extract_ids(...)` /
`extract_items(...)` convert the heap into final sorted output.

### Score Objects

The actual “distance computation” used by a scan is wrapped in small callable
objects:

- `DistanceScore`
- `QueryNormScore`
- `InvNormScore`

These objects capture:

- query pointer
- dimension
- optional precomputed query normalization state

They are instantiated with concrete function pointers at compile time, so they
remain fully specialized.

### Reader and Dataset Scan Helpers

The scan helpers separate traversal from scoring:

- `scan_iterator_scored(...)`
- `scan_ordered_reader_scored(...)`
- `scan_data_reader_scored(...)`
- `scan_dataset_heap_custom(...)`
- `scan_reader_heap_custom(...)`

This structure lets `L1`, `L2`, and cosine all reuse the same traversal logic.

### Dataset Shadowing Semantics

Datasets combine persisted data and pending accumulator state.

The rules are:

- persisted readers are scanned first
- ids modified in the accumulator are skipped from persisted readers
- accumulator entries are then scanned as the freshest view

As a result, each logical id contributes at most one candidate to the heap, and
the newest version wins.

## Cosine-Specific Design

Cosine distance needs more machinery than L1/L2, so it has some extra structure.

### Public Semantics

Cosine distance follows these rules:

- both zero vectors -> distance `0.0`
- one zero vector -> distance `1.0`
- otherwise -> `1 - clamp(cosine, -1, 1)`

These semantics are centralized in:

- `finalize_cosine_distance(...)`
- `finalize_cosine_distance_from_inverse_norms(...)`

This ensures scalar and SIMD paths share the same edge-case behavior.

### Two Cosine Paths

The scanner supports two cosine evaluation modes for stored vectors:

1. Full query-norm path
   - compute query squared norm once
   - candidate path computes candidate norm and dot product

2. Stored inverse-norm path
   - if a reader stores inverse norms, reuse them directly
   - only the dot product is computed per candidate

This is why cosine dispatch uses:

- `resolve_dot(...)`
- `resolve_squared_norm(...)`
- `resolve_dist_with_query_norm(...)`

instead of a single simple `dist(...)` interface.

### Query Norm Computation

The scanner computes the query norm once per query, not once per candidate.

`query_squared_norm<ComputeTarget>(...)` keeps that norm computation on the same
backend family as the rest of the query. That avoids mixing:

- a scalar query setup phase
- with a SIMD candidate loop

and keeps the semantics and performance model consistent.

## Scalar and SIMD Correctness Model

The scalar implementations are the semantic reference path.

Important points:

- `f16` values are widened before arithmetic
- scalar accumulation is done in `double`
- SIMD kernels follow the same public semantics even if the lane width and
  reduction order differ
- cosine zero-vector handling is centralized and shared

This means backend changes are expected to preserve:

- ranking semantics
- edge-case behavior
- supported data types

while allowing normal floating-point drift from different reduction orders.

## Why We Did Not Use a Generic Registration System

A natural alternative would be a dynamic registration table of backends and
per-metric function pointers. That would reduce the number of switch sites, but
it would also push the design toward more indirect calls in the hot path.

The current design deliberately prefers:

- explicit backend enums
- explicit switch-based selection
- templated scan paths

because performance-sensitive code benefits from keeping specialization visible
to the compiler.

This means adding a backend still requires touching multiple places, but the
tradeoff is acceptable at the current scale.

## Testing Strategy

The design is validated at several levels.

### Direct Kernel Tests

Metric-specific unit tests exercise scalar and SIMD implementations directly:

- `utest_compute_l1.cpp`
- `utest_compute_l2.cpp`
- `utest_compute_cos.cpp`

These tests cover:

- correctness against reference implementations
- zero-dimension behavior
- aligned and unaligned input
- extreme-value cases
- resolver selection

### Runtime Dispatch Tests

`utest_compute_runtime.cpp` verifies that forcing a backend causes the resolver
layer to choose the expected concrete functions.

This is especially important now that backend selection is runtime state rather
than a preprocessor-only choice.

### Scanner-Level Tests

`utest_scanner.cpp` verifies end-to-end search behavior across:

- readers
- datasets
- delta state
- accumulator state
- cosine stored-norm and computed-norm paths

## How To Add a New Backend

The current design was created specifically to make future backends practical,
even if not fully plug-and-play.

For example, extending the AVX-512 line further would typically require:

1. Add a new `ComputeBackendKind` entry in `compute_unit.h`
2. Extend parsing, support checks, and auto-detection in `compute_unit.cpp`
3. Reuse or extend the existing AVX-512 build support and ISA-targeting helpers
4. Implement new metric kernels
5. Extend resolver switches in:
   - `compute_l1.h`
   - `compute_l2.h`
   - `compute_cos.h`
6. Extend scanner backend switches in `scanner.cpp`
7. Add runtime-dispatch and direct-kernel tests

This is more mechanical than architectural. The key redesign work is already
done: the backend now exists as a runtime concept rather than as a global
compile-time choice.

## Practical Guidance for Future Work

When extending this subsystem, keep the following rules in mind.

### 1. Do Not Reintroduce Global AVX2/AVX-512 Compile Flags

If the entire project is compiled with a high ISA globally, the portable-binary
property is lost immediately.

### 2. Keep Backend Dispatch Outside the Inner Loop

The scanner should continue to select a backend once per query and then stay on
that path.

### 3. Preserve Public Metric Semantics

Optimized kernels may change instruction selection and reduction order, but they
must preserve:

- supported types
- return-value contract
- zero-vector handling
- accumulator/persisted shadowing semantics at scanner level

### 4. Treat `resolve_*()` as Binding Helpers, Not Per-Element Helpers

They are fine for setup-time dispatch and for non-hot interface paths, but they
are not intended as the inner-loop abstraction boundary.

## File Map

- `compute.h`
  Common compute interfaces and shared distance item types.
- `compute_l1.h`, `compute_l2.h`, `compute_cos.h`
  Scalar implementations and runtime resolver layer.
- `compute_*_avx2.h`, `compute_*_neon.h`
  SIMD-specialized kernels.
- `compute_*_avx512.h`
  AVX-512-specialized kernels: `avx512f` for `f32`/`f16`, plus `avx512_vnni`
  i16 entrypoints that keep exact 64-bit accumulation.
- `compute_avx2_utils.h`
  AVX2 target attributes and small SIMD helpers shared by AVX2 kernels.
- `compute_avx512_utils.h`
  AVX-512 target attributes and small SIMD helpers shared by AVX-512 kernels.
- `scanner.h`, `scanner.cpp`
  Query-time dispatch and top-k scanning.
- `utest_compute_*.cpp`
  Metric correctness and dispatch tests.
- `utest_compute_runtime.cpp`
  Runtime backend selection tests.
- `utest_scanner.cpp`
  End-to-end scanner tests.

## Summary

The new compute design is a runtime-selected, process-wide backend model that:

- keeps one binary portable across different CPUs
- preserves specialized hot loops
- isolates backend choice to query setup time
- keeps scalar behavior as the semantic baseline
- already supports AVX-512F for `f32`/`f16`
- already supports AVX-512F for full-range-correct `i16`
- still exposes an `avx512_vnni` runtime backend for VNNI-capable CPUs
- keeps a clean path for future backends and future AVX-512 variants

That is the core architectural shift in this directory.
