# Compute Benchmarks

This document explains the benchmark executables in `src/core/compute`, what
they measure, how to run them, and how to interpret the results.

The compute subsystem now has two benchmark layers:

- `bench_comp`: a lightweight throughput smoke benchmark for the public compute
  APIs
- `gbench_comp`: a Google Benchmark suite for compute kernels and scanner query
  paths

These benchmarks are complementary.

- `bench_comp` is simple and convenient when you want a fast sanity check.
- `gbench_comp` is the more complete benchmark suite and should be preferred
  for serious performance work.

## Quick Start

If you just want to run benchmarks from the top-level Makefile:

```bash
make bench
make benchext
make benchcomp
```

What those do:

- `make bench`: configure `build/` as a release benchmark build and run the
  essential Google Benchmark suite
- `make benchext`: configure `build/` as a release benchmark build and run the
  extended Google Benchmark suite
- `make benchcomp`: configure `build/` as a release benchmark build and run the
  lightweight compute benchmark

The intended split is:

- `make bench` for normal interactive use and quick regression checks
- `make benchext` for broader, slower runs that may take much longer

These targets reuse the main `build/` tree. That means running a benchmark
Makefile target may reconfigure that directory with
`SKETCH_ENABLE_BENCHMARKS=ON`.

## Benchmark Enablement

Google Benchmark support is optional and controlled by `SKETCH_ENABLE_BENCHMARKS`
(default: `OFF`). A plain CMake configure does not create `gbench_comp` unless
benchmark support is explicitly enabled. Google Benchmark is always fetched and
built locally from source.

If you are configuring CMake directly, use:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSKETCH_ENABLE_BENCHMARKS=ON
cmake --build build --target gbench_comp bench_comp
```

If you use the top-level Makefile, the benchmark targets automatically
reconfigure `build/` with benchmark support enabled before building.

`gbench_comp` also supports benchmark profiles through
`SKETCH2_GBENCH_PROFILE`:

- `essential`: smaller default matrix intended to finish in a few minutes
- `extended`: broader matrix intended for longer or overnight runs

The Makefile sets this automatically:

- `make bench`: `SKETCH2_GBENCH_PROFILE=essential`
- `make benchext`: `SKETCH2_GBENCH_PROFILE=extended`

To restrict scanner benchmarks to a specific mode, set
`SKETCH2_GBENCH_SCANNER_MODE`:

- `reader`: only reader-backed scanner cases
- `dataset_persisted`: only fully-persisted dataset cases
- `dataset_mixed`: only mixed-state dataset cases (persisted + accumulator
  additions/deletions)

When unset, all applicable modes are registered. Example:

```bash
SKETCH2_GBENCH_SCANNER_MODE=dataset_mixed bin/gbench_comp --benchmark_min_time=0.005s
```

The Makefile benchmark targets also set `TMPDIR=/tmp` by default so benchmark
temporary files stay on the local Linux temp filesystem instead of inheriting a
slower custom temp-directory setting.

## What Exists

### 1. `bench_comp`

Source:

- `src/core/compute/bench_compute.cpp`

Purpose:

- quick comparison of public `ComputeL1`, `ComputeL2`, and `ComputeCos`
  performance
- easy manual checks under a forced runtime backend
- simple output that is easy to paste into notes or reviews

What it benchmarks:

- `L1`, `L2`, and cosine
- `f32`, `f16`, and `i16`
- one fixed dimension: `512`
- one pair of vectors per case

How it runs:

- warms up each case
- runs `4000` iterations per sample
- repeats `7` samples
- reports median and best time per call

Important limitation:

- this benchmark only measures the public distance APIs
- it does not benchmark scanner traversal, heap updates, reader iteration, or
  dataset mixed-state behavior

### 2. `gbench_comp`

Source:

- `src/core/compute/gbench_compute.cpp`

Purpose:

- systematic benchmarking of the critical compute hot paths
- apples-to-apples comparisons across runtime backends, metrics, types, and
  dimensions
- realistic scanner-path benchmarking over both `DataReader` and `Dataset`

What it benchmarks:

- direct compute distance calls:
  - essential profile:
    - `L1`, `L2`, `COS`
    - `f32`, `f16`, `i16`
    - dimension `256`
    - runtime backends supported by the current build and CPU
  - extended profile:
    - `L1`, `L2`, `COS`
    - `f32`, `f16`, `i16`
    - dimensions `64`, `256`, `1024`
    - runtime backends supported by the current build and CPU
- scanner `find(...)`:
  - essential profile:
    - representative reader-only scanner cases
    - dimension `256`
    - `k=10`
    - cases:
      - `L2 f32` on `reader`
      - `L2 i16` on `reader`
      - `COS i16` on `reader`
  - extended profile:
    - `L1`, `L2`, `COS`
    - `f32`, `f16`, `i16`
    - dimensions `128`, `256`
    - `k=10` and `k=100`
    - scan modes:
      - `reader`
      - `dataset_persisted`
      - `dataset_mixed`
- scanner `find_items(...)`:
  - essential profile:
    - not registered
    - dataset-backed scanner benchmarks are intentionally extended-only
  - extended profile:
    - same metric/type/backend coverage as scanner `find(...)`
    - dataset-focused modes:
      - `dataset_persisted`
      - `dataset_mixed`

The scanner benchmark creates temporary synthetic data and measures the full
top-k query path rather than just the low-level distance kernel.

Within one `gbench_comp` process, scanner fixtures are reused across matching
benchmark cases so repeated runs against the same synthetic dataset shape do not
rebuild the underlying files for every backend or `k` value.

### Essential vs Extended

The essential suite keeps full coverage for the cheap direct-kernel benchmarks,
but limits scanner coverage to reader-only representative cases so the default
`make bench` run stays practical during normal development.

The extended suite restores the broader matrix for deeper comparisons and is a
better fit for long-running benchmark sweeps.

The scanner fixtures also differ:

- essential profile: smaller synthetic reader fixtures and no dataset-backed
  scanner cases
- extended profile: larger synthetic datasets for deeper scan-throughput runs

## What The Scanner Modes Mean

`gbench_comp` uses three scanner modes.

### `reader`

This benchmarks scanning a `DataReader` directly.

It measures:

- reader iteration
- distance evaluation
- top-k maintenance

It does not include:

- dataset metadata handling
- accumulator shadowing
- persisted-plus-pending merged read behavior

### `dataset_persisted`

This benchmarks a `Dataset` whose vectors have already been flushed to
persisted files.

It measures:

- dataset read-state setup
- dataset reader traversal
- distance evaluation
- top-k maintenance

It is closer to steady-state production scans over persisted data.

### `dataset_mixed`

This benchmarks a `Dataset` with both:

- persisted vectors
- pending accumulator additions and deletions

It measures the most realistic and complex scan path in this benchmark suite,
including:

- persisted reader scans
- accumulator scan
- id shadowing logic
- top-k maintenance across mixed storage state

## What The Measurements Mean

### `bench_comp` output

Example shape:

```text
requested_backend=scalar active_backend=scalar dim=512 iterations=4000 repeats=7
l2-f32           median_ns/call=1007.1       best_ns/call=956.5        median_calls/s=992960
```

Fields:

- `requested_backend`: value from `SKETCH2_COMPUTE_BACKEND`, or `auto`
- `active_backend`: runtime backend actually selected
- `dim`: benchmark vector dimension, currently fixed at `512`
- `iterations`: calls per timing sample
- `repeats`: number of samples
- `median_ns/call`: median nanoseconds per call across the repeated samples
- `best_ns/call`: best observed nanoseconds per call
- `median_calls/s`: reciprocal of `median_ns/call`

How to read it:

- use `median_ns/call` as the primary metric
- use `best_ns/call` as a lower-bound sanity check
- use `median_calls/s` when comparing throughput-oriented summaries

What it is good for:

- "is this backend faster than that backend for this metric/type?"
- "did a simple kernel regression happen?"
- "is my backend override actually being honored?"

### `gbench_comp` output

Example shape:

```text
BM_ComputeDistance/0/1/0/256          0.487 us        0.486 us        28368 bytes_per_second=3.92601Gi/s items_per_second=526.941M/s backend=scalar,metric=L2,type=f32,dim=256
BM_ScannerFindIds/0/1/0/128/10/0       4.42 ms         4.42 ms            3 components/s=474.017M/s queries/s=226.029/s vectors/s=3.70326M/s backend=scalar,metric=L2,type=f32,dim=128,k=10,mode=reader
```

The leading slash-separated integers are benchmark arguments encoded by Google
Benchmark. The human-friendly label at the end is the important part.

For direct compute benchmarks:

- `Time`: wall-clock time per iteration
- `CPU`: CPU time per iteration
- `bytes_per_second`: estimated input bandwidth
- `items_per_second`: processed vector components per second
- label fields:
  - `backend`
  - `metric`
  - `type`
  - `dim`

For scanner benchmarks:

- `Time`: wall-clock time per query
- `CPU`: CPU time per query
- `queries/s`: queries completed per second
- `vectors/s`: visible vectors scanned per second
- `components/s`: scalar components processed per second
- label fields:
  - `backend`
  - `metric`
  - `type`
  - `dim`
  - `k`
  - `mode`

How to read it:

- prefer the `CPU` column for comparisons on a busy system
- use `queries/s` for end-to-end scanner throughput
- use `vectors/s` when comparing scan efficiency independent of `k`
- use `components/s` when comparing across dimensions

## How To Run The Benchmarks

### Makefile targets

Available targets:

- `make benchcfg`
- `make benchbuild`
- `make bench`
- `make benchrel`
- `make benchext`
- `make benchcomp`

Meaning:

- `benchcfg`: configure `build/` as a release benchmark build with
  `SKETCH_ENABLE_BENCHMARKS=ON`
- `benchbuild`: build release benchmark binaries in `build/`
- `bench` / `benchrel`: run `gbench_comp` with the essential profile
- `benchext`: run `gbench_comp` with the extended profile
- `benchcomp`: run `bench_comp`

For most day-to-day use, you usually only need:

- `make bench`
- `make benchext`
- `make benchcomp`

### Direct executable runs

Typical commands:

```bash
SKETCH2_GBENCH_PROFILE=essential bin/gbench_comp --benchmark_min_time=0.005s
SKETCH2_GBENCH_PROFILE=extended bin/gbench_comp --benchmark_min_time=0.05s
SKETCH2_GBENCH_PROFILE=essential bin/gbench_comp --benchmark_filter='BM_ComputeDistance' --benchmark_min_time=0.005s
SKETCH2_GBENCH_PROFILE=essential bin/gbench_comp --benchmark_filter='BM_ScannerFindIds' --benchmark_min_time=0.005s
bin/bench_comp
```

You can also force a runtime backend:

```bash
SKETCH2_COMPUTE_BACKEND=scalar SKETCH2_GBENCH_PROFILE=essential bin/gbench_comp --benchmark_filter='BM_ComputeDistance' --benchmark_min_time=0.005s
SKETCH2_COMPUTE_BACKEND=avx2 bin/bench_comp
SKETCH2_COMPUTE_BACKEND=avx512f bin/bench_comp
SKETCH2_COMPUTE_BACKEND=avx512_vnni bin/bench_comp
```

Or restrict to a specific scanner mode:

```bash
SKETCH2_GBENCH_SCANNER_MODE=reader SKETCH2_GBENCH_PROFILE=essential bin/gbench_comp --benchmark_filter='BM_ScannerFindIds' --benchmark_min_time=0.005s
SKETCH2_GBENCH_SCANNER_MODE=dataset_mixed SKETCH2_GBENCH_PROFILE=extended bin/gbench_comp --benchmark_min_time=0.05s
```

Important behavior:

- `bench_comp` fails fast if a non-`auto` requested backend is not actually the
  active backend
- `gbench_comp` skips unsupported backend/type combinations cleanly

### Filtering Google Benchmark runs

Useful examples:

```bash
SKETCH2_GBENCH_PROFILE=essential bin/gbench_comp --benchmark_filter='BM_ComputeDistance' --benchmark_min_time=0.005s
SKETCH2_GBENCH_PROFILE=essential bin/gbench_comp --benchmark_filter='BM_ScannerFindIds' --benchmark_min_time=0.005s
SKETCH2_GBENCH_PROFILE=essential bin/gbench_comp --benchmark_filter='BM_ScannerFindItems' --benchmark_min_time=0.005s
SKETCH2_GBENCH_PROFILE=essential bin/gbench_comp --benchmark_filter='BM_ComputeDistance/0/1/0/256$' --benchmark_min_time=0.005s
SKETCH2_GBENCH_PROFILE=extended bin/gbench_comp --benchmark_min_time=0.05s
```

Recommended workflow:

1. start with a filter that matches the area you changed
2. use a slightly longer `--benchmark_min_time` for more stable numbers
3. compare the same benchmark label across builds or branches

## How To Interpret Results Correctly

### 1. Compare like with like

Always compare runs with the same:

- build type
- CPU frequency / machine state
- runtime backend
- metric
- type
- dimension
- `k`
- scanner mode

Changing any of those changes the meaning of the result.

### 2. Use release builds

Benchmarks are only meaningful in release mode. Debug builds disable
optimizations, distorting SIMD codegen, inlining, and loop unrolling, which
makes timing results misleading.

### 3. Backend labels matter

Both benchmark suites are runtime-dispatch aware. Make sure the result you are
looking at actually uses the backend you think it does.

For `bench_comp`, check:

- `requested_backend`
- `active_backend`

For `gbench_comp`, check the benchmark label:

- `backend=scalar`
- `backend=avx2`
- `backend=avx512f`
- `backend=avx512_vnni`
- `backend=neon`

### 4. Scanner benchmarks are broader than kernel benchmarks

If a direct compute benchmark improves but a scanner benchmark does not, that
usually means the bottleneck is elsewhere, for example:

- reader iteration
- heap maintenance
- dataset shadowing logic
- memory layout / cache behavior

That is normal and is exactly why both benchmark layers exist.

### 5. `vectors/s` and `components/s` are often more stable than raw time

For scanner work:

- `queries/s` is intuitive
- `vectors/s` is better for raw scan throughput
- `components/s` is useful when dimensions differ

Use all three, not just one.

## Practical Benchmarking Guidance

### For kernel work

Use:

- `gbench_comp --benchmark_filter='BM_ComputeDistance'`
- `bench_comp` for quick spot checks

Good questions:

- did `L2 f32` get faster at `dim=256`?
- did AVX-512 improve relative to AVX2?
- did scalar fallback regress?

### For scanner work

Use:

- `gbench_comp --benchmark_filter='BM_ScannerFindIds'`
- `gbench_comp --benchmark_filter='BM_ScannerFindItems'`

Good questions:

- did a scanner refactor improve persisted scans only?
- did mixed-state dataset scans regress?
- does larger `k` increase overhead meaningfully?

### For backend validation

Use:

- `SKETCH2_COMPUTE_BACKEND=...`
- startup logs
- benchmark labels/output

This is especially important when testing `avx512f` vs `avx512_vnni`.

## Current Caveats

- `bench_comp` uses one fixed dimension (`512`) and is intentionally narrow.
- `gbench_comp` covers representative dimensions, not every possible one.
- `make bench` intentionally runs the smaller essential profile rather than the
  full matrix so default benchmark runs stay interactive.
- unsupported backends are skipped based on the current build and CPU.
- Makefile benchmark targets reuse the main `build/` tree, so running them may
  reconfigure that directory for benchmark use.

## Summary

Use `bench_comp` for quick manual throughput checks. Use `gbench_comp` for real
performance investigations, regression tracking, and backend comparisons. Read
the human-friendly label fields carefully, prefer release builds for final
numbers, and compare only equivalent benchmark cases.
