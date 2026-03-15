# Compute Operations Guide

This note explains how the compute subsystem chooses a backend at runtime, what
inputs can influence that choice, and how to verify what the process actually
selected.

## What The System Does By Default

The compute backend is selected automatically. Normal callers do not need to
set an environment variable to make SIMD work.

At process startup, `Singleton` creates a process-wide `ComputeUnit` by calling
`ComputeUnit::detect_best()`. That selection is then reused by all distance
kernels and scanner paths for the rest of the process lifetime.

Current backend names are:

- `scalar`
- `avx2`
- `avx512f`
- `avx512_vnni`
- `neon`

Selection is architecture-aware:

- On `AArch64`, the runtime selects `neon`.
- On `x86`, the runtime prefers the fastest supported backend in this order:
  `avx512_vnni`, then `avx512f`, then `avx2`, then `scalar`.
- On other architectures, the runtime falls back to `scalar`.

The selected backend must be valid for both:

- the CPU the process is running on
- the code that was compiled into the binary

That second point matters: runtime auto-detection cannot choose a backend that
the current build did not include.

## What Influences The Selection

There are three layers that affect the final result.

### 1. CPU Capabilities

On x86, the runtime checks CPU feature bits before enabling a backend.

Current checks are:

- `avx512_vnni`: `avx512f`, `avx512bw`, `avx512vl`, and `avx512vnni`
- `avx512f`: `avx512f`
- `avx2`: `avx2`, `f16c`, and `fma`

If the CPU does not advertise the required features, that backend is skipped.

### 2. Build-Time Availability

The binary must also have been built with the relevant backend enabled.

Current build gates are:

- `SKETCH_ENABLE_AVX2`
- `SKETCH_ENABLE_AVX512F`
- `SKETCH_ENABLE_AVX512VNNI`

These are compile-time controls. If a backend was not built in, runtime
detection will never select it, even if the CPU supports it.

### 3. Optional Environment Override

If `SKETCH2_COMPUTE_BACKEND` is set, the runtime treats it as an explicit
request.

Accepted values are:

- `auto`
- `scalar`
- `avx2`
- `avx512f`
- `avx512_vnni`
- `neon`

Behavior:

- If the requested backend is known and supported on this build and CPU, it is
  selected.
- If the requested backend is unknown, the runtime falls back to auto-detect.
- If the requested backend is known but unsupported on this build or CPU, the
  runtime falls back to auto-detect.
- If the variable is unset or empty, the runtime just auto-detects.

The override is optional. It is mainly useful for testing, debugging, and
benchmarking.

## What Gets Logged

Backend selection emits an `INFO` log line describing both the chosen compute
unit and the reason it was chosen.

Examples:

- auto-detected AVX-512F because it was available and no higher-priority
  backend matched
- honored `SKETCH2_COMPUTE_BACKEND=avx2` because that backend is supported
- ignored an unsupported or unknown override and fell back to auto-detection

These logs are intended to answer two operator questions quickly:

- what compute backend is active
- why that backend won

## How To Influence Settings Safely

### For Normal Production Use

Usually, do nothing. Let the runtime auto-detect the best backend.

This is the safest mode because it adapts to the current host and avoids
forcing a backend that the machine cannot execute.

### To Force A Specific Backend

Set `SKETCH2_COMPUTE_BACKEND` before the process initializes the singleton.

Examples:

```bash
export SKETCH2_COMPUTE_BACKEND=scalar
export SKETCH2_COMPUTE_BACKEND=avx2
export SKETCH2_COMPUTE_BACKEND=avx512f
export SKETCH2_COMPUTE_BACKEND=avx512_vnni
```

Important behavior:

- forcing a backend does not bypass safety checks
- unsupported requests fall back to auto-detection
- the backend is process-wide, not per-query

### To Return To Automatic Selection

Either unset the variable or set:

```bash
export SKETCH2_COMPUTE_BACKEND=auto
```

## When Settings Are Read

Compute backend selection is effectively startup configuration.

The singleton chooses the backend when it is first constructed, and that choice
is then reused. In practice, this means environment variables that affect
selection should be set before normal runtime initialization.

Relevant startup/config environment variables in the same process are:

- `SKETCH2_COMPUTE_BACKEND`: optional compute backend override
- `SKETCH2_CONFIG`: optional config file path
- `SKETCH2_LOG_LEVEL`: optional log level override
- `SKETCH2_LOG_FILE`: optional log sink path
- `SKETCH2_THREAD_POOL_SIZE`: optional thread-pool size override

Configuration precedence for logging and thread-pool settings is:

1. built-in defaults
2. config file from `SKETCH2_CONFIG`
3. direct env vars

The compute backend follows a separate rule:

1. honor `SKETCH2_COMPUTE_BACKEND` if valid and supported
2. otherwise auto-detect

## How To Verify The Active Backend

There are three practical ways to confirm what happened.

### 1. Read The INFO Log

This is the primary operator-facing signal. The log line explains both the
selected backend and the reason.

If needed, route logs to a file by setting:

```bash
export SKETCH2_LOG_FILE=/path/to/sketch2.log
```

### 2. Use The Benchmark Tool

`bench_comp` prints both the requested backend and the active backend.

Example:

```bash
SKETCH2_COMPUTE_BACKEND=avx2 bin-dbg/bench_comp
```

If a non-`auto` backend was requested but not actually honored, the benchmark
exits with an error instead of silently timing the wrong backend.

### 3. Use Tests

Runtime dispatch tests in `utest_compute_runtime.cpp` verify that forced
backends resolve to the expected functions. This is useful when validating
changes to resolver tables or backend detection logic.

## Testing-Only Override

Tests can also force the active backend by calling:

- `Singleton::force_compute_unit_for_testing(...)`

That path is intended for unit tests. It is not the production control surface
for live per-query switching.

## Common Operational Scenarios

### "I want the fastest setting."

Leave `SKETCH2_COMPUTE_BACKEND` unset. The runtime will auto-detect the best
supported backend for the current machine and build.

### "I want reproducible benchmarks."

Set `SKETCH2_COMPUTE_BACKEND` explicitly and verify the startup log or
`bench_comp` output to ensure the request was actually honored.

### "The machine supports AVX-512, but the process still picked AVX2."

Typical causes are:

- the current build did not enable `SKETCH_ENABLE_AVX512F`
- the current build did not enable `SKETCH_ENABLE_AVX512VNNI`
- the CPU is missing one of the required runtime feature bits
- a higher-level environment or test override forced another backend

Check the startup `INFO` log first. It should explain why the selection landed
where it did.

### "Can the backend change after startup?"

Not in normal production flow. The compute unit is treated as process-wide
state and is selected once for stable dispatch behavior.

## Summary

The default behavior is automatic runtime detection. Operators only need to set
`SKETCH2_COMPUTE_BACKEND` when they want to force a backend for debugging,
testing, or benchmarking. Even then, the runtime keeps safety checks in place
and falls back to auto-detection if the request cannot be honored.
