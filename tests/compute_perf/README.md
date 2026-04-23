# Performance Test Harness

The `compute_perf` test harness evaluates the performance of the current
Sketch2 runtime across different score functions (`cos`, `dot`, `l2`).

It performs repeated K-Nearest Neighbor (KNN) queries on a large, stable dataset
to measure end-to-end scan throughput and direct kernel timings.

## Architecture

The harness consists of six main components:

1.  **`driver.sh` / `driver.py`**: The entry point. `driver.sh` is a thin compatibility wrapper around the Python driver, which orchestrates the test execution, manages environment variables, selects the runtime directory, and collects logs. It reuses a persistent dataset cache when available.
2.  **`initializer.py`**: Sets up the test environment, including creating the temporary database, configuring Sketch2, and generating the datasets. It uses the native metric-aware test-data generator exposed by Sketch2.
3.  **`runner.py`**: Executes the actual benchmarks. It runs each score function in its own child process. It can run two complementary benchmark layers:
    - a **kernel-only** layer that times direct metric kernels without scanner traversal
    - a **scan** layer that performs full KNN queries through the existing dataset/scanner path
4.  **`bench_compute`**: A native benchmark executable under `src/core/compute` that measures the direct Highway kernels and composed scan-time score paths for the selected metric, type, and dimension. It always reports `dist`, adds `dot` and `squared_norm` whenever those kernels are available, and includes composed stored-norm paths (`dist_with_stored_norms`) plus the cosine query-norm fallback (`dist_with_query_norm`) when supported.
5.  **`common.py`**: Shared utility library containing configuration logic, binary discovery, query generation, and small reporting helpers. It imports shared vector logic from the central `sketch2_test_vectors.py` module.
6.  **`sketch2_test_vectors.py`**: (Located in `src/pytest`) The authoritative source for all shared vector generation, quantization, formatting, and score functions used across tests and demos.

---

## 1. Driver (`driver.sh` / `driver.py`)

The driver is a Python script with a small shell wrapper that manages the lifecycle of a performance run.

- **Environment Setup**: Applies defaults for all `COMPUTE_PERF_TEST_*` variables only when they are not already set (see [Configuration](#configuration)) and exports a diagnostic directory path for child processes.
- **Runtime Identification**: Perf runs intentionally use release artifacts only from `REPO_ROOT/bin`. The driver uses `Sketch2.compute_engine()` backed by the `sk_compute_engine()` API from the selected `libsketch2.so` to confirm the compiled runtime label.
- **Persistent Cache**: By default, uses a fixed dataset cache root at `/tmp/sketch2_tests_compute_perf`. If `SKETCH2_CONFIG_ROOT` is set externally, the driver uses that directory instead.
- **Metadata Authority**: The cache root stores `dataset_metadata.json`. When that file exists, its dataset shape (`count`, `dims`, `k`, `type`, `dist`, `range_size`, dataset name) overrides the driver defaults and is reported in the driver output.
- **Workflow**:
    1.  If the cache root does not exist, runs `initializer.py` once to create all datasets, then writes `dataset_metadata.json`.
    2.  If the cache root already exists, requires `dataset_metadata.json` and reuses the existing datasets instead of regenerating them.
    3.  Verifies the existence of all dataset directories described by the metadata.
    4.  For each score function, runs `runner.py` against the single Highway runtime reported by the loaded library.
    5.  Captures stdout and stderr for each initializer/runner invocation into separate log files.
    6.  Runs `reporter.py` at the end so each harness invocation emits the summary tables for the just-collected logs.
- **Crash Diagnostics**: Requests core dumps when the platform allows them, logs the current core-dump limit and core pattern, writes a `run_env.txt` snapshot of the exported harness variables, and points to per-runtime diagnostic files and repro scripts when a runner fails.
- **Cleanup**: Preserves the dataset cache and logs by default so later runs can reuse them. Set `COMPUTE_PERF_TEST_CLEANUP=1` only if you explicitly want the cache root removed after the run.

## 2. Initializer (`initializer.py`)

The initializer prepares the database for benchmarking.

- **Configuration**: Generates a `config.ini` file in the database root with specified log levels and thread pool sizes.
- **Safety**: Before initializing, it performs a safety check on the database directory. It only wipes the directory if it looks like a harness-owned temporary location (`/tmp/sketch2_COMPUTE_PERF.*`) or if it already contains an existing Sketch2 configuration, preventing accidental data loss.
- **Dataset Creation**: Creates one dataset for each score function specified in `COMPUTE_PERF_TEST_DIST`. Each dataset is explicitly closed after initialization to allow sequential processing.
- **Data Generation**:
    - Uses the optimized `sketch2.generate_test_data()` (native C++ generator) with the shared `perf_test` pattern to generate one binary corpus, load it into the first dataset, and then reuse that same file for the remaining datasets.

## 3. Runner (`runner.py`)

The runner performs the measurements for the compiled Highway runtime.

- **Per-Metric Isolation**: Launches a child Python process for each score function. This localizes native crashes so the failing metric pair is explicit.
- **Kernel Benchmark Layer**: When `kernel` is enabled in `COMPUTE_PERF_TEST_BENCHMARKS`, runs the native `bench_compute` executable first and records direct-kernel timings without any dataset traversal, heap maintenance, or scanner logic. This makes it much easier to distinguish kernel regressions from scan-path overhead.
- **Warm-up**: Executes one un-timed KNN query to ensure caches are primed and any lazy-initialization overhead is excluded from the performance report.
- **Scan Benchmark Layer**: When `scan` is enabled in `COMPUTE_PERF_TEST_BENCHMARKS`, executes `COMPUTE_PERF_TEST_REPEAT` iterations of a KNN query.
- **Reporting**: Prints a kernel performance report when kernel mode is enabled, and a scan performance report containing Min, Max, and Average query times when scan mode is enabled.
- **Crash Diagnostics**: Writes a per-runtime/per-metric JSON state file containing the last completed stage, dataset paths, query digest, PID, timing summary, and generated repro scripts. If a child process segfaults, the state file still shows the last stage reached before the crash. The runner also emits one-shot and loop-based repro shell scripts for the exact runtime/metric pair.

## 4. Shared Logic (`common.py` and `sketch2_test_vectors.py`)

These modules ensure consistency between the initializer and the runner.

- **Vector Generation (`sketch2_test_vectors.py`)**:
    - `cosine_demo_vector`: Generates vectors optimized for cosine similarity (period 6545).
    - `quantize_value`/`quantize_values`: Ensures consistent floating-point behavior across different data types (`f32`, `f16`, `i16`).
- **Query Generation (`common.py`)**:
    - Uses `cosine_demo_query` for cosine datasets and `generic_demo_query` for DOT/L2 datasets.
- **Configuration**: `load_config` parses environment variables into a `PerfConfig` dataclass, ensuring type safety and providing defaults.

---

## Configuration

The harness is configured via environment variables.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `SKETCH2_CONFIG_ROOT` | Root directory for the persistent dataset cache. | `/tmp/sketch2_tests_compute_perf` |
| `COMPUTE_PERF_SKIP_INIT` | Legacy knob. The driver now prefers automatic cache reuse based on `dataset_metadata.json`. | `0` |
| `COMPUTE_PERF_TEST_DATASET` | Base name for the datasets. | `perf_test` |
| `COMPUTE_PERF_TEST_DIMS` | Number of dimensions per vector. | `256` |
| `COMPUTE_PERF_TEST_COUNT` | Number of vectors to generate. | `100000` |
| `COMPUTE_PERF_TEST_REPEAT` | Number of query iterations per run. | `10` |
| `COMPUTE_PERF_TEST_K` | Number of nearest neighbors to find. | `20` |
| `COMPUTE_PERF_TEST_TYPE` | Data type of vectors (`f32`, `f16`, `i16`). | `f32` |
| `COMPUTE_PERF_TEST_DIST` | Comma-separated list of score functions. | `cos,l2,dot` |
| `COMPUTE_PERF_TEST_RANGE_SIZE` | Dataset range size used at creation time. | `10000` |
| `COMPUTE_PERF_RUNTIME_LABEL` | Runtime label shown in the final summary tables. The driver sets this from `sk_compute_engine()`. | `highway` |
| `COMPUTE_PERF_TEST_BENCHMARKS` | Comma-separated benchmark layers to run. Supported values: `scan`, `kernel`. | `scan,kernel` |
| `COMPUTE_PERF_TEST_LOG_LEVEL` | Log level for the Sketch2 runtime. | `ERROR` |
| `COMPUTE_PERF_TEST_THREAD_POOL_SIZE` | Internal thread pool size for Sketch2. | `1` |
| `COMPUTE_PERF_KERNEL_ITERATIONS` | Calls per timing sample in the kernel-only benchmark. | `200000` |
| `COMPUTE_PERF_KERNEL_WARMUP_ITERATIONS` | Un-timed warm-up calls before kernel measurement. | `5000` |
| `COMPUTE_PERF_KERNEL_REPEATS` | Number of kernel timing samples per case. | `7` |
| `COMPUTE_PERF_TEST_CLEANUP` | Delete the temporary database root after the run (`1`) or preserve it (`0`). | `0` |
| `COMPUTE_PERF_DIAG_DIR` | Directory where per-metric diagnostic JSON files and repro scripts are written. | `${SKETCH2_CONFIG_ROOT}/logs/diag` |

---

## How to Run

Build the release runtime, then execute the driver script from the repository
root.

```bash
make rel
./tests/compute_perf/driver.sh
```

The harness does not use the default debug output under `bin-dbg`; that is
intentional so performance numbers come from release builds only.

Logs and timing reports will be printed to stdout and saved in `${SKETCH2_CONFIG_ROOT}/logs`.

The final reporter prints two summary tables:

- **Performance Summary**: end-to-end scan average time per runtime and metric
- **Kernel Summary**: direct `dist` kernel average nanoseconds per call per runtime and metric

The driver also writes `${SKETCH2_CONFIG_ROOT}/logs/run_env.txt`, which captures the exported harness configuration used for the run. The dataset cache writes `${SKETCH2_CONFIG_ROOT}/dataset_metadata.json`, and that file becomes the authoritative source for dataset shape on later runs.

When investigating a crash, inspect `${COMPUTE_PERF_DIAG_DIR}/diag_<runtime>_<dist>.json` for the last recorded stage, then rerun the generated `${COMPUTE_PERF_DIAG_DIR}/repro_<runtime>_<dist>.sh` or `${COMPUTE_PERF_DIAG_DIR}/repro_loop_<runtime>_<dist>.sh`. On failure, `driver.sh` prints the diagnostic directory and the matching generated artifact paths to make that handoff explicit.

## Notes After The Compute Redesign

- Historical observations about `src/core/compute` no longer describe the
  current code path. The active query implementation now lives under
  `src/core/compute`.
- Perf runs are intentionally release-only. Build `bin` with `make rel`
  before invoking the harness.
- `bench_compute` is the authoritative native microbenchmark entry point for the
  current compute layer.
