# Performance Test Harness for Compute Engines

The `compute_perf` test harness evaluates the performance and correctness of
the current Sketch2 compute path across different score functions (`cos`, `dot`,
`l2`).

After the compute redesign, the top-level engine is selected at build time via
`SKETCH2_COMPUTE_ENGINE`. In practice that means a given build benchmarks
either `highway` or `numkong`.

It performs repeated K-Nearest Neighbor (KNN) queries on a large, stable dataset and compares the results against a pre-calculated ground truth to ensure that performance optimizations do not compromise accuracy.

## Architecture

The harness consists of six main components:

1.  **`driver.sh` / `driver.py`**: The entry point. `driver.sh` is a thin compatibility wrapper around the Python driver, which orchestrates the test execution, manages environment variables, selects the runtime directory, and collects logs. It reuses a persistent dataset cache when available.
2.  **`initializer.py`**: Sets up the test environment, including creating the temporary database, configuring Sketch2, generating the dataset, and calculating/persisting the **Ground Truth**. It uses a mix of native and Python-based data generation.
3.  **`runner.py`**: Executes the actual benchmarks. For each compute engine, it benchmarks each score function in its own child process. It can run two complementary benchmark layers:
    - a **kernel-only** layer that times direct metric kernels without scanner traversal
    - a **scan** layer that performs full KNN queries through the existing dataset/scanner path
4.  **`bench_compute`**: A native benchmark executable under `src/core/compute` that measures the direct kernels and composed scan-time score paths for the selected engine, metric, type, and dimension. It always reports `dist`, adds `dot` and `squared_norm` whenever those kernels are available, and includes composed stored-norm paths (`dist_with_stored_norms`) plus the cosine query-norm fallback (`dist_with_query_norm`) when supported.
5.  **`common.py`**: Shared utility library containing configuration logic, binary discovery, ground truth calculation, and **robust validation** (handling tie-breaking via score comparison). It imports shared vector logic from the central `sketch2_test_vectors.py` module.
6.  **`sketch2_test_vectors.py`**: (Located in `src/pytest`) The authoritative source for all shared vector generation, quantization, formatting, and score functions used across tests and demos.

---

## 1. Driver (`driver.sh` / `driver.py`)

The driver is a Python script with a small shell wrapper that manages the lifecycle of a performance run.

- **Environment Setup**: Applies defaults for all `COMPUTE_PERF_TEST_*` variables only when they are not already set (see [Configuration](#configuration)) and exports a diagnostic directory path for child processes.
- **Runtime Selection**: `--engine` is required. Perf runs intentionally use release artifacts only. Passing `--engine highway` uses `REPO_ROOT/bin-hwy`, while `--engine numkong` uses `REPO_ROOT/bin-nk`. The driver uses `Sketch2.compute_engine()` backed by the `sk_compute_engine()` API from the selected `libsketch2.so` to report the compiled engine and rejects mismatches.
- **Persistent Cache**: By default, uses a fixed dataset cache root at `/tmp/sketch2_tests_compute_perf`. If `SKETCH2_CONFIG_ROOT` is set externally, the driver uses that directory instead.
- **Metadata Authority**: The cache root stores `dataset_metadata.json`. When that file exists, its dataset shape (`count`, `dims`, `k`, `type`, `dist`, `range_size`, dataset name) overrides the driver defaults and is reported in the driver output.
- **Workflow**:
    1.  If the cache root does not exist, runs `initializer.py` once to create all datasets and ground-truth files, then writes `dataset_metadata.json`.
    2.  If the cache root already exists, requires `dataset_metadata.json` and reuses the existing datasets instead of regenerating them.
    3.  Verifies the existence of all dataset directories and ground-truth JSON files described by the metadata.
    4.  For each score function, runs `runner.py` against the single compiled engine reported by the loaded library. The driver leaves `SKETCH2_COMPUTE_ENGINE` unset so Sketch2 uses the library's built-in engine selection.
    5.  Captures stdout and stderr for each initializer/runner invocation into separate log files.
    6.  Runs `reporter.py` at the end so each harness invocation emits the summary tables for the just-collected logs.
- **Crash Diagnostics**: Requests core dumps when the platform allows them, logs the current core-dump limit and core pattern, writes a `run_env.txt` snapshot of the exported harness variables, and points to per-engine diagnostic files and repro scripts when a runner fails.
- **Cleanup**: Preserves the dataset cache and logs by default so later runs can reuse them. Set `COMPUTE_PERF_TEST_CLEANUP=1` only if you explicitly want the cache root removed after the run.

## 2. Initializer (`initializer.py`)

The initializer prepares the database for benchmarking.

- **Configuration**: Generates a `config.ini` file in the database root with specified log levels and thread pool sizes.
- **Safety**: Before initializing, it performs a safety check on the database directory. It only wipes the directory if it looks like a harness-owned temporary location (`/tmp/sketch2_COMPUTE_PERF.*`) or if it already contains an existing Sketch2 configuration, preventing accidental data loss.
- **Dataset Creation**: Creates one dataset for each score function specified in `COMPUTE_PERF_TEST_DIST`. Each dataset is explicitly closed after initialization to allow sequential processing.
- **Data Generation**:
    - **DOT/L2**: Uses the optimized `sketch2.generate_test_data()` (native C++ generator) to create unique, non-periodic vectors.
    - **COS**: Uses a Python-based parallel generator to produce vectors with a specific value distribution (period 6545) suitable for cosine similarity testing.
    - **Ground Truth**: Calculates the exact Top-K results for each score function and saves them as JSON files in the database root to be shared across all engine runners. For DOT/L2, it uses `native_sequential_vector` to match the native generator's output.

## 3. Runner (`runner.py`)

The runner performs the measurements for a single compute engine.

- **Per-Metric Isolation**: Launches a child Python process for each score function. This localizes native crashes so the failing engine/metric pair is explicit.
- **Kernel Benchmark Layer**: When `kernel` is enabled in `COMPUTE_PERF_TEST_BENCHMARKS`, runs the native `bench_compute` executable first and records direct-kernel timings without any dataset traversal, heap maintenance, or scanner logic. This makes it much easier to distinguish kernel regressions from scan-path overhead.
- **Warm-up**: Executes one un-timed KNN query to ensure caches are primed and any lazy-initialization overhead is excluded from the performance report.
- **Scan Benchmark Layer**: When `scan` is enabled in `COMPUTE_PERF_TEST_BENCHMARKS`, executes `COMPUTE_PERF_TEST_REPEAT` iterations of a KNN query.
- **Validation**: Loads the pre-calculated Ground Truth from the JSON file. Every warm-up and timed result is validated. To avoid false positives due to tie-breaking differences between optimized engines, the validator requires unique IDs and compares the sorted returned-score multiset against the expected scores.
- **Reporting**: Prints a kernel performance report when kernel mode is enabled, and a scan performance report containing Min, Max, and Average query times when scan mode is enabled.
- **Crash Diagnostics**: Writes a per-engine/per-metric JSON state file containing the last completed stage, dataset paths, query digest, expected-ID preview, PID, timing summary, and generated repro scripts. If a child process segfaults, the state file still shows the last stage reached before the crash. The runner also emits one-shot and loop-based repro shell scripts for the exact engine/metric pair.

## 4. Shared Logic (`common.py` and `sketch2_test_vectors.py`)

These modules ensure consistency between the initializer and the runner.

- **Vector Generation (`sketch2_test_vectors.py`)**:
    - `cosine_demo_vector`: Generates vectors optimized for cosine similarity (period 6545).
    - `native_sequential_vector`: Produces unique vectors matching the native `sk_generate_test_data` pattern for DOT/L2 score functions.
    - `quantize_value`/`quantize_values`: Ensures consistent floating-point behavior across different data types (`f32`, `f16`, `i16`).
- **Ground Truth & Persistence (`common.py`)**:
    - Implements pure-Python versions of `cosine_distance`, `dot_distance`, and `l2_distance_sq`.
    - `get_ground_truth_knn`: Efficiently calculates the exact top-K indices and scores. For DOT/L2, it processes the full dataset to account for unique vectors.
    - `save_ground_truth`/`load_ground_truth`: Handles JSON serialization of ground truth data.
- **Robust Validation**: `validate_knn_results` handles score ties to ensure correctness verification is reliable across different SIMD-optimized engines.
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
| `COMPUTE_PERF_TEST_REPEAT` | Number of query iterations per engine. | `10` |
| `COMPUTE_PERF_TEST_K` | Number of nearest neighbors to find. | `20` |
| `COMPUTE_PERF_TEST_TYPE` | Data type of vectors (`f32`, `f16`, `i16`). | `f32` |
| `COMPUTE_PERF_TEST_DIST` | Comma-separated list of score functions. | `cos,l2,dot` |
| `COMPUTE_PERF_TEST_RANGE_SIZE` | Dataset range size used at creation time. | `10000` |
| `COMPUTE_PERF_TEST_ENGINES` | Engine labels shown in the final summary tables. The driver now sets this to the single compiled engine reported by the selected `libsketch2.so`. | detected from `sk_compute_engine()` |
| `COMPUTE_PERF_TEST_BENCHMARKS` | Comma-separated benchmark layers to run. Supported values: `scan`, `kernel`. | `scan,kernel` |
| `COMPUTE_PERF_TEST_LOG_LEVEL` | Log level for the Sketch2 engine. | `ERROR` |
| `COMPUTE_PERF_TEST_THREAD_POOL_SIZE` | Internal thread pool size for Sketch2. | `1` |
| `COMPUTE_PERF_KERNEL_ITERATIONS` | Calls per timing sample in the kernel-only benchmark. | `200000` |
| `COMPUTE_PERF_KERNEL_WARMUP_ITERATIONS` | Un-timed warm-up calls before kernel measurement. | `5000` |
| `COMPUTE_PERF_KERNEL_REPEATS` | Number of kernel timing samples per case. | `7` |
| `COMPUTE_PERF_TEST_CLEANUP` | Delete the temporary database root after the run (`1`) or preserve it (`0`). | `0` |
| `COMPUTE_PERF_DIAG_DIR` | Directory where per-metric diagnostic JSON files and repro scripts are written. | `${SKETCH2_CONFIG_ROOT}/logs/diag` |

---

## How to Run

Build the release runtime for the engine you want to benchmark, then execute the
driver script from the repository root.

For the Highway build:

```bash
make rel
./tests/compute_perf/driver.sh --engine highway
```

For the NumKong build:

```bash
make rel-nk
./tests/compute_perf/driver.sh --engine numkong
```

The harness does not use the default debug outputs under `bin-dbg-hwy` or
`bin-dbg-nk`; that is intentional so performance numbers come from release
builds only.

Logs and timing reports will be printed to stdout and saved in `${SKETCH2_CONFIG_ROOT}/logs`.

The final reporter prints two summary tables:

- **Performance Summary**: end-to-end scan average time per engine and metric
- **Kernel Summary**: direct `dist` kernel average nanoseconds per call per engine and metric

The driver also writes `${SKETCH2_CONFIG_ROOT}/logs/run_env.txt`, which captures the exported harness configuration used for the run. The dataset cache writes `${SKETCH2_CONFIG_ROOT}/dataset_metadata.json`, and that file becomes the authoritative source for dataset shape on later runs.

When investigating a crash, inspect `${COMPUTE_PERF_DIAG_DIR}/diag_<engine>_<dist>.json` for the last recorded stage, then rerun the generated `${COMPUTE_PERF_DIAG_DIR}/repro_<engine>_<dist>.sh` or `${COMPUTE_PERF_DIAG_DIR}/repro_loop_<engine>_<dist>.sh`. On failure, `driver.sh` prints the diagnostic directory and the matching generated artifact paths to make that handoff explicit.

## Notes After The Compute Redesign

- Historical observations about `src/core/compute` no longer describe the
  current code path. The active query implementation now lives under
  `src/core/compute`.
- Use separate build directories if you want to compare Highway and NumKong.
  The redesign made top-level engine choice a configure-time decision, not a
  broad runtime matrix inside one build tree.
- Perf runs are intentionally release-only. Build `bin-hwy` with `make rel`
  and `bin-nk` with `make rel-nk` before invoking the harness.
- `bench_compute` is the authoritative native microbenchmark entry point for the
  current compute layer.
