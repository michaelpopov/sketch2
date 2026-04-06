# Performance Test Harness for Compute Engines

The `compute_perf` test harness evaluates the performance and correctness of various Sketch2 compute engines (e.g., `scalar`, `auto`, `highway`, `numkong`) across different distance functions (`cos`, `l1`, `l2`).

It performs repeated K-Nearest Neighbor (KNN) queries on a large, stable dataset and compares the results against a pre-calculated ground truth to ensure that performance optimizations do not compromise accuracy.

## Architecture

The harness consists of five main components:

1.  **`driver.sh`**: The entry point. Orchestrates the test execution, manages environment variables, and collects logs. It verifies the success of the initialization phase before proceeding.
2.  **`initializer.py`**: Sets up the test environment, including creating the temporary database, configuring Sketch2, generating the dataset, and calculating/persisting the **Ground Truth**. It uses a mix of native and Python-based data generation.
3.  **`runner.py`**: Executes the actual benchmarks. For each compute engine, it benchmarks each distance function in its own child process, performs a **warm-up** iteration, follows with multiple timed iterations of KNN queries, and validates results against the shared ground truth.
4.  **`common.py`**: Shared utility library containing configuration logic, ground truth calculation, and **robust validation** (handling tie-breaking via distance comparison). It imports shared vector logic from the central `sketch2_test_vectors.py` module.
5.  **`sketch2_test_vectors.py`**: (Located in `src/pytest`) The authoritative source for all shared vector generation, quantization, formatting, and distance functions used across tests and demos.

---

## 1. Driver (`driver.sh`)

The driver is a Bash script that manages the lifecycle of a performance run.

- **Environment Setup**: Exports the internal defaults for all `COMPUTE_PERF_TEST_*` variables (see [Configuration](#configuration)) and a diagnostic directory path for child processes.
- **Isolation**: By default, creates a unique temporary directory for the database root (`SKETCH2_CONFIG_ROOT`) to avoid interference between runs. If `SKETCH2_CONFIG_ROOT` is set externally, the driver will use the provided directory instead of creating a temporary one.
- **Workflow**:
    1.  Runs `initializer.py` once to build the dataset and ground truth, unless `COMPUTE_PERF_SKIP_INIT=1` is set.
    2.  Verifies the existence of all generated dataset directories and ground truth JSON files.
    3.  Iterates through the list of engines in `COMPUTE_PERF_TEST_ENGINES`.
    4.  For each engine, sets `SKETCH2_COMPUTE_ENGINE` and executes `runner.py`. The special engine value `auto` leaves the environment variable unset so Sketch2 can use its default engine selection.
    5.  Captures stdout and stderr for each runner into separate log files. Initializer output is always streamed to stdout, and the driver recreates the log and diagnostic directories after initialization in case the initializer rebuilt the temporary database root.
- **Crash Diagnostics**: Requests core dumps when the platform allows them, logs the current core-dump limit and core pattern, writes a `run_env.txt` snapshot of the exported harness variables, and points to per-engine diagnostic files and repro scripts when a runner fails.
- **Cleanup**: Preserves the temporary database root and logs by default for inspection. Set `COMPUTE_PERF_TEST_CLEANUP=1` to delete it automatically on completion or interruption.

## 2. Initializer (`initializer.py`)

The initializer prepares the database for benchmarking.

- **Configuration**: Generates a `config.ini` file in the database root with specified log levels and thread pool sizes.
- **Safety**: Before initializing, it performs a safety check on the database directory. It only wipes the directory if it looks like a harness-owned temporary location (`/tmp/sketch2_COMPUTE_PERF.*`) or if it already contains an existing Sketch2 configuration, preventing accidental data loss.
- **Dataset Creation**: Creates one dataset for each distance function specified in `COMPUTE_PERF_TEST_DIST`. Each dataset is explicitly closed after initialization to allow sequential processing.
- **Data Generation**:
    - **L1/L2**: Uses the optimized `sketch2.generate_test_data()` (native C++ generator) to create unique, non-periodic vectors.
    - **COS**: Uses a Python-based parallel generator to produce vectors with a specific value distribution (period 6545) suitable for cosine similarity testing.
- **Ground Truth**: Calculates the exact Top-K results for each distance function and saves them as JSON files in the database root to be shared across all engine runners. For L1/L2, it uses `native_sequential_vector` to match the native generator's output. If `DUMMY_CALC=1` is set, this expensive calculation is skipped and an empty ground truth file is saved instead.

## 3. Runner (`runner.py`)

The runner performs the measurements for a single compute engine.

- **Per-Distance Isolation**: Launches a child Python process for each distance function. This localizes native crashes so the failing engine/distance pair is explicit.
- **Warm-up**: Executes one un-timed KNN query to ensure caches are primed and any lazy-initialization overhead is excluded from the performance report.
- **Measurement**: Executes `COMPUTE_PERF_TEST_REPEAT` iterations of a KNN query.
- **Validation**: Loads the pre-calculated Ground Truth from the JSON file. Every warm-up and timed result is validated. To avoid false positives due to tie-breaking differences between optimized engines, the validator requires unique IDs and compares the sorted returned-distance multiset against the expected distances. If `DUMMY_CALC=1` is set, validation is skipped.
- **Reporting**: Collects timing data using `time.perf_counter()` and prints a performance report containing Min, Max, and Average execution times.
- **Crash Diagnostics**: Writes a per-engine/per-distance JSON state file containing the last completed stage, dataset paths, query digest, expected-ID preview, PID, timing summary, and generated repro scripts. If a child process segfaults, the state file still shows the last stage reached before the crash. The runner also emits one-shot and loop-based repro shell scripts for the exact engine/distance pair.

## 4. Shared Logic (`common.py` and `sketch2_test_vectors.py`)

These modules ensure consistency between the initializer and the runner.

- **Vector Generation (`sketch2_test_vectors.py`)**:
    - `cosine_demo_vector`: Generates vectors optimized for cosine similarity (period 6545).
    - `native_sequential_vector`: Produces unique vectors matching the native `sk_generate_test_data` pattern for L1/L2 distances.
    - `quantize_value`/`quantize_values`: Ensures consistent floating-point behavior across different data types (`f32`, `f16`, `i16`).
- **Ground Truth & Persistence (`common.py`)**:
    - Implements pure-Python versions of `cosine_distance`, `l1_distance`, and `l2_distance_sq`.
    - `get_ground_truth_knn`: Efficiently calculates the exact top-K indices and distances. For L1/L2, it processes the full dataset to account for unique vectors.
    - `save_ground_truth`/`load_ground_truth`: Handles JSON serialization of ground truth data.
- **Robust Validation**: `validate_knn_results` handles distance ties to ensure correctness verification is reliable across different SIMD-optimized engines.
- **Configuration**: `load_config` parses environment variables into a `PerfConfig` dataclass, ensuring type safety and providing defaults.

---

## Configuration

The harness is configured via environment variables.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `SKETCH2_CONFIG_ROOT` | Root directory for the temporary database. | `/tmp/sketch2_COMPUTE_PERF.XXXXXX` |
| `COMPUTE_PERF_SKIP_INIT` | Skip the initialization phase (dataset generation and ground truth calculation) if set to `1`. | `0` |
| `COMPUTE_PERF_TEST_DATASET` | Base name for the datasets. | `perf_test` |
| `COMPUTE_PERF_TEST_DIMS` | Number of dimensions per vector. | `256` |
| `COMPUTE_PERF_TEST_COUNT` | Number of vectors to generate. | `100000` |
| `COMPUTE_PERF_TEST_REPEAT` | Number of query iterations per engine. | `10` |
| `COMPUTE_PERF_TEST_K` | Number of nearest neighbors to find. | `20` |
| `COMPUTE_PERF_TEST_TYPE` | Data type of vectors (`f32`, `f16`, `i16`). | `f32` |
| `COMPUTE_PERF_TEST_DIST` | Comma-separated list of distance functions. | `cos,l2,l1` |
| `COMPUTE_PERF_TEST_RANGE_SIZE` | Dataset range size used at creation time. | `10000` |
| `COMPUTE_PERF_TEST_ENGINES` | Comma-separated list of engines to test. Use `auto` for the library default selection. | `scalar,auto,highway,numkong` |
| `COMPUTE_PERF_TEST_LOG_LEVEL` | Log level for the Sketch2 engine. | `ERROR` |
| `COMPUTE_PERF_TEST_THREAD_POOL_SIZE` | Internal thread pool size for Sketch2. | `1` |
| `COMPUTE_PERF_TEST_CLEANUP` | Delete the temporary database root after the run (`1`) or preserve it (`0`). | `0` |
| `DUMMY_CALC` | If `1`, skips ground truth calculation and scan results validation. | `0` |
| `COMPUTE_PERF_DIAG_DIR` | Directory where per-distance diagnostic JSON files and repro scripts are written. | `${SKETCH2_CONFIG_ROOT}/logs/diag` |

---

## How to Run

Simply execute the driver script from the repository root:

```bash
./tests/compute_perf/driver.sh
```

Logs and timing reports will be printed to stdout and saved in `${SKETCH2_CONFIG_ROOT}/logs`.

The driver also writes `${SKETCH2_CONFIG_ROOT}/logs/run_env.txt`, which captures the exported harness configuration used for the run. This file is recreated after initialization so it survives cases where `initializer.py` rebuilds the temporary root.

When investigating a crash, inspect `${COMPUTE_PERF_DIAG_DIR}/diag_<engine>_<dist>.json` for the last recorded stage, then rerun the generated `${COMPUTE_PERF_DIAG_DIR}/repro_<engine>_<dist>.sh` or `${COMPUTE_PERF_DIAG_DIR}/repro_loop_<engine>_<dist>.sh`. On failure, `driver.sh` prints the diagnostic directory and the matching generated artifact paths to make that handoff explicit.

## Performance Test Results on Arm

Observations:
1. Custom SIMD functions in src/core/compute perform better than library functions in Google Highway or NumKong.
2. Main outlier is COS distance calculations using NumKong. It's surprisingly underperforming.

Conclusion:
Let's keep "auto" compute engine implmented in src/core/compute.


```
===== COS ============================================

[runner] running 10 iterations for cos
--- PERFORMANCE REPORT ---
Compute Engine: scalar
Distance:       cos
Iterations:     10
Min Time:       1.843347s
Max Time:       1.903374s
Avg Time:       1.874914s

[runner] running 10 iterations for cos
--- PERFORMANCE REPORT ---
Compute Engine: auto
Distance:       cos
Iterations:     10
Min Time:       0.710732s
Max Time:       0.899973s
Avg Time:       0.733400s

[runner] running 10 iterations for cos
--- PERFORMANCE REPORT ---
Compute Engine: highway
Distance:       cos
Iterations:     10
Min Time:       0.825271s
Max Time:       0.873012s
Avg Time:       0.830884s

[runner] running 10 iterations for cos
--- PERFORMANCE REPORT ---
Compute Engine: numkong
Distance:       cos
Iterations:     10
Min Time:       1.465265s
Max Time:       1.529799s
Avg Time:       1.484654s

===== L2 ============================================

[runner] running 10 iterations for l2
--- PERFORMANCE REPORT ---
Compute Engine: scalar
Distance:       l2
Iterations:     10
Min Time:       2.273663s
Max Time:       2.295021s
Avg Time:       2.287499s

--- PERFORMANCE REPORT ---
Compute Engine: auto
Distance:       l2
Iterations:     10
Min Time:       0.721202s
Max Time:       0.725755s
Avg Time:       0.723489s

[runner] running 10 iterations for l2
--- PERFORMANCE REPORT ---
Compute Engine: highway
Distance:       l2
Iterations:     10
Min Time:       0.749104s
Max Time:       0.759716s
Avg Time:       0.751441s


[runner] running 10 iterations for l2
--- PERFORMANCE REPORT ---
Compute Engine: numkong
Distance:       l2
Iterations:     10
Min Time:       1.485843s
Max Time:       1.573914s
Avg Time:       1.512157s

===== L1 ============================================

[runner] running 10 iterations for l1
--- PERFORMANCE REPORT ---
Compute Engine: scalar
Distance:       l1
Iterations:     10
Min Time:       1.785692s
Max Time:       1.827798s
Avg Time:       1.810127s

[runner] running 10 iterations for l1
--- PERFORMANCE REPORT ---
Compute Engine: auto
Distance:       l1
Iterations:     10
Min Time:       0.615906s
Max Time:       0.729482s
Avg Time:       0.628406s

[runner] running 10 iterations for l1
--- PERFORMANCE REPORT ---
Compute Engine: highway
Distance:       l1
Iterations:     10
Min Time:       0.745369s
Max Time:       0.857848s
Avg Time:       0.759219s

[runner] running 10 iterations for l1
--- PERFORMANCE REPORT ---
Compute Engine: numkong
Distance:       l1
Iterations:     10
Min Time:       0.748876s
Max Time:       0.872248s
Avg Time:       0.767781s


```