# Simple Smoke Test

A long-running soak that exercises creation, staged mutations, and reads on a Sketch2 dataset. It aims to demonstrate that the system can run for extended periods without exhausting memory or disk space.

## Components
1. `driver.sh` — orchestrates the run, sets env vars, spawns processes, and collects logs.
2. `initializer.py` — creates a fresh temp DB root, writes `config.ini`, creates the dataset, and loads the initial vector set.
3. `writer.py` — performs rolling deletes/updates/inserts to keep steady write pressure.
4. `reader.py` — issues repeated KNN scans to keep steady read pressure.

## Driver
- Exports the smoke-test environment (dataset name, dims, counts, sleep/repeat counts, etc.) and resets `SKETCH2_CONFIG` to the temp DB it creates.
- Runs the initializer, then creates a per-run `logs/` directory under the temp DB.
- Starts one writer and `SIMPLE_SMOKE_TEST_READERS` reader processes; each process writes its own `stdout`/`stderr` files inside `logs/`.
- Waits for all children and surfaces the first failing exit code.

## Initializer
- Builds a new DB root in `/tmp/sketch2_simple_smoke.*`.
- Writes `config.ini` with the configured log level and thread-pool size.
- Creates the dataset with three storage parts: `part_00`, `part_01`, `part_02`.
- Loads `SIMPLE_SMOKE_TEST_COUNT` vectors generated from a deterministic pattern.

## Writer
- Opens the dataset and runs for roughly `SIMPLE_SMOKE_TEST_REPEAT / 2` iterations (at least one).
- Per iteration: deletes the oldest third of live IDs, updates the next third with a new revision, inserts the same number of new IDs, then sleeps for `SIMPLE_SMOKE_TEST_SLEEP * 2` seconds.
- Keeps the live vector count stable while steadily advancing IDs to exercise file churn and merges.

## Reader
- Opens the dataset and runs `SIMPLE_SMOKE_TEST_REPEAT` iterations.
- Per iteration: issues a KNN query (`k = SIMPLE_SMOKE_TEST_K` capped by dataset size) against a deterministic query vector, then sleeps `SIMPLE_SMOKE_TEST_SLEEP` seconds.

## Environment Variables (set by `driver.sh`)
- `SIMPLE_SMOKE_TEST_DB_DIR` — temp DB root (includes `config.ini` and logs).
- `SIMPLE_SMOKE_TEST_DATASET` — dataset name.
- `SIMPLE_SMOKE_TEST_DIMS` — vector dimensionality.
- `SIMPLE_SMOKE_TEST_COUNT` — initial vector count.
- `SIMPLE_SMOKE_TEST_SLEEP` — reader sleep; writer uses double.
- `SIMPLE_SMOKE_TEST_REPEAT` — reader iteration count; writer runs about half.
- `SIMPLE_SMOKE_TEST_READERS` — number of reader processes.
- `SIMPLE_SMOKE_TEST_K` — KNN result size.
- `SIMPLE_SMOKE_TEST_TYPE` — vector type (e.g., `f16`).
- `SIMPLE_SMOKE_TEST_DIST` — distance metric (e.g., `l2`).
- `SIMPLE_SMOKE_TEST_RANGE_SIZE` — range size for dataset creation.
- `SIMPLE_SMOKE_TEST_LOG_LEVEL` — log level written to `config.ini`.
- `SIMPLE_SMOKE_TEST_THREAD_POOL_SIZE` — thread-pool size in `config.ini`.
- `SKETCH2_CONFIG` — path to the generated `config.ini`.
