# Source Index

- `demo.py`: Small Python demo script for exercising the Sketch2 bindings with text or binary dataset loading paths.
- `integ_helpers.py`: Shared helpers for Python integration tests (temp dirs, subprocess execution, diagnostics).
- `sketch2_wrapper.py`: Python wrapper layer around the `libsketch2.so` shared library, including bulk text and binary generation helpers.
- `shell.py`: Interactive helper shell for running Sketch2 operations from Python.
- `test_demo.py`: Tests for demo-script argument handling and key execution paths.
- `test_integ_accumulator_pressure.py`: Integration tests for high-pressure accumulator flush/merge scenarios.
- `test_integ_bulk_incremental_compact.py`: Integration tests covering bulk load, incremental updates, and compaction cycles.
- `test_integ_continuous_ingestion.py`: Integration tests for repeated ingestion with interleaved reads.
- `test_integ_crash_recovery.py`: Integration tests for crash recovery via accumulator WAL replay.
- `test_integ_delete_heavy.py`: Integration tests focused on delete-heavy workloads and visibility semantics.
- `test_integ_distance_functions.py`: Integration tests validating distance-function behavior across interfaces.
- `test_integ_multi_consumer.py`: Integration tests for concurrent/multi-consumer read scenarios.
- `test_integ_multi_range.py`: Integration tests for multi-range dataset layout and query behavior.
- `test_sketch2_basic.py`: Basic Python tests for the Sketch2 wrapper.
- `test_sketch2_errors.py`: Python tests covering Sketch2 error handling.
- `test_update_notifier.py`: Python tests for update-notifier behavior and cache-invalidating update signals.
