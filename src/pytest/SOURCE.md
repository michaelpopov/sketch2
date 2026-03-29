# Source Index

- `demo.py`: Small Python demo script for exercising the Sketch2 bindings with text or binary dataset loading paths.
- `integ_helpers.py`: Shared helpers for Python integration tests (temp dirs, subprocess execution, diagnostics).
- `sketch2_wrapper.py`: Python wrapper layer around the `libsketch2.so` shared library, including bulk text and binary generation helpers.
- `shell.py`: Interactive helper shell for running Sketch2 operations from Python.
- `test_demo.py`: Tests for demo-script argument handling and key execution paths.
- `test_integ_bulk_incremental_compact.py`: Integration tests covering bulk load, incremental updates, and compaction cycles.
- `test_integ_continuous_ingestion.py`: Integration tests for repeated ingestion with interleaved reads.
