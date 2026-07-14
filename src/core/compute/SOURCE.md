# Source Index

- `CMakeLists.txt`: Build rules for the compute library, benchmarks, and tests.
- `bench_compute.cpp`: Direct kernel benchmark for the compiled Highway backend.
- `compute_kernels.h`: Generic compute-kernel function-pointer types and the ComputeKernels bundle used by scanner helpers, benchmarks, and tests.
- `compute_value_helpers.h`: Deterministic type-specific value fillers for compute tests and benchmarks; its f8 filler selects grid-exact codebook values.
- `dist_item.h`: Ranked-result types and ordering helpers shared by compute scanners.
- `highway.cpp`: Highway-backed scanner and kernel resolver using the foreach_target pattern for multi-target compilation and runtime dispatch.
- `highway.h`: Highway-backed scan entry point and kernel resolver declarations.
- `metric_finalizers.h`: Distance-metric finalizers that combine intermediate values (dot product, stored norms) into the public distance contract for each metric.
- `norm_utils.h`: Scalar norm helpers shared by compute and storage code.
- `scanner.h`: Highway-backed top-k scanner API declarations.
- `scanner_dataset_scan.h`: Shared dataset-level scanner traversal helpers.
- `scanner_heap_utils.h`: Shared scanner heap and result utilities.
- `scanner_log_utils.h`: Shared scanner logging helpers.
- `scanner_query_context.h`: Per-query context structs passed to scan kernels.
- `scanner_scan_loops.h`: Shared scanner hot scan loops.
- `utest_compute_helpers.h`: Shared helper utilities for compute kernel unit tests.
- `utest_hwy_kernels.cpp`: Unit tests for Highway distance kernels.
- `utest_scanner.cpp`: Unit tests for Scanner nearest-neighbor scanning.
