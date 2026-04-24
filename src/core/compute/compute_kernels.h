// Declares the generic compute-kernel function-pointer types and the
// ComputeKernels bundle used by the scanner helpers, benchmarks, and
// kernel-focused tests.

#pragma once
#include "utils/shared_types.h"
#include <cstdint>

namespace sketch2 {

// Function pointer types matching the existing compute kernel conventions.
using ComputeDistFn              = double (*)(const uint8_t*, const uint8_t*, size_t);
using ComputeDistWithQueryNormFn = double (*)(const uint8_t*, const uint8_t*, size_t, double);
using ComputeSquaredNormFn       = double (*)(const uint8_t*, size_t);
using ComputeDotFn               = double (*)(const uint8_t*, const uint8_t*, size_t);

// Holds all resolved function pointers for one (metric, DataType) combination.
// Benchmarks and kernel-focused tests use this runtime-resolved view to
// exercise helper kernels directly.
//
// For DOT, both `dist` and `dot` are populated with the same kernel so callers
// can use explicit dot-oriented names on the DOT execution path.
// For L2, `dist` is always populated and `dot`/`squared_norm` may also be
// populated so callers can compare the raw path with the stored-norm helpers.
// For COS all four fields are populated.
struct ComputeKernels {
    ComputeDistFn              dist = nullptr;
    ComputeDistWithQueryNormFn dist_with_query_norm = nullptr;
    ComputeSquaredNormFn       squared_norm = nullptr;
    ComputeDotFn               dot = nullptr;
};

} // namespace sketch2
