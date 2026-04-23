// Defines the compute-engine abstraction used by compute logging/metadata and by
// benchmark/kernel-test helpers that resolve metric kernels for the compiled
// backend.

#pragma once
#include "utils/shared_types.h"
#include <cstdint>

namespace sketch2 {

enum class ComputeEngine : uint8_t {
    highway,
};

constexpr ComputeEngine compiled_compute_engine() {
    return ComputeEngine::highway;
}

const char* compute_engine_name(ComputeEngine engine);

// Function pointer types matching the existing compute kernel conventions.
using ComputeDistFn             = double (*)(const uint8_t*, const uint8_t*, size_t);
using ComputeDistWithQueryNormFn = double (*)(const uint8_t*, const uint8_t*, size_t, double);
using ComputeSquaredNormFn      = double (*)(const uint8_t*, size_t);
using ComputeDotFn              = double (*)(const uint8_t*, const uint8_t*, size_t);

// Holds all resolved function pointers for one (metric, DataType) combination
// in the compiled compute backend. Benchmarks and kernel-focused tests use this
// runtime-resolved view to exercise helper kernels directly.
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

// Resolves the benchmark/test kernel set for the compiled engine.
ComputeKernels resolve_compute_kernels(DistFunc func, DataType type);

// Performs one-time compute-engine runtime warm-up for the compiled backend.
// Highway uses this to resolve and cache its final dispatched function pointers
// before scan hot loops begin.
void initialize_compute_engine_runtime();

} // namespace sketch2
