// Defines the calc-engine abstraction: an engine selector enum and a struct of
// resolved distance-kernel function pointers. ScannerEx dispatches through
// CalcKernels so the scanning logic is completely decoupled from the concrete
// SIMD library backend.

#pragma once
#include "core/utils/compute_unit.h"
#include "utils/shared_types.h"
#include <cstdint>
#include <stdexcept>

namespace sketch2 {

enum class CalcEngine : uint8_t {
    compute,
    highway,
    numkong,
};

// Function pointer types matching the existing compute kernel conventions.
using CalcDistFn             = double (*)(const uint8_t*, const uint8_t*, size_t);
using CalcDistWithQueryNormFn = double (*)(const uint8_t*, const uint8_t*, size_t, double);
using CalcSquaredNormFn      = double (*)(const uint8_t*, size_t);
using CalcDotFn              = double (*)(const uint8_t*, const uint8_t*, size_t);

// Holds all resolved function pointers for one (metric, DataType) combination.
// For DOT only `dist` is populated.
// For L2, `dist` is always populated and `dot`/`squared_norm` may also be
// populated so scanners can reuse persisted squared norms.
// For COS all four fields are populated.
struct CalcKernels {
    CalcDistFn              dist = nullptr;
    CalcDistWithQueryNormFn dist_with_query_norm = nullptr;
    CalcSquaredNormFn       squared_norm = nullptr;
    CalcDotFn               dot = nullptr;
};

// Resolves the kernel set for a given engine, metric, and data type.
CalcKernels resolve_calc_kernels(CalcEngine engine, DistFunc func, DataType type);

// Maps the active compute backend family to the corresponding CalcEngine.
CalcEngine selected_calc_engine(ComputeBackendKind kind);

} // namespace sketch2
