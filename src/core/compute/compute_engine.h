// Defines the calc-engine abstraction: an engine selector enum and a struct of
// resolved distance-kernel function pointers. ScannerEx dispatches through
// CalcKernels so the scanning logic is completely decoupled from the concrete
// SIMD library backend.

#pragma once
#include "utils/shared_types.h"
#include <cstdint>

namespace sketch2 {

enum class CalcEngine : uint8_t {
    highway,
    numkong,
};

constexpr CalcEngine compiled_calc_engine() {
#if SKETCH_CALC_ENGINE_HIGHWAY
    return CalcEngine::highway;
#elif SKETCH_CALC_ENGINE_NUMKONG
    return CalcEngine::numkong;
#else
#error "Exactly one calc engine must be compiled."
#endif
}

const char* calc_engine_name(CalcEngine engine);

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

} // namespace sketch2
