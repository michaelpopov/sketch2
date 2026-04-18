#include "core/calc/calc_engine.h"

#if SKETCH_CALC_ENGINE_HIGHWAY
#include "core/calc/hwy_kernels.h"
#elif SKETCH_CALC_ENGINE_NUMKONG
#include "core/calc/nk_kernels.h"
#endif

namespace sketch2 {

CalcKernels resolve_calc_kernels(CalcEngine engine, DistFunc func, DataType type) {
    switch (engine) {
#if SKETCH_CALC_ENGINE_HIGHWAY
        case CalcEngine::highway:
            return resolve_hwy_kernels(func, type);
#endif
#if SKETCH_CALC_ENGINE_NUMKONG
        case CalcEngine::numkong:
            return resolve_nk_kernels(func, type);
#endif
        default:
            break;
    }
    throw std::runtime_error("resolve_calc_kernels: unsupported engine.");
}

CalcEngine selected_calc_engine(ComputeBackendKind kind) {
    if (kind != compiled_compute_backend_kind()) {
        throw std::runtime_error("selected_calc_engine: backend is not available in this build.");
    }
    return compiled_calc_engine();
}

} // namespace sketch2
