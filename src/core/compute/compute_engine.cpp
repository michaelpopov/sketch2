#include "core/compute/compute_engine.h"

#if SKETCH_CALC_ENGINE_HIGHWAY
#include "core/compute/scanner_hw.h"
#elif SKETCH_CALC_ENGINE_NUMKONG
#include "core/compute/scanner_nk.h"
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

} // namespace sketch2
