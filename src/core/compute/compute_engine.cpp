#include "core/compute/compute_engine.h"

#if SKETCH_CALC_ENGINE_HIGHWAY
#include "core/compute/scanner_hw.h"
#elif SKETCH_CALC_ENGINE_NUMKONG
#include "core/compute/scanner_nk.h"
#endif

namespace sketch2 {

ComputeKernels resolve_compute_kernels(ComputeEngine engine, DistFunc func, DataType type) {
    switch (engine) {
#if SKETCH_CALC_ENGINE_HIGHWAY
        case ComputeEngine::highway:
            return resolve_hwy_kernels(func, type);
#endif
#if SKETCH_CALC_ENGINE_NUMKONG
        case ComputeEngine::numkong:
            return resolve_nk_kernels(func, type);
#endif
        default:
            break;
    }
    throw std::runtime_error("resolve_compute_kernels: unsupported engine.");
}

} // namespace sketch2
