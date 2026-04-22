#include "core/compute/compute_engine.h"

#if SKETCH2_COMPUTE_ENGINE_HIGHWAY
#include "core/compute/highway.h"

namespace sketch2 {
void initialize_hwy_runtime();
}
#elif SKETCH2_COMPUTE_ENGINE_NUMKONG
#include "core/compute/numkong.h"
#endif

namespace sketch2 {

ComputeKernels resolve_compute_kernels(DistFunc func, DataType type) {
#if SKETCH2_COMPUTE_ENGINE_HIGHWAY
    return resolve_hwy_kernels(func, type);
#elif SKETCH2_COMPUTE_ENGINE_NUMKONG
    return resolve_nk_kernels(func, type);
#else
#error "Exactly one compute engine must be compiled."
#endif
}

void initialize_compute_engine_runtime() {
#if SKETCH2_COMPUTE_ENGINE_HIGHWAY
    initialize_hwy_runtime();
#elif SKETCH2_COMPUTE_ENGINE_NUMKONG
    // NumKong resolves capability-specific kernels per thread, so there is no
    // comparable process-wide warm-up step here.
#endif
}

} // namespace sketch2
