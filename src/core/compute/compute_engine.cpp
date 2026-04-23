#include "core/compute/compute_engine.h"

#include "core/compute/highway.h"

namespace sketch2 {
void initialize_hwy_runtime();
}

namespace sketch2 {

ComputeKernels resolve_compute_kernels(DistFunc func, DataType type) {
    return resolve_hwy_kernels(func, type);
}

void initialize_compute_engine_runtime() {
    initialize_hwy_runtime();
}

} // namespace sketch2
