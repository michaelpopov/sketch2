// Declares the NumKong kernel resolver and backend metadata helpers.

#pragma once

#include "core/compute/compute_engine.h"

namespace sketch2 {

// Runtime kernel resolver kept for benchmarks and kernel-focused tests.
ComputeKernels resolve_nk_kernels(DistFunc func, DataType type);
bool nk_compute_uses_dynamic_dispatch();
uint64_t nk_compute_compiled_capabilities();
uint64_t nk_compute_available_capabilities();
const char* nk_compute_backend_name(DistFunc func, DataType type);
const char* nk_compute_backend_name_for_capabilities(DistFunc func, DataType type, uint64_t capabilities);

} // namespace sketch2
