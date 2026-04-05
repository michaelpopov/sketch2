// Declares the NumKong-backed kernel resolver.

#pragma once
#include "core/calc/calc_engine.h"

namespace sketch2 {

CalcKernels resolve_nk_kernels(DistFunc func, DataType type);
bool nk_calc_uses_dynamic_dispatch();
uint64_t nk_calc_compiled_capabilities();
uint64_t nk_calc_available_capabilities();
const char* nk_calc_backend_name(DistFunc func, DataType type);
const char* nk_calc_backend_name_for_capabilities(DistFunc func, DataType type, uint64_t capabilities);

} // namespace sketch2
