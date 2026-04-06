// Declares the Highway-backed kernel resolver.

#pragma once
#include "core/calc/calc_engine.h"

namespace sketch2 {

CalcKernels resolve_hwy_kernels(DistFunc func, DataType type);

} // namespace sketch2
