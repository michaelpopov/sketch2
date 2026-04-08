#pragma once

#include "core/calc/calc_engine.h"

namespace sketch2 {

CalcKernels resolve_compute_kernels(DistFunc func, DataType type);

} // namespace sketch2
