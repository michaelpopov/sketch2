#include "core/calc/calc_engine.h"

#include "core/calc/hwy_kernels.h"
#include "core/calc/nk_kernels.h"

namespace sketch2 {

CalcKernels resolve_calc_kernels(CalcEngine engine, DistFunc func, DataType type) {
    switch (engine) {
        case CalcEngine::highway:
            return resolve_hwy_kernels(func, type);
        case CalcEngine::numkong:
            return resolve_nk_kernels(func, type);
    }
    throw std::runtime_error("resolve_calc_kernels: unsupported engine.");
}

CalcEngine selected_calc_engine(ComputeBackendKind kind) {
    switch (kind) {
        case ComputeBackendKind::highway:
            return CalcEngine::highway;
        case ComputeBackendKind::nk:
            return CalcEngine::numkong;
    }
    throw std::runtime_error("selected_calc_engine: unsupported backend kind.");
}

} // namespace sketch2
