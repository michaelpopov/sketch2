#include "core/calc/compute_kernels.h"

#include <stdexcept>

#include "core/compute/compute_cos.h"
#include "core/compute/compute_dot.h"
#include "core/compute/compute_l2.h"

namespace sketch2 {

CalcKernels resolve_compute_kernels(DistFunc func, DataType type) {
    CalcKernels k;
    switch (func) {
        case DistFunc::DOT:
            k.dist = ComputeDOT::resolve_dist(type);
            break;
        case DistFunc::L2:
            k.dist = ComputeL2::resolve_dist(type);
            k.squared_norm = ComputeDotNorm::resolve_squared_norm(type);
            k.dot = ComputeDotNorm::resolve_dot(type);
            break;
        case DistFunc::COS:
            k.dist = ComputeCos::resolve_dist(type);
            k.dist_with_query_norm = ComputeCos::resolve_dist_with_query_norm(type);
            k.squared_norm = ComputeDotNorm::resolve_squared_norm(type);
            k.dot = ComputeDotNorm::resolve_dot(type);
            break;
        default:
            throw std::runtime_error("resolve_compute_kernels: unsupported DistFunc.");
    }
    return k;
}

} // namespace sketch2
