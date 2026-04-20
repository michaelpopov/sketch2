// Highway-backed scanner and kernel resolver.
// Uses the foreach_target pattern for automatic multi-target compilation
// and runtime dispatch.

#include "core/compute/highway.h"

#include "core/compute/cosine_distance.h"
#include "core/compute/scanner_dataset_scan.h"
#include "core/compute/scanner_heap_utils.h"
#include "core/compute/scanner_log_utils.h"
#include "core/compute/scanner_query_context.h"
#include "core/utils/timer.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "core/compute/highway.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>

HWY_BEFORE_NAMESPACE();
namespace sketch2 {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

template <typename T>
HWY_INLINE const T* AsElements(const uint8_t* ptr) {
    return reinterpret_cast<const T*>(ptr);
}

template <class DF>
HWY_INLINE hn::VFromD<DF> LoadF16AsF32(DF df, const uint8_t* ptr) {
    const hn::Rebind<uint16_t, DF> du16;
    const hn::Rebind<hwy::float16_t, DF> df16;
    const auto u16 = hn::LoadU(du16, AsElements<uint16_t>(ptr));
    return hn::PromoteTo(df, hn::BitCast(df16, u16));
}

template <class DI32>
HWY_INLINE hn::VFromD<DI32> LoadI16AsI32(DI32 di32, const uint8_t* ptr) {
    const hn::Rebind<int16_t, DI32> di16;
    return hn::PromoteTo(di32, hn::LoadU(di16, AsElements<int16_t>(ptr)));
}

double DistL2F32(const uint8_t* a, const uint8_t* b, size_t dim) {
    const float* va = AsElements<float>(a);
    const float* vb = AsElements<float>(b);
    const hn::ScalableTag<float> df;
    const size_t N = hn::Lanes(df);
    auto acc = hn::Zero(df);
    size_t i = 0;
    for (; i + N <= dim; i += N) {
        const auto av = hn::LoadU(df, va + i);
        const auto bv = hn::LoadU(df, vb + i);
        const auto diff = hn::Sub(av, bv);
        acc = hn::MulAdd(diff, diff, acc);
    }
    double sum = hn::ReduceSum(df, acc);
    for (; i < dim; ++i) {
        const double d = static_cast<double>(va[i]) - static_cast<double>(vb[i]);
        sum += d * d;
    }
    return sum;
}

double DistL2F16(const uint8_t* a, const uint8_t* b, size_t dim) {
    const hn::ScalableTag<float> df;
    const size_t N = hn::Lanes(df);
    auto acc = hn::Zero(df);
    size_t i = 0;
    for (; i + N <= dim; i += N) {
        const auto av = LoadF16AsF32(df, a + i * 2);
        const auto bv = LoadF16AsF32(df, b + i * 2);
        const auto diff = hn::Sub(av, bv);
        acc = hn::MulAdd(diff, diff, acc);
    }
    double sum = hn::ReduceSum(df, acc);
    const auto* va = AsElements<hwy::float16_t>(a);
    const auto* vb = AsElements<hwy::float16_t>(b);
    for (; i < dim; ++i) {
        const double d = static_cast<double>(va[i]) - static_cast<double>(vb[i]);
        sum += d * d;
    }
    return sum;
}

double DistL2I16(const uint8_t* a, const uint8_t* b, size_t dim) {
    const int16_t* va = AsElements<int16_t>(a);
    const int16_t* vb = AsElements<int16_t>(b);
    const hn::ScalableTag<int32_t> di32;
    const hn::ScalableTag<float> df;
    const size_t N = hn::Lanes(di32);
    auto acc = hn::Zero(df);
    size_t i = 0;
    for (; i + N <= dim; i += N) {
        const auto av = LoadI16AsI32(di32, a + i * 2);
        const auto bv = LoadI16AsI32(di32, b + i * 2);
        const auto diff = hn::Sub(av, bv);
        const auto diff_f = hn::ConvertTo(df, diff);
        acc = hn::MulAdd(diff_f, diff_f, acc);
    }
    double sum = hn::ReduceSum(df, acc);
    for (; i < dim; ++i) {
        const double d = static_cast<double>(va[i]) - static_cast<double>(vb[i]);
        sum += d * d;
    }
    return sum;
}

double DotF32(const uint8_t* a, const uint8_t* b, size_t dim) {
    const float* va = AsElements<float>(a);
    const float* vb = AsElements<float>(b);
    const hn::ScalableTag<float> df;
    const size_t N = hn::Lanes(df);
    auto acc = hn::Zero(df);
    size_t i = 0;
    for (; i + N <= dim; i += N) {
        const auto av = hn::LoadU(df, va + i);
        const auto bv = hn::LoadU(df, vb + i);
        acc = hn::MulAdd(av, bv, acc);
    }
    double sum = hn::ReduceSum(df, acc);
    for (; i < dim; ++i) {
        sum += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
    }
    return sum;
}

double DotF16(const uint8_t* a, const uint8_t* b, size_t dim) {
    const auto* va = AsElements<hwy::float16_t>(a);
    const auto* vb = AsElements<hwy::float16_t>(b);
    const hn::ScalableTag<float> df;
    const size_t N = hn::Lanes(df);
    auto acc = hn::Zero(df);
    size_t i = 0;
    for (; i + N <= dim; i += N) {
        const auto av = LoadF16AsF32(df, a + i * 2);
        const auto bv = LoadF16AsF32(df, b + i * 2);
        acc = hn::MulAdd(av, bv, acc);
    }
    double sum = hn::ReduceSum(df, acc);
    for (; i < dim; ++i) {
        sum += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
    }
    return sum;
}

double DotI16(const uint8_t* a, const uint8_t* b, size_t dim) {
    const int16_t* va = AsElements<int16_t>(a);
    const int16_t* vb = AsElements<int16_t>(b);
    const hn::ScalableTag<int32_t> di32;
    const hn::ScalableTag<int64_t> di64;
    const size_t N = hn::Lanes(di32);
    auto acc_lo = hn::Zero(di64);
    auto acc_hi = hn::Zero(di64);
    size_t i = 0;
    for (; i + N <= dim; i += N) {
        const auto av = LoadI16AsI32(di32, a + i * 2);
        const auto bv = LoadI16AsI32(di32, b + i * 2);
        acc_lo = hn::WidenMulAccumulate(di64, av, bv, acc_lo, acc_hi);
    }
    int64_t sum = hn::ReduceSum(di64, acc_lo) + hn::ReduceSum(di64, acc_hi);
    for (; i < dim; ++i) {
        sum += static_cast<int64_t>(va[i]) * static_cast<int64_t>(vb[i]);
    }
    return static_cast<double>(sum);
}

double SquaredNormF32(const uint8_t* a, size_t dim) {
    const float* va = AsElements<float>(a);
    const hn::ScalableTag<float> df;
    const size_t N = hn::Lanes(df);
    auto acc = hn::Zero(df);
    size_t i = 0;
    for (; i + N <= dim; i += N) {
        const auto av = hn::LoadU(df, va + i);
        acc = hn::MulAdd(av, av, acc);
    }
    double sum = hn::ReduceSum(df, acc);
    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        sum += ai * ai;
    }
    return sum;
}

double SquaredNormF16(const uint8_t* a, size_t dim) {
    const hn::ScalableTag<float> df;
    const size_t N = hn::Lanes(df);
    auto acc = hn::Zero(df);
    size_t i = 0;
    for (; i + N <= dim; i += N) {
        const auto av = LoadF16AsF32(df, a + i * 2);
        acc = hn::MulAdd(av, av, acc);
    }
    double sum = hn::ReduceSum(df, acc);
    const auto* va = AsElements<hwy::float16_t>(a);
    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        sum += ai * ai;
    }
    return sum;
}

double SquaredNormI16(const uint8_t* a, size_t dim) {
    const int16_t* va = AsElements<int16_t>(a);
    const hn::ScalableTag<int32_t> di32;
    const hn::ScalableTag<float> df;
    const size_t N = hn::Lanes(di32);
    auto acc = hn::Zero(df);
    size_t i = 0;
    for (; i + N <= dim; i += N) {
        const auto av = LoadI16AsI32(di32, a + i * 2);
        const auto av_f = hn::ConvertTo(df, av);
        acc = hn::MulAdd(av_f, av_f, acc);
    }
    double sum = hn::ReduceSum(df, acc);
    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        sum += ai * ai;
    }
    return sum;
}

double DistCosF32(const uint8_t* a, const uint8_t* b, size_t dim) {
    const float* va = AsElements<float>(a);
    const float* vb = AsElements<float>(b);
    const hn::ScalableTag<float> df;
    const size_t N = hn::Lanes(df);
    auto dot_acc = hn::Zero(df);
    auto na_acc = hn::Zero(df);
    auto nb_acc = hn::Zero(df);
    size_t i = 0;
    for (; i + N <= dim; i += N) {
        const auto av = hn::LoadU(df, va + i);
        const auto bv = hn::LoadU(df, vb + i);
        dot_acc = hn::MulAdd(av, bv, dot_acc);
        na_acc = hn::MulAdd(av, av, na_acc);
        nb_acc = hn::MulAdd(bv, bv, nb_acc);
    }
    double dot = hn::ReduceSum(df, dot_acc);
    double norm_a = hn::ReduceSum(df, na_acc);
    double norm_b = hn::ReduceSum(df, nb_acc);
    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        const double bi = static_cast<double>(vb[i]);
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    return finalize_cosine_distance(dot, norm_a, norm_b);
}

double DistCosF16(const uint8_t* a, const uint8_t* b, size_t dim) {
    const hn::ScalableTag<float> df;
    const size_t N = hn::Lanes(df);
    auto dot_acc = hn::Zero(df);
    auto na_acc = hn::Zero(df);
    auto nb_acc = hn::Zero(df);
    size_t i = 0;
    for (; i + N <= dim; i += N) {
        const auto av = LoadF16AsF32(df, a + i * 2);
        const auto bv = LoadF16AsF32(df, b + i * 2);
        dot_acc = hn::MulAdd(av, bv, dot_acc);
        na_acc = hn::MulAdd(av, av, na_acc);
        nb_acc = hn::MulAdd(bv, bv, nb_acc);
    }
    double dot = hn::ReduceSum(df, dot_acc);
    double norm_a = hn::ReduceSum(df, na_acc);
    double norm_b = hn::ReduceSum(df, nb_acc);
    const auto* va = AsElements<hwy::float16_t>(a);
    const auto* vb = AsElements<hwy::float16_t>(b);
    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        const double bi = static_cast<double>(vb[i]);
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    return finalize_cosine_distance(dot, norm_a, norm_b);
}

double DistCosI16(const uint8_t* a, const uint8_t* b, size_t dim) {
    const int16_t* va = AsElements<int16_t>(a);
    const int16_t* vb = AsElements<int16_t>(b);
    const hn::ScalableTag<int32_t> di32;
    const hn::ScalableTag<float> df;
    const size_t N = hn::Lanes(di32);
    auto dot_acc = hn::Zero(df);
    auto na_acc = hn::Zero(df);
    auto nb_acc = hn::Zero(df);
    size_t i = 0;
    for (; i + N <= dim; i += N) {
        const auto av = LoadI16AsI32(di32, a + i * 2);
        const auto bv = LoadI16AsI32(di32, b + i * 2);
        const auto av_f = hn::ConvertTo(df, av);
        const auto bv_f = hn::ConvertTo(df, bv);
        dot_acc = hn::MulAdd(av_f, bv_f, dot_acc);
        na_acc = hn::MulAdd(av_f, av_f, na_acc);
        nb_acc = hn::MulAdd(bv_f, bv_f, nb_acc);
    }
    double dot = hn::ReduceSum(df, dot_acc);
    double norm_a = hn::ReduceSum(df, na_acc);
    double norm_b = hn::ReduceSum(df, nb_acc);
    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        const double bi = static_cast<double>(vb[i]);
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    return finalize_cosine_distance(dot, norm_a, norm_b);
}

double DistCosWithQueryNormF32(const uint8_t* a, const uint8_t* b, size_t dim, double query_norm_sq) {
    const float* va = AsElements<float>(a);
    const float* vb = AsElements<float>(b);
    const hn::ScalableTag<float> df;
    const size_t N = hn::Lanes(df);
    auto dot_acc = hn::Zero(df);
    auto na_acc = hn::Zero(df);
    size_t i = 0;
    for (; i + N <= dim; i += N) {
        const auto av = hn::LoadU(df, va + i);
        const auto bv = hn::LoadU(df, vb + i);
        dot_acc = hn::MulAdd(av, bv, dot_acc);
        na_acc = hn::MulAdd(av, av, na_acc);
    }
    double dot = hn::ReduceSum(df, dot_acc);
    double norm_a = hn::ReduceSum(df, na_acc);
    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        const double bi = static_cast<double>(vb[i]);
        dot += ai * bi;
        norm_a += ai * ai;
    }
    return finalize_cosine_distance(dot, norm_a, query_norm_sq);
}

double DistCosWithQueryNormF16(const uint8_t* a, const uint8_t* b, size_t dim, double query_norm_sq) {
    const hn::ScalableTag<float> df;
    const size_t N = hn::Lanes(df);
    auto dot_acc = hn::Zero(df);
    auto na_acc = hn::Zero(df);
    size_t i = 0;
    for (; i + N <= dim; i += N) {
        const auto av = LoadF16AsF32(df, a + i * 2);
        const auto bv = LoadF16AsF32(df, b + i * 2);
        dot_acc = hn::MulAdd(av, bv, dot_acc);
        na_acc = hn::MulAdd(av, av, na_acc);
    }
    double dot = hn::ReduceSum(df, dot_acc);
    double norm_a = hn::ReduceSum(df, na_acc);
    const auto* va = AsElements<hwy::float16_t>(a);
    const auto* vb = AsElements<hwy::float16_t>(b);
    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        const double bi = static_cast<double>(vb[i]);
        dot += ai * bi;
        norm_a += ai * ai;
    }
    return finalize_cosine_distance(dot, norm_a, query_norm_sq);
}

double DistCosWithQueryNormI16(const uint8_t* a, const uint8_t* b, size_t dim, double query_norm_sq) {
    const int16_t* va = AsElements<int16_t>(a);
    const int16_t* vb = AsElements<int16_t>(b);
    const hn::ScalableTag<int32_t> di32;
    const hn::ScalableTag<float> df;
    const size_t N = hn::Lanes(di32);
    auto dot_acc = hn::Zero(df);
    auto na_acc = hn::Zero(df);
    size_t i = 0;
    for (; i + N <= dim; i += N) {
        const auto av = LoadI16AsI32(di32, a + i * 2);
        const auto bv = LoadI16AsI32(di32, b + i * 2);
        const auto av_f = hn::ConvertTo(df, av);
        const auto bv_f = hn::ConvertTo(df, bv);
        dot_acc = hn::MulAdd(av_f, bv_f, dot_acc);
        na_acc = hn::MulAdd(av_f, av_f, na_acc);
    }
    double dot = hn::ReduceSum(df, dot_acc);
    double norm_a = hn::ReduceSum(df, na_acc);
    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        const double bi = static_cast<double>(vb[i]);
        dot += ai * bi;
        norm_a += ai * ai;
    }
    return finalize_cosine_distance(dot, norm_a, query_norm_sq);
}

}  // namespace HWY_NAMESPACE
}  // namespace sketch2
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace sketch2 {

namespace {

HWY_EXPORT(DistL2F32);
HWY_EXPORT(DistL2F16);
HWY_EXPORT(DistL2I16);
HWY_EXPORT(DotF32);
HWY_EXPORT(DotF16);
HWY_EXPORT(DotI16);
HWY_EXPORT(SquaredNormF32);
HWY_EXPORT(SquaredNormF16);
HWY_EXPORT(SquaredNormI16);
HWY_EXPORT(DistCosF32);
HWY_EXPORT(DistCosF16);
HWY_EXPORT(DistCosI16);
HWY_EXPORT(DistCosWithQueryNormF32);
HWY_EXPORT(DistCosWithQueryNormF16);
HWY_EXPORT(DistCosWithQueryNormI16);

double hwy_dist_l2_f32(const uint8_t* a, const uint8_t* b, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(DistL2F32)(a, b, dim);
}
double hwy_dist_l2_f16(const uint8_t* a, const uint8_t* b, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(DistL2F16)(a, b, dim);
}
double hwy_dist_l2_i16(const uint8_t* a, const uint8_t* b, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(DistL2I16)(a, b, dim);
}

double hwy_dot_f32(const uint8_t* a, const uint8_t* b, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(DotF32)(a, b, dim);
}
double hwy_dot_f16(const uint8_t* a, const uint8_t* b, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(DotF16)(a, b, dim);
}
double hwy_dot_i16(const uint8_t* a, const uint8_t* b, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(DotI16)(a, b, dim);
}

double hwy_squared_norm_f32(const uint8_t* a, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(SquaredNormF32)(a, dim);
}
double hwy_squared_norm_f16(const uint8_t* a, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(SquaredNormF16)(a, dim);
}
double hwy_squared_norm_i16(const uint8_t* a, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(SquaredNormI16)(a, dim);
}

double hwy_dist_cos_f32(const uint8_t* a, const uint8_t* b, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(DistCosF32)(a, b, dim);
}
double hwy_dist_cos_f16(const uint8_t* a, const uint8_t* b, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(DistCosF16)(a, b, dim);
}
double hwy_dist_cos_i16(const uint8_t* a, const uint8_t* b, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(DistCosI16)(a, b, dim);
}

double hwy_dist_cos_qn_f32(const uint8_t* a, const uint8_t* b, size_t dim, double qn) {
    return HWY_DYNAMIC_DISPATCH(DistCosWithQueryNormF32)(a, b, dim, qn);
}
double hwy_dist_cos_qn_f16(const uint8_t* a, const uint8_t* b, size_t dim, double qn) {
    return HWY_DYNAMIC_DISPATCH(DistCosWithQueryNormF16)(a, b, dim, qn);
}
double hwy_dist_cos_qn_i16(const uint8_t* a, const uint8_t* b, size_t dim, double qn) {
    return HWY_DYNAMIC_DISPATCH(DistCosWithQueryNormI16)(a, b, dim, qn);
}

Ret validate_hwy_kernel_support(DistFunc func, DataType type) {
    try {
        const ComputeKernels kernels = resolve_hwy_kernels(func, type);
        if (kernels.dist == nullptr) {
            return Ret(std::string("Highway::find_items: missing dist kernel for ")
                + dist_func_to_string(func) + "/" + data_type_to_string(type) + ".");
        }
        if ((func == DistFunc::L2 || func == DistFunc::COS)
                && (kernels.dot == nullptr || kernels.squared_norm == nullptr)) {
            return Ret(std::string("Highway::find_items: missing stored-norm helpers for ")
                + dist_func_to_string(func) + "/" + data_type_to_string(type) + ".");
        }
        if (func == DistFunc::COS && kernels.dist_with_query_norm == nullptr) {
            return Ret(std::string("Highway::find_items: missing query-norm kernel for ")
                + dist_func_to_string(func) + "/" + data_type_to_string(type) + ".");
        }
        return Ret(0);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

#define HWY_FOR_EACH_DOT_KERNEL(X) \
    X(f32, hwy_dot_f32) \
    X(f16, hwy_dot_f16) \
    X(i16, hwy_dot_i16)

#define HWY_FOR_EACH_L2_KERNEL(X) \
    X(f32, hwy_dot_f32, hwy_dist_l2_f32, hwy_squared_norm_f32) \
    X(f16, hwy_dot_f16, hwy_dist_l2_f16, hwy_squared_norm_f16) \
    X(i16, hwy_dot_i16, hwy_dist_l2_i16, hwy_squared_norm_i16)

#define HWY_FOR_EACH_COS_KERNEL(X) \
    X(f32, hwy_dot_f32, hwy_dist_cos_f32, hwy_dist_cos_qn_f32, hwy_squared_norm_f32) \
    X(f16, hwy_dot_f16, hwy_dist_cos_f16, hwy_dist_cos_qn_f16, hwy_squared_norm_f16) \
    X(i16, hwy_dot_i16, hwy_dist_cos_i16, hwy_dist_cos_qn_i16, hwy_squared_norm_i16)

#define HWY_DISPATCH_COS_CASE(TYPE_TAG, DOT_KERNEL, DIST_KERNEL, DIST_QN_KERNEL, SQ_NORM_KERNEL) \
    case DataType::TYPE_TAG: { \
        const double query_norm_sq = SQ_NORM_KERNEL(vec, dim); \
        log_query_branch(query_id, "cos_with_optional_norms", \
            query_norm_sq, query_norm_sq == 0.0); \
        const QueryCosContext query{ \
            vec, dim, query_norm_sq, query_inverse_norm(query_norm_sq)}; \
        CHECK((scan_dataset_heap_with_optional_cosine_norms< \
            DOT_KERNEL, DIST_QN_KERNEL>( \
            query_id, dataset, count, &heap, query, func, bitset))); \
        break; \
    }

#define HWY_DISPATCH_DOT_CASE(TYPE_TAG, DOT_KERNEL) \
    case DataType::TYPE_TAG: \
        CHECK(scan_dataset_heap_with_dot<DOT_KERNEL>( \
            query_id, dataset, count, &heap, query, func, bitset)); \
        break;

#define HWY_DISPATCH_L2_CASE(TYPE_TAG, DOT_KERNEL, DIST_KERNEL, SQ_NORM_KERNEL) \
    case DataType::TYPE_TAG: { \
        const double query_norm_sq = SQ_NORM_KERNEL(vec, dim); \
        log_query_branch(query_id, "l2_with_optional_norms", \
            query_norm_sq, query_norm_sq == 0.0); \
        const QueryL2Context query{vec, dim, query_norm_sq}; \
        CHECK((scan_dataset_heap_with_optional_l2_norms< \
            DOT_KERNEL, DIST_KERNEL>( \
            query_id, dataset, count, &heap, query, func, bitset))); \
        break; \
    }

} // namespace

Ret find_items_hw(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>* result, const BitsetFilter* bitset, uint64_t query_id) {
    if (vec == nullptr || count == 0 || result == nullptr) {
        return Ret("Highway::find_items: invalid arguments.");
    }

    result->clear();
    const DistFunc func = dataset.dist_func();
    const size_t dim = dataset.dim();
    const DataType type = dataset.type();
    CHECK(validate_hwy_kernel_support(func, type));
    if (query_id == 0) {
        query_id = next_scanner_query_id();
    }
    log_query_start(query_id, dataset.name(), func, type, dim, count,
        ComputeEngine::highway, bitset != nullptr);

    DistHeap heap(DistItemCompare{func});
    heap.reserve(count);
    Timer timer("highway::query");

    if (func == DistFunc::COS) {
        switch (type) {
            HWY_FOR_EACH_COS_KERNEL(HWY_DISPATCH_COS_CASE);
            default:
                return Ret("Highway::find_items: unsupported DataType for COS.");
        }
    } else if (func == DistFunc::DOT) {
        log_query_branch(query_id, "dot");
        const QueryDotContext query{vec, dim};
        switch (type) {
            HWY_FOR_EACH_DOT_KERNEL(HWY_DISPATCH_DOT_CASE);
            default:
                return Ret("Highway::find_items: unsupported DataType for DOT.");
        }
    } else if (func == DistFunc::L2) {
        switch (type) {
            HWY_FOR_EACH_L2_KERNEL(HWY_DISPATCH_L2_CASE);
            default:
                return Ret("Highway::find_items: unsupported DataType for L2.");
        }
    } else {
        return Ret("Highway::find_items: unsupported DistFunc.");
    }

    extract_items(&heap, result);
    log_query_finish(query_id, dataset.name(), func, type, dim, count,
        ComputeEngine::highway, timer.elapsed_ms(), *result);
    return Ret(0);
}

ComputeKernels resolve_hwy_kernels(DistFunc func, DataType type) {
    ComputeKernels k;
    switch (func) {
        case DistFunc::DOT:
            switch (type) {
#define HWY_RESOLVE_DOT_CASE(TYPE_TAG, DOT_KERNEL) \
                case DataType::TYPE_TAG: k.dist = &DOT_KERNEL; k.dot = &DOT_KERNEL; break;
                HWY_FOR_EACH_DOT_KERNEL(HWY_RESOLVE_DOT_CASE)
#undef HWY_RESOLVE_DOT_CASE
                default:
                    throw std::runtime_error("resolve_hwy_kernels: unsupported DataType for DOT.");
            }
            break;
        case DistFunc::L2:
            switch (type) {
#define HWY_RESOLVE_L2_CASE(TYPE_TAG, DOT_KERNEL, DIST_KERNEL, SQ_NORM_KERNEL) \
                case DataType::TYPE_TAG: \
                    k.dist = &DIST_KERNEL; \
                    k.squared_norm = &SQ_NORM_KERNEL; \
                    k.dot = &DOT_KERNEL; \
                    break;
                HWY_FOR_EACH_L2_KERNEL(HWY_RESOLVE_L2_CASE)
#undef HWY_RESOLVE_L2_CASE
                default:
                    throw std::runtime_error("resolve_hwy_kernels: unsupported DataType for L2.");
            }
            break;
        case DistFunc::COS:
            switch (type) {
#define HWY_RESOLVE_COS_CASE(TYPE_TAG, DOT_KERNEL, DIST_KERNEL, DIST_QN_KERNEL, SQ_NORM_KERNEL) \
                case DataType::TYPE_TAG: \
                    k.dist = &DIST_KERNEL; \
                    k.dist_with_query_norm = &DIST_QN_KERNEL; \
                    k.squared_norm = &SQ_NORM_KERNEL; \
                    k.dot = &DOT_KERNEL; \
                    break;
                HWY_FOR_EACH_COS_KERNEL(HWY_RESOLVE_COS_CASE)
#undef HWY_RESOLVE_COS_CASE
                default:
                    throw std::runtime_error("resolve_hwy_kernels: unsupported DataType for COS.");
            }
            break;
        default:
            throw std::runtime_error("resolve_hwy_kernels: unsupported DistFunc.");
    }
    return k;
}

#undef HWY_DISPATCH_COS_CASE
#undef HWY_DISPATCH_DOT_CASE
#undef HWY_DISPATCH_L2_CASE
#undef HWY_FOR_EACH_DOT_KERNEL
#undef HWY_FOR_EACH_L2_KERNEL
#undef HWY_FOR_EACH_COS_KERNEL

}  // namespace sketch2

#endif  // HWY_ONCE
