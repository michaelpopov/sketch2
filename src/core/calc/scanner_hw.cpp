// Highway-backed scanner and kernel resolver.
// Uses the foreach_target pattern for automatic multi-target compilation
// and runtime dispatch.

#include "core/calc/scanner_hw.h"

#include "core/calc/cosine_distance.h"
#include "core/calc/scanner_dataset_scan.h"
#include "core/calc/scanner_heap_utils.h"
#include "core/calc/scanner_log_utils.h"
#include "core/calc/scanner_query_context.h"
#include "core/utils/timer.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "core/calc/scanner_hw.cpp"
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

Ret find_items_hw_impl(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>* result, const BitsetFilter* bitset) {
    if (vec == nullptr || count == 0 || result == nullptr) {
        return Ret("ScannerHw::find_items: invalid arguments.");
    }

    result->clear();
    const DistFunc func = dataset.dist_func();
    const size_t dim = dataset.dim();
    const CalcKernels kernels = resolve_hwy_kernels(func, dataset.type());

    DistHeap heap(DistItemCompare{func});
    heap.reserve(count);
    Timer timer("scanner_hw::query");

    if (func == DistFunc::COS) {
        assert(kernels.squared_norm && kernels.dot && kernels.dist_with_query_norm);
        const double query_norm_sq = kernels.squared_norm(vec, dim);
        const QueryCosContext query{vec, dim, query_norm_sq, query_inverse_norm(query_norm_sq)};
        CHECK(scan_dataset_heap_with_optional_cosine_norms(
            dataset, count, &heap, kernels.dot, kernels.dist_with_query_norm,
            query, func, bitset));
    } else if (func == DistFunc::L2 && kernels.squared_norm && kernels.dot) {
        assert(kernels.dist);
        const QueryL2Context query{vec, dim, kernels.squared_norm(vec, dim)};
        CHECK(scan_dataset_heap_with_optional_l2_norms(
            dataset, count, &heap, kernels.dot, kernels.dist, query, func, bitset));
    } else {
        assert(kernels.dist);
        const QueryDistContext query{vec, dim};
        CHECK(scan_dataset_heap_with_dist(
            dataset, count, &heap, kernels.dist, query, func, bitset));
    }

    log_query(dataset.name(), func, dataset.type(), dim, count, CalcEngine::highway, timer.elapsed_ms());
    extract_items(&heap, result);
    return Ret(0);
}

} // namespace

Ret find_items_hw(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>* result, const BitsetFilter* bitset) {
    return find_items_hw_impl(dataset, count, vec, result, bitset);
}

CalcKernels resolve_hwy_kernels(DistFunc func, DataType type) {
    CalcKernels k;
    switch (func) {
        case DistFunc::DOT:
            switch (type) {
                case DataType::f32: k.dist = &hwy_dot_f32; break;
                case DataType::f16: k.dist = &hwy_dot_f16; break;
                case DataType::i16: k.dist = &hwy_dot_i16; break;
                default:
                    throw std::runtime_error("resolve_hwy_kernels: unsupported DataType for DOT.");
            }
            break;
        case DistFunc::L2:
            switch (type) {
                case DataType::f32:
                    k.dist = &hwy_dist_l2_f32;
                    k.squared_norm = &hwy_squared_norm_f32;
                    k.dot = &hwy_dot_f32;
                    break;
                case DataType::f16:
                    k.dist = &hwy_dist_l2_f16;
                    k.squared_norm = &hwy_squared_norm_f16;
                    k.dot = &hwy_dot_f16;
                    break;
                case DataType::i16:
                    k.dist = &hwy_dist_l2_i16;
                    k.squared_norm = &hwy_squared_norm_i16;
                    k.dot = &hwy_dot_i16;
                    break;
                default:
                    throw std::runtime_error("resolve_hwy_kernels: unsupported DataType for L2.");
            }
            break;
        case DistFunc::COS:
            switch (type) {
                case DataType::f32:
                    k.dist = &hwy_dist_cos_f32;
                    k.dist_with_query_norm = &hwy_dist_cos_qn_f32;
                    k.squared_norm = &hwy_squared_norm_f32;
                    k.dot = &hwy_dot_f32;
                    break;
                case DataType::f16:
                    k.dist = &hwy_dist_cos_f16;
                    k.dist_with_query_norm = &hwy_dist_cos_qn_f16;
                    k.squared_norm = &hwy_squared_norm_f16;
                    k.dot = &hwy_dot_f16;
                    break;
                case DataType::i16:
                    k.dist = &hwy_dist_cos_i16;
                    k.dist_with_query_norm = &hwy_dist_cos_qn_i16;
                    k.squared_norm = &hwy_squared_norm_i16;
                    k.dot = &hwy_dot_i16;
                    break;
                default:
                    throw std::runtime_error("resolve_hwy_kernels: unsupported DataType for COS.");
            }
            break;
        default:
            throw std::runtime_error("resolve_hwy_kernels: unsupported DistFunc.");
    }
    return k;
}

Ret ScannerHw::find(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) const {
    try {
        std::vector<DistItem> items;
        CHECK(find_items_hw(dataset, count, vec, &items, nullptr));
        extract_ids_from_items(items, &result);
        return Ret(0);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret ScannerHw::find_items(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result, const BitsetFilter* bitset) const {
    try {
        return find_items_hw(dataset, count, vec, &result, bitset);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

}  // namespace sketch2

#endif  // HWY_ONCE
