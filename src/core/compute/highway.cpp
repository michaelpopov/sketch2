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

// Per-reader scanners. Each compiles once per Highway target and threads the
// kernel through the existing scan-loop templates as a compile-time parameter,
// so the kernel inlines into the record loop. Runtime target dispatch then
// happens once per reader instead of once per candidate.

void ScanDotF32(const DataReader& reader, size_t count, DistHeapEx* heap,
        const QueryDotContext& query, const BitsetFilter* bitset) {
    scan_data_reader_with_dot<DotF32>(reader, count, heap, query, bitset);
}
void ScanDotF16(const DataReader& reader, size_t count, DistHeapEx* heap,
        const QueryDotContext& query, const BitsetFilter* bitset) {
    scan_data_reader_with_dot<DotF16>(reader, count, heap, query, bitset);
}
void ScanDotI16(const DataReader& reader, size_t count, DistHeapEx* heap,
        const QueryDotContext& query, const BitsetFilter* bitset) {
    scan_data_reader_with_dot<DotI16>(reader, count, heap, query, bitset);
}

void ScanL2F32Stored(const DataReader& reader, size_t count, DistHeapEx* heap,
        const QueryL2Context& query, const BitsetFilter* bitset) {
    scan_data_reader_with_l2_stored_norms<DotF32>(reader, count, heap, query, bitset);
}
void ScanL2F16Stored(const DataReader& reader, size_t count, DistHeapEx* heap,
        const QueryL2Context& query, const BitsetFilter* bitset) {
    scan_data_reader_with_l2_stored_norms<DotF16>(reader, count, heap, query, bitset);
}
void ScanL2I16Stored(const DataReader& reader, size_t count, DistHeapEx* heap,
        const QueryL2Context& query, const BitsetFilter* bitset) {
    scan_data_reader_with_l2_stored_norms<DotI16>(reader, count, heap, query, bitset);
}

void ScanL2F32Fallback(const DataReader& reader, size_t count, DistHeapEx* heap,
        const QueryDistContext& query, const BitsetFilter* bitset) {
    scan_data_reader_with_dist<DistL2F32>(reader, count, heap, query, bitset);
}
void ScanL2F16Fallback(const DataReader& reader, size_t count, DistHeapEx* heap,
        const QueryDistContext& query, const BitsetFilter* bitset) {
    scan_data_reader_with_dist<DistL2F16>(reader, count, heap, query, bitset);
}
void ScanL2I16Fallback(const DataReader& reader, size_t count, DistHeapEx* heap,
        const QueryDistContext& query, const BitsetFilter* bitset) {
    scan_data_reader_with_dist<DistL2I16>(reader, count, heap, query, bitset);
}

void ScanCosF32Stored(const DataReader& reader, size_t count, DistHeapEx* heap,
        const QueryCosContext& query, const BitsetFilter* bitset) {
    scan_data_reader_with_cos_stored_norms<DotF32>(reader, count, heap, query, bitset);
}
void ScanCosF16Stored(const DataReader& reader, size_t count, DistHeapEx* heap,
        const QueryCosContext& query, const BitsetFilter* bitset) {
    scan_data_reader_with_cos_stored_norms<DotF16>(reader, count, heap, query, bitset);
}
void ScanCosI16Stored(const DataReader& reader, size_t count, DistHeapEx* heap,
        const QueryCosContext& query, const BitsetFilter* bitset) {
    scan_data_reader_with_cos_stored_norms<DotI16>(reader, count, heap, query, bitset);
}

void ScanCosF32QueryNorm(const DataReader& reader, size_t count, DistHeapEx* heap,
        const QueryCosContext& query, const BitsetFilter* bitset) {
    scan_data_reader_with_query_norm<DistCosWithQueryNormF32>(reader, count, heap, query, bitset);
}
void ScanCosF16QueryNorm(const DataReader& reader, size_t count, DistHeapEx* heap,
        const QueryCosContext& query, const BitsetFilter* bitset) {
    scan_data_reader_with_query_norm<DistCosWithQueryNormF16>(reader, count, heap, query, bitset);
}
void ScanCosI16QueryNorm(const DataReader& reader, size_t count, DistHeapEx* heap,
        const QueryCosContext& query, const BitsetFilter* bitset) {
    scan_data_reader_with_query_norm<DistCosWithQueryNormI16>(reader, count, heap, query, bitset);
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

HWY_EXPORT(ScanDotF32);
HWY_EXPORT(ScanDotF16);
HWY_EXPORT(ScanDotI16);
HWY_EXPORT(ScanL2F32Stored);
HWY_EXPORT(ScanL2F16Stored);
HWY_EXPORT(ScanL2I16Stored);
HWY_EXPORT(ScanL2F32Fallback);
HWY_EXPORT(ScanL2F16Fallback);
HWY_EXPORT(ScanL2I16Fallback);
HWY_EXPORT(ScanCosF32Stored);
HWY_EXPORT(ScanCosF16Stored);
HWY_EXPORT(ScanCosI16Stored);
HWY_EXPORT(ScanCosF32QueryNorm);
HWY_EXPORT(ScanCosF16QueryNorm);
HWY_EXPORT(ScanCosI16QueryNorm);

using ScanDotFn = void (*)(const DataReader&, size_t, DistHeapEx*,
    const QueryDotContext&, const BitsetFilter*);
using ScanL2StoredFn = void (*)(const DataReader&, size_t, DistHeapEx*,
    const QueryL2Context&, const BitsetFilter*);
using ScanDistFn = void (*)(const DataReader&, size_t, DistHeapEx*,
    const QueryDistContext&, const BitsetFilter*);
using ScanCosFn = void (*)(const DataReader&, size_t, DistHeapEx*,
    const QueryCosContext&, const BitsetFilter*);

struct HighwayDotScan {
    ScanDotFn scan = nullptr;
};
struct HighwayL2Scan {
    ScanL2StoredFn stored = nullptr;
    ScanDistFn fallback = nullptr;
};
struct HighwayCosScan {
    ScanCosFn stored = nullptr;
    ScanCosFn fallback = nullptr;
};

struct HighwayKernelCache {
    ComputeKernels dot_f32;
    ComputeKernels dot_f16;
    ComputeKernels dot_i16;
    ComputeKernels l2_f32;
    ComputeKernels l2_f16;
    ComputeKernels l2_i16;
    ComputeKernels cos_f32;
    ComputeKernels cos_f16;
    ComputeKernels cos_i16;

    HighwayDotScan scan_dot_f32;
    HighwayDotScan scan_dot_f16;
    HighwayDotScan scan_dot_i16;
    HighwayL2Scan scan_l2_f32;
    HighwayL2Scan scan_l2_f16;
    HighwayL2Scan scan_l2_i16;
    HighwayCosScan scan_cos_f32;
    HighwayCosScan scan_cos_f16;
    HighwayCosScan scan_cos_i16;
};

void warm_hwy_kernel_cache(HighwayKernelCache* cache) {
    assert(cache != nullptr);
    hwy::GetChosenTarget().Update(hwy::SupportedTargets());

    const ComputeDotFn dot_f32 = HWY_DYNAMIC_POINTER(DotF32);
    const ComputeDotFn dot_f16 = HWY_DYNAMIC_POINTER(DotF16);
    const ComputeDotFn dot_i16 = HWY_DYNAMIC_POINTER(DotI16);
    const ComputeDistFn l2_f32 = HWY_DYNAMIC_POINTER(DistL2F32);
    const ComputeDistFn l2_f16 = HWY_DYNAMIC_POINTER(DistL2F16);
    const ComputeDistFn l2_i16 = HWY_DYNAMIC_POINTER(DistL2I16);
    const ComputeSquaredNormFn sq_f32 = HWY_DYNAMIC_POINTER(SquaredNormF32);
    const ComputeSquaredNormFn sq_f16 = HWY_DYNAMIC_POINTER(SquaredNormF16);
    const ComputeSquaredNormFn sq_i16 = HWY_DYNAMIC_POINTER(SquaredNormI16);
    const ComputeDistFn cos_f32 = HWY_DYNAMIC_POINTER(DistCosF32);
    const ComputeDistFn cos_f16 = HWY_DYNAMIC_POINTER(DistCosF16);
    const ComputeDistFn cos_i16 = HWY_DYNAMIC_POINTER(DistCosI16);
    const ComputeDistWithQueryNormFn cos_qn_f32 =
        HWY_DYNAMIC_POINTER(DistCosWithQueryNormF32);
    const ComputeDistWithQueryNormFn cos_qn_f16 =
        HWY_DYNAMIC_POINTER(DistCosWithQueryNormF16);
    const ComputeDistWithQueryNormFn cos_qn_i16 =
        HWY_DYNAMIC_POINTER(DistCosWithQueryNormI16);

    const auto set_dot_kernels = [](ComputeKernels* kernels, ComputeDotFn dot_fn) {
        kernels->dist = dot_fn;
        kernels->dot = dot_fn;
    };
    const auto set_l2_kernels = [](
            ComputeKernels* kernels, ComputeDistFn dist_fn, ComputeSquaredNormFn squared_norm_fn,
            ComputeDotFn dot_fn) {
        kernels->dist = dist_fn;
        kernels->squared_norm = squared_norm_fn;
        kernels->dot = dot_fn;
    };
    const auto set_cos_kernels = [](
            ComputeKernels* kernels, ComputeDistFn dist_fn,
            ComputeDistWithQueryNormFn dist_with_query_norm_fn,
            ComputeSquaredNormFn squared_norm_fn, ComputeDotFn dot_fn) {
        kernels->dist = dist_fn;
        kernels->dist_with_query_norm = dist_with_query_norm_fn;
        kernels->squared_norm = squared_norm_fn;
        kernels->dot = dot_fn;
    };

    set_dot_kernels(&cache->dot_f32, dot_f32);
    set_dot_kernels(&cache->dot_f16, dot_f16);
    set_dot_kernels(&cache->dot_i16, dot_i16);

    set_l2_kernels(&cache->l2_f32, l2_f32, sq_f32, dot_f32);
    set_l2_kernels(&cache->l2_f16, l2_f16, sq_f16, dot_f16);
    set_l2_kernels(&cache->l2_i16, l2_i16, sq_i16, dot_i16);

    set_cos_kernels(&cache->cos_f32, cos_f32, cos_qn_f32, sq_f32, dot_f32);
    set_cos_kernels(&cache->cos_f16, cos_f16, cos_qn_f16, sq_f16, dot_f16);
    set_cos_kernels(&cache->cos_i16, cos_i16, cos_qn_i16, sq_i16, dot_i16);

    cache->scan_dot_f32.scan = HWY_DYNAMIC_POINTER(ScanDotF32);
    cache->scan_dot_f16.scan = HWY_DYNAMIC_POINTER(ScanDotF16);
    cache->scan_dot_i16.scan = HWY_DYNAMIC_POINTER(ScanDotI16);

    cache->scan_l2_f32.stored = HWY_DYNAMIC_POINTER(ScanL2F32Stored);
    cache->scan_l2_f16.stored = HWY_DYNAMIC_POINTER(ScanL2F16Stored);
    cache->scan_l2_i16.stored = HWY_DYNAMIC_POINTER(ScanL2I16Stored);
    cache->scan_l2_f32.fallback = HWY_DYNAMIC_POINTER(ScanL2F32Fallback);
    cache->scan_l2_f16.fallback = HWY_DYNAMIC_POINTER(ScanL2F16Fallback);
    cache->scan_l2_i16.fallback = HWY_DYNAMIC_POINTER(ScanL2I16Fallback);

    cache->scan_cos_f32.stored = HWY_DYNAMIC_POINTER(ScanCosF32Stored);
    cache->scan_cos_f16.stored = HWY_DYNAMIC_POINTER(ScanCosF16Stored);
    cache->scan_cos_i16.stored = HWY_DYNAMIC_POINTER(ScanCosI16Stored);
    cache->scan_cos_f32.fallback = HWY_DYNAMIC_POINTER(ScanCosF32QueryNorm);
    cache->scan_cos_f16.fallback = HWY_DYNAMIC_POINTER(ScanCosF16QueryNorm);
    cache->scan_cos_i16.fallback = HWY_DYNAMIC_POINTER(ScanCosI16QueryNorm);
}

const HighwayKernelCache& hwy_kernel_cache() {
    static const HighwayKernelCache cache = []() {
        HighwayKernelCache resolved;
        warm_hwy_kernel_cache(&resolved);
        return resolved;
    }();
    return cache;
}

const ComputeKernels& cached_hwy_kernels(DistFunc func, DataType type) {
    const HighwayKernelCache& cache = hwy_kernel_cache();
    switch (func) {
        case DistFunc::DOT:
            switch (type) {
                case DataType::f32: return cache.dot_f32;
                case DataType::f16: return cache.dot_f16;
                case DataType::i16: return cache.dot_i16;
            }
            break;
        case DistFunc::L2:
            switch (type) {
                case DataType::f32: return cache.l2_f32;
                case DataType::f16: return cache.l2_f16;
                case DataType::i16: return cache.l2_i16;
            }
            break;
        case DistFunc::COS:
            switch (type) {
                case DataType::f32: return cache.cos_f32;
                case DataType::f16: return cache.cos_f16;
                case DataType::i16: return cache.cos_i16;
            }
            break;
    }
    throw std::runtime_error("cached_hwy_kernels: unsupported DistFunc/DataType.");
}

const HighwayDotScan& cached_hwy_dot_scan(DataType type) {
    const HighwayKernelCache& cache = hwy_kernel_cache();
    switch (type) {
        case DataType::f32: return cache.scan_dot_f32;
        case DataType::f16: return cache.scan_dot_f16;
        case DataType::i16: return cache.scan_dot_i16;
    }
    throw std::runtime_error("cached_hwy_dot_scan: unsupported DataType.");
}

const HighwayL2Scan& cached_hwy_l2_scan(DataType type) {
    const HighwayKernelCache& cache = hwy_kernel_cache();
    switch (type) {
        case DataType::f32: return cache.scan_l2_f32;
        case DataType::f16: return cache.scan_l2_f16;
        case DataType::i16: return cache.scan_l2_i16;
    }
    throw std::runtime_error("cached_hwy_l2_scan: unsupported DataType.");
}

const HighwayCosScan& cached_hwy_cos_scan(DataType type) {
    const HighwayKernelCache& cache = hwy_kernel_cache();
    switch (type) {
        case DataType::f32: return cache.scan_cos_f32;
        case DataType::f16: return cache.scan_cos_f16;
        case DataType::i16: return cache.scan_cos_i16;
    }
    throw std::runtime_error("cached_hwy_cos_scan: unsupported DataType.");
}

Ret validate_hwy_kernel_support(const ComputeKernels& kernels, DistFunc func, DataType type) {
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
}

Ret validate_hwy_scan_support(const HighwayDotScan& scan, DataType type) {
    if (scan.scan == nullptr) {
        return Ret(std::string("Highway::find_items: missing dot scanner for ")
            + data_type_to_string(type) + ".");
    }
    return Ret(0);
}

Ret validate_hwy_scan_support(const HighwayL2Scan& scan, DataType type) {
    if (scan.stored == nullptr || scan.fallback == nullptr) {
        return Ret(std::string("Highway::find_items: missing l2 scanner for ")
            + data_type_to_string(type) + ".");
    }
    return Ret(0);
}

Ret validate_hwy_scan_support(const HighwayCosScan& scan, DataType type) {
    if (scan.stored == nullptr || scan.fallback == nullptr) {
        return Ret(std::string("Highway::find_items: missing cos scanner for ")
            + data_type_to_string(type) + ".");
    }
    return Ret(0);
}

} // namespace

void initialize_hwy_runtime() {
    (void) hwy_kernel_cache();
}

Ret find_items_hw(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>* result, const BitsetFilter* bitset, uint64_t query_id) {
    if (vec == nullptr || count == 0 || result == nullptr) {
        return Ret("Highway::find_items: invalid arguments.");
    }

    result->clear();
    const DistFunc func = dataset.dist_func();
    const size_t dim = dataset.dim();
    const DataType type = dataset.type();
    const ComputeKernels& kernels = cached_hwy_kernels(func, type);
    CHECK(validate_hwy_kernel_support(kernels, func, type));
    if (query_id == 0) {
        query_id = next_scanner_query_id();
    }
    log_query_start(query_id, dataset.name(), func, type, dim, count,
        bitset != nullptr);

    DistHeap heap(DistItemCompare{func});
    heap.reserve(count);
    Timer timer("highway::query");

    std::vector<DataReaderPtr> readers;
    CHECK(collect_dataset_readers(dataset, query_id, &readers));

    if (func == DistFunc::DOT) {
        log_query_branch(query_id, "dot");
        const QueryDotContext query{vec, dim};
        const HighwayDotScan& dot_scan = cached_hwy_dot_scan(type);
        CHECK(validate_hwy_scan_support(dot_scan, type));
        const ScanDotFn scan_fn = dot_scan.scan;
        CHECK(scan_dataset_readers(
            query_id, readers, count, &heap,
            [query_id, query, scan_fn](const DataReader& reader, size_t local_count,
                    DistHeapEx* local_heap, const BitsetFilter* bitset_filter) {
                log_reader_scan_plan(query_id, reader, "dot", false, bitset_filter != nullptr);
                scan_fn(reader, local_count, local_heap, query, bitset_filter);
            },
            func, bitset));
    } else if (func == DistFunc::L2) {
        const double query_norm_sq = kernels.squared_norm(vec, dim);
        log_query_branch(query_id, "l2_with_optional_norms",
            query_norm_sq, query_norm_sq == 0.0);
        const QueryL2Context query{vec, dim, query_norm_sq};
        const HighwayL2Scan& scan = cached_hwy_l2_scan(type);
        CHECK(validate_hwy_scan_support(scan, type));
        const ScanL2StoredFn scan_stored = scan.stored;
        const ScanDistFn scan_fallback = scan.fallback;
        CHECK(scan_dataset_readers(
            query_id, readers, count, &heap,
            [query_id, query, scan_stored, scan_fallback](const DataReader& reader,
                    size_t local_count, DistHeapEx* local_heap,
                    const BitsetFilter* bitset_filter) {
                const bool uses_stored = reader.has_matching_stored_norms(DistFunc::L2);
                log_reader_scan_plan(query_id, reader,
                    uses_stored ? "l2_stored_norms" : "l2_dist_fallback",
                    uses_stored, bitset_filter != nullptr);
                if (uses_stored) {
                    scan_stored(reader, local_count, local_heap, query, bitset_filter);
                } else {
                    scan_fallback(reader, local_count, local_heap,
                        QueryDistContext{query.vec, query.dim}, bitset_filter);
                }
            },
            func, bitset));
    } else if (func == DistFunc::COS) {
        const double query_norm_sq = kernels.squared_norm(vec, dim);
        log_query_branch(query_id, "cos_with_optional_norms",
            query_norm_sq, query_norm_sq == 0.0);
        const QueryCosContext query{
            vec, dim, query_norm_sq, query_inverse_norm(query_norm_sq)};
        const HighwayCosScan& scan = cached_hwy_cos_scan(type);
        CHECK(validate_hwy_scan_support(scan, type));
        const ScanCosFn scan_stored = scan.stored;
        const ScanCosFn scan_fallback = scan.fallback;
        CHECK(scan_dataset_readers(
            query_id, readers, count, &heap,
            [query_id, query, scan_stored, scan_fallback](const DataReader& reader,
                    size_t local_count, DistHeapEx* local_heap,
                    const BitsetFilter* bitset_filter) {
                const bool uses_stored = reader.has_matching_stored_norms(DistFunc::COS);
                log_reader_scan_plan(query_id, reader,
                    uses_stored ? "cos_stored_norms" : "cos_query_norm_fallback",
                    uses_stored, bitset_filter != nullptr);
                if (uses_stored) {
                    scan_stored(reader, local_count, local_heap, query, bitset_filter);
                } else {
                    scan_fallback(reader, local_count, local_heap, query, bitset_filter);
                }
            },
            func, bitset));
    } else {
        return Ret("Highway::find_items: unsupported DistFunc.");
    }

    extract_items(&heap, result);
    log_query_finish(query_id, dataset.name(), func, type, dim, count,
        timer.elapsed_ms(), *result);
    return Ret(0);
}

ComputeKernels resolve_hwy_kernels(DistFunc func, DataType type) {
    return cached_hwy_kernels(func, type);
}

}  // namespace sketch2

#endif  // HWY_ONCE
