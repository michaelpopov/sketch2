// Highway-backed scanner and kernel resolver.
// Uses the foreach_target pattern for automatic multi-target compilation
// and runtime dispatch.

#include "core/compute/highway.h"

#include "core/compute/metric_finalizers.h"
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
    const hn::ScalableTag<int64_t> di64;
    const size_t N = hn::Lanes(di32);
    // Diffs of i16 fit in i32 and their squares in i64, so accumulate exactly
    // via WidenMulAccumulate rather than converting to f32 (which loses bits as
    // the running sum grows). Mirrors DotI16.
    auto acc_lo = hn::Zero(di64);
    auto acc_hi = hn::Zero(di64);
    size_t i = 0;
    for (; i + N <= dim; i += N) {
        const auto av = LoadI16AsI32(di32, a + i * 2);
        const auto bv = LoadI16AsI32(di32, b + i * 2);
        const auto diff = hn::Sub(av, bv);
        acc_lo = hn::WidenMulAccumulate(di64, diff, diff, acc_lo, acc_hi);
    }
    int64_t sum = hn::ReduceSum(di64, acc_lo) + hn::ReduceSum(di64, acc_hi);
    for (; i < dim; ++i) {
        const int64_t d = static_cast<int64_t>(va[i]) - static_cast<int64_t>(vb[i]);
        sum += d * d;
    }
    return static_cast<double>(sum);
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
    const hn::ScalableTag<int64_t> di64;
    const size_t N = hn::Lanes(di32);
    // Squares of i16 are integers; accumulate exactly in i64 via
    // WidenMulAccumulate instead of f32. Mirrors DotI16.
    auto acc_lo = hn::Zero(di64);
    auto acc_hi = hn::Zero(di64);
    size_t i = 0;
    for (; i + N <= dim; i += N) {
        const auto av = LoadI16AsI32(di32, a + i * 2);
        acc_lo = hn::WidenMulAccumulate(di64, av, av, acc_lo, acc_hi);
    }
    int64_t sum = hn::ReduceSum(di64, acc_lo) + hn::ReduceSum(di64, acc_hi);
    for (; i < dim; ++i) {
        const int64_t ai = static_cast<int64_t>(va[i]);
        sum += ai * ai;
    }
    return static_cast<double>(sum);
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
    double norm_a_sq = hn::ReduceSum(df, na_acc);
    double norm_b_sq = hn::ReduceSum(df, nb_acc);
    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        const double bi = static_cast<double>(vb[i]);
        dot += ai * bi;
        norm_a_sq += ai * ai;
        norm_b_sq += bi * bi;
    }
    return cos_dist_from_squared_norms(dot, norm_a_sq, norm_b_sq);
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
    double norm_a_sq = hn::ReduceSum(df, na_acc);
    double norm_b_sq = hn::ReduceSum(df, nb_acc);
    const auto* va = AsElements<hwy::float16_t>(a);
    const auto* vb = AsElements<hwy::float16_t>(b);
    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        const double bi = static_cast<double>(vb[i]);
        dot += ai * bi;
        norm_a_sq += ai * ai;
        norm_b_sq += bi * bi;
    }
    return cos_dist_from_squared_norms(dot, norm_a_sq, norm_b_sq);
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
    double norm_a_sq = hn::ReduceSum(df, na_acc);
    double norm_b_sq = hn::ReduceSum(df, nb_acc);
    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        const double bi = static_cast<double>(vb[i]);
        dot += ai * bi;
        norm_a_sq += ai * ai;
        norm_b_sq += bi * bi;
    }
    return cos_dist_from_squared_norms(dot, norm_a_sq, norm_b_sq);
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
    double norm_a_sq = hn::ReduceSum(df, na_acc);
    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        const double bi = static_cast<double>(vb[i]);
        dot += ai * bi;
        norm_a_sq += ai * ai;
    }
    return cos_dist_from_squared_norms(dot, norm_a_sq, query_norm_sq);
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
    double norm_a_sq = hn::ReduceSum(df, na_acc);
    const auto* va = AsElements<hwy::float16_t>(a);
    const auto* vb = AsElements<hwy::float16_t>(b);
    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        const double bi = static_cast<double>(vb[i]);
        dot += ai * bi;
        norm_a_sq += ai * ai;
    }
    return cos_dist_from_squared_norms(dot, norm_a_sq, query_norm_sq);
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
    double norm_a_sq = hn::ReduceSum(df, na_acc);
    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        const double bi = static_cast<double>(vb[i]);
        dot += ai * bi;
        norm_a_sq += ai * ai;
    }
    return cos_dist_from_squared_norms(dot, norm_a_sq, query_norm_sq);
}

// Per-reader scanners. Each compiles once per Highway target and threads the
// kernel through the existing scan-loop templates as a compile-time parameter,
// so the kernel inlines into the record loop. Runtime target dispatch then
// happens once per reader instead of once per candidate.

void ScanDotF32(const DataReader& reader, size_t count, LocalDistHeap* heap,
        const QueryDotContext& query, const BitsetFilter* bitset) {
    scan_dot<DotF32>(reader, count, heap, query, bitset);
}
void ScanDotF16(const DataReader& reader, size_t count, LocalDistHeap* heap,
        const QueryDotContext& query, const BitsetFilter* bitset) {
    scan_dot<DotF16>(reader, count, heap, query, bitset);
}
void ScanDotI16(const DataReader& reader, size_t count, LocalDistHeap* heap,
        const QueryDotContext& query, const BitsetFilter* bitset) {
    scan_dot<DotI16>(reader, count, heap, query, bitset);
}

void ScanL2F32Stored(const DataReader& reader, size_t count, LocalDistHeap* heap,
        const QueryL2Context& query, const BitsetFilter* bitset) {
    scan_l2_stored_norms<DotF32>(reader, count, heap, query, bitset);
}
void ScanL2F16Stored(const DataReader& reader, size_t count, LocalDistHeap* heap,
        const QueryL2Context& query, const BitsetFilter* bitset) {
    scan_l2_stored_norms<DotF16>(reader, count, heap, query, bitset);
}
void ScanL2I16Stored(const DataReader& reader, size_t count, LocalDistHeap* heap,
        const QueryL2Context& query, const BitsetFilter* bitset) {
    scan_l2_stored_norms<DotI16>(reader, count, heap, query, bitset);
}

void ScanCosF32Stored(const DataReader& reader, size_t count, LocalDistHeap* heap,
        const QueryCosContext& query, const BitsetFilter* bitset) {
    scan_cos_stored_norms<DotF32>(reader, count, heap, query, bitset);
}
void ScanCosF16Stored(const DataReader& reader, size_t count, LocalDistHeap* heap,
        const QueryCosContext& query, const BitsetFilter* bitset) {
    scan_cos_stored_norms<DotF16>(reader, count, heap, query, bitset);
}
void ScanCosI16Stored(const DataReader& reader, size_t count, LocalDistHeap* heap,
        const QueryCosContext& query, const BitsetFilter* bitset) {
    scan_cos_stored_norms<DotI16>(reader, count, heap, query, bitset);
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
HWY_EXPORT(ScanCosF32Stored);
HWY_EXPORT(ScanCosF16Stored);
HWY_EXPORT(ScanCosI16Stored);

using ScanDotFn = void (*)(const DataReader&, size_t, LocalDistHeap*,
    const QueryDotContext&, const BitsetFilter*);
using ScanL2StoredFn = void (*)(const DataReader&, size_t, LocalDistHeap*,
    const QueryL2Context&, const BitsetFilter*);
using ScanCosFn = void (*)(const DataReader&, size_t, LocalDistHeap*,
    const QueryCosContext&, const BitsetFilter*);

ComputeDotFn pick_dot(DataType type) {
    switch (type) {
        case DataType::f32: return HWY_DYNAMIC_POINTER(DotF32);
        case DataType::f16: return HWY_DYNAMIC_POINTER(DotF16);
        case DataType::i16: return HWY_DYNAMIC_POINTER(DotI16);
        case DataType::f8: return nullptr;
    }
    return nullptr;
}

ComputeDistFn pick_l2_dist(DataType type) {
    switch (type) {
        case DataType::f32: return HWY_DYNAMIC_POINTER(DistL2F32);
        case DataType::f16: return HWY_DYNAMIC_POINTER(DistL2F16);
        case DataType::i16: return HWY_DYNAMIC_POINTER(DistL2I16);
        case DataType::f8: return nullptr;
    }
    return nullptr;
}

ComputeDistFn pick_cos_dist(DataType type) {
    switch (type) {
        case DataType::f32: return HWY_DYNAMIC_POINTER(DistCosF32);
        case DataType::f16: return HWY_DYNAMIC_POINTER(DistCosF16);
        case DataType::i16: return HWY_DYNAMIC_POINTER(DistCosI16);
        case DataType::f8: return nullptr;
    }
    return nullptr;
}

ComputeDistWithQueryNormFn pick_cos_query_norm(DataType type) {
    switch (type) {
        case DataType::f32: return HWY_DYNAMIC_POINTER(DistCosWithQueryNormF32);
        case DataType::f16: return HWY_DYNAMIC_POINTER(DistCosWithQueryNormF16);
        case DataType::i16: return HWY_DYNAMIC_POINTER(DistCosWithQueryNormI16);
        case DataType::f8: return nullptr;
    }
    return nullptr;
}

ComputeSquaredNormFn pick_squared_norm(DataType type) {
    switch (type) {
        case DataType::f32: return HWY_DYNAMIC_POINTER(SquaredNormF32);
        case DataType::f16: return HWY_DYNAMIC_POINTER(SquaredNormF16);
        case DataType::i16: return HWY_DYNAMIC_POINTER(SquaredNormI16);
        case DataType::f8: return nullptr;
    }
    return nullptr;
}

ScanDotFn pick_scan_dot(DataType type) {
    switch (type) {
        case DataType::f32: return HWY_DYNAMIC_POINTER(ScanDotF32);
        case DataType::f16: return HWY_DYNAMIC_POINTER(ScanDotF16);
        case DataType::i16: return HWY_DYNAMIC_POINTER(ScanDotI16);
        case DataType::f8: return nullptr;
    }
    return nullptr;
}

ScanL2StoredFn pick_scan_l2_stored(DataType type) {
    switch (type) {
        case DataType::f32: return HWY_DYNAMIC_POINTER(ScanL2F32Stored);
        case DataType::f16: return HWY_DYNAMIC_POINTER(ScanL2F16Stored);
        case DataType::i16: return HWY_DYNAMIC_POINTER(ScanL2I16Stored);
        case DataType::f8: return nullptr;
    }
    return nullptr;
}

ScanCosFn pick_scan_cos_stored(DataType type) {
    switch (type) {
        case DataType::f32: return HWY_DYNAMIC_POINTER(ScanCosF32Stored);
        case DataType::f16: return HWY_DYNAMIC_POINTER(ScanCosF16Stored);
        case DataType::i16: return HWY_DYNAMIC_POINTER(ScanCosI16Stored);
        case DataType::f8: return nullptr;
    }
    return nullptr;
}

// Guards a runtime-resolved kernel pointer. The pick_* resolvers return nullptr
// for any DataType they do not handle (a corrupted header or a future enum
// value), so callers translate that into a clean error instead of calling
// through a null function pointer.
template <typename Fn>
Ret require_kernel(Fn fn) {
    if (fn == nullptr) {
        return Ret("Highway::find_items: unsupported data type.");
    }
    return Ret(0);
}

} // namespace

void initialize_hwy_runtime() {
    static const bool warmed = []() {
        hwy::GetChosenTarget().Update(hwy::SupportedTargets());
        return true;
    }();
    (void) warmed;
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
    if (query_id == 0) {
        query_id = next_scanner_query_id();
    }
    log_query_start(query_id, dataset.name(), func, type, dim, count,
        bitset != nullptr);

    Timer timer("highway::query");

    std::vector<DataReaderPtr> readers;
    CHECK(collect_dataset_readers(dataset, query_id, &readers));

    switch (func) {
        case DistFunc::DOT: {
            log_query_branch(query_id, "dot");
            const QueryDotContext query{vec, dim};
            const ScanDotFn scan_fn = pick_scan_dot(type);
            CHECK(require_kernel(scan_fn));
            CHECK(scan_dataset_readers(
                query_id, readers, count, result,
                [query_id, query, scan_fn](const DataReader& reader, size_t local_count,
                        LocalDistHeap* local_heap, const BitsetFilter* bitset_filter) {
                    log_reader_scan_plan(query_id, reader, "dot", false, bitset_filter != nullptr);
                    scan_fn(reader, local_count, local_heap, query, bitset_filter);
                },
                func, bitset));
            break;
        }
        case DistFunc::L2: {
            const ComputeSquaredNormFn squared_norm_fn = pick_squared_norm(type);
            CHECK(require_kernel(squared_norm_fn));
            const double query_norm_sq = squared_norm_fn(vec, dim);
            log_query_branch(query_id, "l2_stored_norms",
                query_norm_sq, query_norm_sq == 0.0);
            const QueryL2Context query{vec, dim, query_norm_sq};
            const ScanL2StoredFn scan_stored = pick_scan_l2_stored(type);
            CHECK(require_kernel(scan_stored));
            CHECK(scan_dataset_readers(
                query_id, readers, count, result,
                [query_id, query, scan_stored](const DataReader& reader,
                        size_t local_count, LocalDistHeap* local_heap,
                        const BitsetFilter* bitset_filter) {
                    // DatasetReader::open_reader_ rejects L2 files without matching
                    // stored norms, so every collected reader is guaranteed to have them.
                    assert(reader.has_matching_stored_norms(DistFunc::L2));
                    log_reader_scan_plan(query_id, reader, "l2_stored_norms",
                        true, bitset_filter != nullptr);
                    scan_stored(reader, local_count, local_heap, query, bitset_filter);
                },
                func, bitset));
            break;
        }
        case DistFunc::COS: {
            const ComputeSquaredNormFn squared_norm_fn = pick_squared_norm(type);
            CHECK(require_kernel(squared_norm_fn));
            const double query_norm_sq = squared_norm_fn(vec, dim);
            log_query_branch(query_id, "cos_stored_norms",
                query_norm_sq, query_norm_sq == 0.0);
            const QueryCosContext query{
                vec, dim, query_norm_sq, query_inverse_norm(query_norm_sq)};
            const ScanCosFn scan_stored = pick_scan_cos_stored(type);
            CHECK(require_kernel(scan_stored));
            CHECK(scan_dataset_readers(
                query_id, readers, count, result,
                [query_id, query, scan_stored](const DataReader& reader,
                        size_t local_count, LocalDistHeap* local_heap,
                        const BitsetFilter* bitset_filter) {
                    // DatasetReader::open_reader_ rejects COS files without matching
                    // stored inverse norms, so every collected reader has them.
                    assert(reader.has_matching_stored_norms(DistFunc::COS));
                    log_reader_scan_plan(query_id, reader, "cos_stored_norms",
                        true, bitset_filter != nullptr);
                    scan_stored(reader, local_count, local_heap, query, bitset_filter);
                },
                func, bitset));
            break;
        }
        default:
            return Ret("Highway::find_items: unsupported DistFunc.");
    }

    log_query_finish(query_id, dataset.name(), func, type, dim, count,
        timer.elapsed_ms(), *result);
    return Ret(0);
}

ComputeKernels resolve_hwy_kernels(DistFunc func, DataType type) {
    ComputeKernels k;
    switch (func) {
        case DistFunc::DOT: {
            const ComputeDotFn dot = pick_dot(type);
            k.dist = dot;
            k.dot = dot;
            return k;
        }
        case DistFunc::L2:
            k.dist = pick_l2_dist(type);
            k.squared_norm = pick_squared_norm(type);
            k.dot = pick_dot(type);
            return k;
        case DistFunc::COS:
            k.dist = pick_cos_dist(type);
            k.dist_with_query_norm = pick_cos_query_norm(type);
            k.squared_norm = pick_squared_norm(type);
            k.dot = pick_dot(type);
            return k;
    }
    return k;
}

}  // namespace sketch2

#endif  // HWY_ONCE
