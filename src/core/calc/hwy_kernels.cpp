// Highway-backed distance kernels for DOT, L2, and Cosine metrics.
// Uses the foreach_target pattern for automatic multi-target compilation
// and runtime dispatch.

#include "core/calc/hwy_kernels.h"
#include "core/calc/cosine_distance.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "core/calc/hwy_kernels.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

HWY_BEFORE_NAMESPACE();
namespace sketch2 {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

template <typename T>
HWY_INLINE const T* AsElements(const uint8_t* ptr) {
    return reinterpret_cast<const T*>(ptr);
}

// Loads n float16 values (stored as raw uint16_t-compatible memory) into a
// float vector by loading as uint16, bitcasting to float16_t, and promoting.
// The half-lane count means we load N/2 uint16 values to fill N float lanes.
template <class DF>
HWY_INLINE hn::VFromD<DF> LoadF16AsF32(DF df, const uint8_t* ptr) {
    const hn::Rebind<uint16_t, DF> du16;
    const hn::Rebind<hwy::float16_t, DF> df16;
    const auto u16 = hn::LoadU(du16, AsElements<uint16_t>(ptr));
    return hn::PromoteTo(df, hn::BitCast(df16, u16));
}

// Loads n int16 values and promotes to int32.
template <class DI32>
HWY_INLINE hn::VFromD<DI32> LoadI16AsI32(DI32 di32, const uint8_t* ptr) {
    const hn::Rebind<int16_t, DI32> di16;
    return hn::PromoteTo(di32, hn::LoadU(di16, AsElements<int16_t>(ptr)));
}

// ---------------------------------------------------------------------------
// DOT distance (Manhattan)
// ---------------------------------------------------------------------------

double DistDOTF32(const uint8_t* a, const uint8_t* b, size_t dim) {
    const float* va = AsElements<float>(a);
    const float* vb = AsElements<float>(b);
#if HWY_TARGET == HWY_SCALAR
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        sum += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
    }
    return sum;
#else
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
#endif
}

double DistDOTF16(const uint8_t* a, const uint8_t* b, size_t dim) {
    // Scalar tail
    const auto* va = AsElements<hwy::float16_t>(a);
    const auto* vb = AsElements<hwy::float16_t>(b);
#if HWY_TARGET == HWY_SCALAR
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        sum += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
    }
    return sum;
#else
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
#endif
}

double DistDOTI16(const uint8_t* a, const uint8_t* b, size_t dim) {
    const int16_t* va = AsElements<int16_t>(a);
    const int16_t* vb = AsElements<int16_t>(b);
    // Keep i16 DOT exact by accumulating in int64, but still vectorize.
    // Float-lane accumulation can lose precision for larger products/sums.
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

// ---------------------------------------------------------------------------
// L2 distance (Squared Euclidean)
// ---------------------------------------------------------------------------

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
    // Accumulate in double to match the existing compute kernels and avoid
    // overflow: signed int16 values can differ by up to 65535
    // (32767 - (-32768)), and squaring that exceeds int32.
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

// ---------------------------------------------------------------------------
// Dot product
// ---------------------------------------------------------------------------

double DotF32(const uint8_t* a, const uint8_t* b, size_t dim) {
    const float* va = AsElements<float>(a);
    const float* vb = AsElements<float>(b);
#if HWY_TARGET == HWY_SCALAR
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        sum += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
    }
    return sum;
#else
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
#endif
}

double DotF16(const uint8_t* a, const uint8_t* b, size_t dim) {
    const auto* va = AsElements<hwy::float16_t>(a);
    const auto* vb = AsElements<hwy::float16_t>(b);
#if HWY_TARGET == HWY_SCALAR
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        sum += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
    }
    return sum;
#else
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
#endif
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

// ---------------------------------------------------------------------------
// Squared norm
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Cosine distance (full: computes both norms)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Cosine distance with pre-computed query norm
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// HWY_ONCE: export tables, trampolines, and resolver
// ---------------------------------------------------------------------------

#if HWY_ONCE

namespace sketch2 {

// DOT
HWY_EXPORT(DistDOTF32);
HWY_EXPORT(DistDOTF16);
HWY_EXPORT(DistDOTI16);
// L2
HWY_EXPORT(DistL2F32);
HWY_EXPORT(DistL2F16);
HWY_EXPORT(DistL2I16);
// Dot
HWY_EXPORT(DotF32);
HWY_EXPORT(DotF16);
HWY_EXPORT(DotI16);
// Squared norm
HWY_EXPORT(SquaredNormF32);
HWY_EXPORT(SquaredNormF16);
HWY_EXPORT(SquaredNormI16);
// Cosine distance
HWY_EXPORT(DistCosF32);
HWY_EXPORT(DistCosF16);
HWY_EXPORT(DistCosI16);
// Cosine with query norm
HWY_EXPORT(DistCosWithQueryNormF32);
HWY_EXPORT(DistCosWithQueryNormF16);
HWY_EXPORT(DistCosWithQueryNormI16);

// Trampolines: each calls HWY_DYNAMIC_DISPATCH to pick the best target.

static double hwy_dist_dot_f32(const uint8_t* a, const uint8_t* b, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(DistDOTF32)(a, b, dim);
}
static double hwy_dist_dot_f16(const uint8_t* a, const uint8_t* b, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(DistDOTF16)(a, b, dim);
}
static double hwy_dist_dot_i16(const uint8_t* a, const uint8_t* b, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(DistDOTI16)(a, b, dim);
}

static double hwy_dist_l2_f32(const uint8_t* a, const uint8_t* b, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(DistL2F32)(a, b, dim);
}
static double hwy_dist_l2_f16(const uint8_t* a, const uint8_t* b, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(DistL2F16)(a, b, dim);
}
static double hwy_dist_l2_i16(const uint8_t* a, const uint8_t* b, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(DistL2I16)(a, b, dim);
}

static double hwy_dot_f32(const uint8_t* a, const uint8_t* b, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(DotF32)(a, b, dim);
}
static double hwy_dot_f16(const uint8_t* a, const uint8_t* b, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(DotF16)(a, b, dim);
}
static double hwy_dot_i16(const uint8_t* a, const uint8_t* b, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(DotI16)(a, b, dim);
}

static double hwy_squared_norm_f32(const uint8_t* a, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(SquaredNormF32)(a, dim);
}
static double hwy_squared_norm_f16(const uint8_t* a, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(SquaredNormF16)(a, dim);
}
static double hwy_squared_norm_i16(const uint8_t* a, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(SquaredNormI16)(a, dim);
}

static double hwy_dist_cos_f32(const uint8_t* a, const uint8_t* b, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(DistCosF32)(a, b, dim);
}
static double hwy_dist_cos_f16(const uint8_t* a, const uint8_t* b, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(DistCosF16)(a, b, dim);
}
static double hwy_dist_cos_i16(const uint8_t* a, const uint8_t* b, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(DistCosI16)(a, b, dim);
}

static double hwy_dist_cos_qn_f32(const uint8_t* a, const uint8_t* b, size_t dim, double qn) {
    return HWY_DYNAMIC_DISPATCH(DistCosWithQueryNormF32)(a, b, dim, qn);
}
static double hwy_dist_cos_qn_f16(const uint8_t* a, const uint8_t* b, size_t dim, double qn) {
    return HWY_DYNAMIC_DISPATCH(DistCosWithQueryNormF16)(a, b, dim, qn);
}
static double hwy_dist_cos_qn_i16(const uint8_t* a, const uint8_t* b, size_t dim, double qn) {
    return HWY_DYNAMIC_DISPATCH(DistCosWithQueryNormI16)(a, b, dim, qn);
}

// ---------------------------------------------------------------------------
// Resolver
// ---------------------------------------------------------------------------

CalcKernels resolve_hwy_kernels(DistFunc func, DataType type) {
    CalcKernels k;
    switch (func) {
        case DistFunc::DOT:
            switch (type) {
                case DataType::f32: k.dist = &hwy_dist_dot_f32; break;
                case DataType::f16: k.dist = &hwy_dist_dot_f16; break;
                case DataType::i16: k.dist = &hwy_dist_dot_i16; break;
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

}  // namespace sketch2

#endif  // HWY_ONCE
