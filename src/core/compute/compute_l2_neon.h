// Implements NEON-optimized L2-distance kernels.

#pragma once
#include "core/compute/compute.h"
#include "core/compute/compute_neon_utils.h"
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace sketch2 {

// ComputeL2_Neon exists to provide NEON-specialized squared-L2 kernels for ARM
// targets while preserving the same typed entry points as the scalar/x86 backends.
class ComputeL2_Neon {
public:
    static double dist_f32(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_f16(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_i16(const uint8_t *a, const uint8_t *b, size_t dim);
};

#if defined(__aarch64__)

inline void accumulate_squared_i32_as_i64_l2(int32x4_t diff, int64x2_t* acc0, int64x2_t* acc1) {
    *acc0 = vaddq_s64(*acc0, vmull_s32(vget_low_s32(diff), vget_low_s32(diff)));
    *acc1 = vaddq_s64(*acc1, vmull_s32(vget_high_s32(diff), vget_high_s32(diff)));
}

inline double ComputeL2_Neon::dist_f32(const uint8_t *a, const uint8_t *b, size_t dim) {
    const float *va = reinterpret_cast<const float *>(a);
    const float *vb = reinterpret_cast<const float *>(b);
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);

    size_t i = 0;
    const size_t simd8_end = dim & ~static_cast<size_t>(7);
    for (; i < simd8_end; i += 8) {
        const float32x4_t d0 = vsubq_f32(vld1q_f32(va + i),     vld1q_f32(vb + i));
        const float32x4_t d1 = vsubq_f32(vld1q_f32(va + i + 4), vld1q_f32(vb + i + 4));
        acc0 = vmlaq_f32(acc0, d0, d0);
        acc1 = vmlaq_f32(acc1, d1, d1);
    }
    const size_t simd4_end = dim & ~static_cast<size_t>(3);
    for (; i < simd4_end; i += 4) {
        const float32x4_t d = vsubq_f32(vld1q_f32(va + i), vld1q_f32(vb + i));
        acc0 = vmlaq_f32(acc0, d, d);
    }

    double sum = static_cast<double>(vaddvq_f32(vaddq_f32(acc0, acc1)));
    for (; i < dim; ++i) {
        const double d = static_cast<double>(va[i]) - static_cast<double>(vb[i]);
        sum += d * d;
    }
    return sum;
}

inline double ComputeL2_Neon::dist_f16(const uint8_t *a, const uint8_t *b, size_t dim) {
    const float16 *va = reinterpret_cast<const float16 *>(a);
    const float16 *vb = reinterpret_cast<const float16 *>(b);
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);

    size_t i = 0;
    const size_t simd8_end = dim & ~static_cast<size_t>(7);
    const size_t simd4_end = dim & ~static_cast<size_t>(3);
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    for (; i < simd8_end; i += 8) {
        const float16x8_t a8 = vld1q_f16(reinterpret_cast<const float16_t *>(va + i));
        const float16x8_t b8 = vld1q_f16(reinterpret_cast<const float16_t *>(vb + i));
        const float16x8_t d8 = vsubq_f16(a8, b8);
        const float32x4_t d_lo = vcvt_f32_f16(vget_low_f16(d8));
        const float32x4_t d_hi = vcvt_f32_f16(vget_high_f16(d8));
        acc0 = vmlaq_f32(acc0, d_lo, d_lo);
        acc1 = vmlaq_f32(acc1, d_hi, d_hi);
    }
#endif
    for (; i < simd4_end; i += 4) {
        const float16x4_t a4 = vld1_f16(reinterpret_cast<const float16_t *>(va + i));
        const float16x4_t b4 = vld1_f16(reinterpret_cast<const float16_t *>(vb + i));
        const float32x4_t a4_f32 = vcvt_f32_f16(a4);
        const float32x4_t b4_f32 = vcvt_f32_f16(b4);
        const float32x4_t d4 = vsubq_f32(a4_f32, b4_f32);
        acc0 = vmlaq_f32(acc0, d4, d4);
    }

    double sum = static_cast<double>(vaddvq_f32(vaddq_f32(acc0, acc1)));
    for (; i < dim; ++i) {
        const double d = static_cast<double>(va[i]) - static_cast<double>(vb[i]);
        sum += d * d;
    }
    return sum;
}

inline double ComputeL2_Neon::dist_i16(const uint8_t *a, const uint8_t *b, size_t dim) {
    const int16_t *va = reinterpret_cast<const int16_t *>(a);
    const int16_t *vb = reinterpret_cast<const int16_t *>(b);
    int64x2_t acc0 = vdupq_n_s64(0);
    int64x2_t acc1 = vdupq_n_s64(0);
    int64x2_t acc2 = vdupq_n_s64(0);
    int64x2_t acc3 = vdupq_n_s64(0);

    size_t i = 0;
    const size_t simd8_end = dim & ~static_cast<size_t>(7);
    for (; i < simd8_end; i += 8) {
        const int16x8_t a8 = vld1q_s16(va + i);
        const int16x8_t b8 = vld1q_s16(vb + i);
        const int32x4_t d_lo = vsubq_s32(vmovl_s16(vget_low_s16(a8)), vmovl_s16(vget_low_s16(b8)));
        const int32x4_t d_hi = vsubq_s32(vmovl_s16(vget_high_s16(a8)), vmovl_s16(vget_high_s16(b8)));
        accumulate_squared_i32_as_i64_l2(d_lo, &acc0, &acc1);
        accumulate_squared_i32_as_i64_l2(d_hi, &acc2, &acc3);
    }

    double sum = static_cast<double>(hsum_s64x2(acc0) + hsum_s64x2(acc1) +
                                     hsum_s64x2(acc2) + hsum_s64x2(acc3));
    for (; i < dim; ++i) {
        const int64_t d = static_cast<int64_t>(va[i]) - static_cast<int64_t>(vb[i]);
        sum += static_cast<double>(d * d);
    }
    return sum;
}

#else

inline double ComputeL2_Neon::dist_f32(const uint8_t *, const uint8_t *, size_t) {
    throw std::runtime_error("NEON f32 not supported on this platform");
}

inline double ComputeL2_Neon::dist_f16(const uint8_t *, const uint8_t *, size_t) {
    throw std::runtime_error("NEON f16 not supported on this platform");
}

inline double ComputeL2_Neon::dist_i16(const uint8_t *, const uint8_t *, size_t) {
    throw std::runtime_error("NEON i16 not supported on this platform");
}

#endif

} // namespace sketch2
