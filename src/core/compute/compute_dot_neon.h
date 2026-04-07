// Implements NEON-optimized DOT-distance kernels.

#pragma once
#include "core/compute/compute.h"
#include "core/compute/compute_neon_utils.h"
#include <cmath>
#include <cstdint>
#include <stdexcept>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace sketch2 {

// Computes DOT (Manhattan) distance between two vectors using NEON.
// ComputeDOT_Neon exists to provide NEON-specialized DOT kernels for ARM scan
// workloads while keeping the same typed API as the portable implementation.
class ComputeDOT_Neon {
public:
    static double dist_f32(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_f16(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_i16(const uint8_t *a, const uint8_t *b, size_t dim);
};

#if defined(__aarch64__)

inline void accumulate_mul_i32_as_i64_dot(int32x4_t a, int32x4_t b, int64x2_t* acc0, int64x2_t* acc1) {
    *acc0 = vaddq_s64(*acc0, vmull_s32(vget_low_s32(a), vget_low_s32(b)));
    *acc1 = vaddq_s64(*acc1, vmull_s32(vget_high_s32(a), vget_high_s32(b)));
}

inline double ComputeDOT_Neon::dist_f32(const uint8_t *a, const uint8_t *b, size_t dim) {
    const float *va = reinterpret_cast<const float *>(a);
    const float *vb = reinterpret_cast<const float *>(b);
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);

    size_t i = 0;
    const size_t simd8_end = dim & ~static_cast<size_t>(7);
    for (; i < simd8_end; i += 8) {
        acc0 = vmlaq_f32(acc0, vld1q_f32(va + i),     vld1q_f32(vb + i));
        acc1 = vmlaq_f32(acc1, vld1q_f32(va + i + 4), vld1q_f32(vb + i + 4));
    }
    const size_t simd4_end = dim & ~static_cast<size_t>(3);
    for (; i < simd4_end; i += 4) {
        acc0 = vmlaq_f32(acc0, vld1q_f32(va + i), vld1q_f32(vb + i));
    }

    double sum = static_cast<double>(vaddvq_f32(vaddq_f32(acc0, acc1)));
    for (; i < dim; ++i) {
        sum += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
    }
    return sum;
}

inline double ComputeDOT_Neon::dist_f16(const uint8_t *a, const uint8_t *b, size_t dim) {
    const float16 *va = reinterpret_cast<const float16 *>(a);
    const float16 *vb = reinterpret_cast<const float16 *>(b);
    double sum = 0.0;
    size_t i = 0;

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    const size_t simd8_end = dim & ~static_cast<size_t>(7);
    for (; i < simd8_end; i += 8) {
        const float16x8_t a8 = vld1q_f16(reinterpret_cast<const float16_t *>(va + i));
        const float16x8_t b8 = vld1q_f16(reinterpret_cast<const float16_t *>(vb + i));
        acc0 = vmlaq_f32(acc0, vcvt_f32_f16(vget_low_f16(a8)), vcvt_f32_f16(vget_low_f16(b8)));
        acc1 = vmlaq_f32(acc1, vcvt_f32_f16(vget_high_f16(a8)), vcvt_f32_f16(vget_high_f16(b8)));
    }
    sum = static_cast<double>(vaddvq_f32(vaddq_f32(acc0, acc1)));
#else
    float32x4_t acc = vdupq_n_f32(0.0f);
    const size_t simd4_end = dim & ~static_cast<size_t>(3);
    for (; i < simd4_end; i += 4) {
        float16x4_t a4 = vld1_f16(reinterpret_cast<const float16_t *>(va + i));
        float16x4_t b4 = vld1_f16(reinterpret_cast<const float16_t *>(vb + i));
        float32x4_t a4_f32 = vcvt_f32_f16(a4);
        float32x4_t b4_f32 = vcvt_f32_f16(b4);
        acc = vmlaq_f32(acc, a4_f32, b4_f32);
    }
    sum = static_cast<double>(vaddvq_f32(acc));
#endif

    for (; i < dim; ++i) {
        sum += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
    }
    return sum;
}

inline double ComputeDOT_Neon::dist_i16(const uint8_t *a, const uint8_t *b, size_t dim) {
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
        const int32x4_t a_lo = vmovl_s16(vget_low_s16(a8));
        const int32x4_t a_hi = vmovl_s16(vget_high_s16(a8));
        const int32x4_t b_lo = vmovl_s16(vget_low_s16(b8));
        const int32x4_t b_hi = vmovl_s16(vget_high_s16(b8));

        accumulate_mul_i32_as_i64_dot(a_lo, b_lo, &acc0, &acc1);
        accumulate_mul_i32_as_i64_dot(a_hi, b_hi, &acc2, &acc3);
    }

    int64_t total_sum = hsum_s64x2(acc0) + hsum_s64x2(acc1) + hsum_s64x2(acc2) + hsum_s64x2(acc3);

    for (; i < dim; ++i) {
        total_sum += static_cast<int64_t>(va[i]) * static_cast<int64_t>(vb[i]);
    }
    return static_cast<double>(total_sum);
}

#else

inline double ComputeDOT_Neon::dist_f32(const uint8_t *, const uint8_t *, size_t) {
    throw std::runtime_error("NEON f32 not supported on this platform");
}

inline double ComputeDOT_Neon::dist_f16(const uint8_t *, const uint8_t *, size_t) {
    throw std::runtime_error("NEON f16 not supported on this platform");
}

inline double ComputeDOT_Neon::dist_i16(const uint8_t *, const uint8_t *, size_t) {
    throw std::runtime_error("NEON i16 not supported on this platform");
}

#endif // __aarch64__

} // namespace sketch2
