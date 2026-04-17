// Implements NEON-optimized shared dot-product and squared-norm kernels.

#pragma once

#include "core/compute/compute.h"
#include "core/compute/compute_neon_utils.h"

#include <cstdint>
#include <stdexcept>

namespace sketch2 {

// ComputeDotNorm_Neon owns the NEON-specialized shared dot/norm kernels used
// by both cosine and L2 scan paths.
class ComputeDotNorm_Neon {
public:
    static double squared_norm_f32(const uint8_t *a, size_t dim);
    static double squared_norm_f16(const uint8_t *a, size_t dim);
    static double squared_norm_i16(const uint8_t *a, size_t dim);
    static double dot_f32(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dot_f16(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dot_i16(const uint8_t *a, const uint8_t *b, size_t dim);
};

#if defined(__aarch64__)

inline void accumulate_mul_i32_as_i64_cos(int32x4_t a, int32x4_t b, int64x2_t* acc0, int64x2_t* acc1) {
    *acc0 = vaddq_s64(*acc0, vmull_s32(vget_low_s32(a), vget_low_s32(b)));
    *acc1 = vaddq_s64(*acc1, vmull_s32(vget_high_s32(a), vget_high_s32(b)));
}

inline double ComputeDotNorm_Neon::squared_norm_f32(const uint8_t *a, size_t dim) {
    const float *va = reinterpret_cast<const float *>(a);
    float32x4_t norm_acc0 = vdupq_n_f32(0.0f);
    float32x4_t norm_acc1 = vdupq_n_f32(0.0f);

    size_t i = 0;
    const size_t simd8_end = dim & ~static_cast<size_t>(7);
    for (; i < simd8_end; i += 8) {
        const float32x4_t a0 = vld1q_f32(va + i);
        const float32x4_t a1 = vld1q_f32(va + i + 4);
        norm_acc0 = vmlaq_f32(norm_acc0, a0, a0);
        norm_acc1 = vmlaq_f32(norm_acc1, a1, a1);
    }
    const size_t simd4_end = dim & ~static_cast<size_t>(3);
    for (; i < simd4_end; i += 4) {
        const float32x4_t a0 = vld1q_f32(va + i);
        norm_acc0 = vmlaq_f32(norm_acc0, a0, a0);
    }

    double norm = static_cast<double>(vaddvq_f32(vaddq_f32(norm_acc0, norm_acc1)));
    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        norm += ai * ai;
    }
    return norm;
}

inline double ComputeDotNorm_Neon::dot_f32(const uint8_t *a, const uint8_t *b, size_t dim) {
    const float *va = reinterpret_cast<const float *>(a);
    const float *vb = reinterpret_cast<const float *>(b);
    float32x4_t dot_acc0 = vdupq_n_f32(0.0f);
    float32x4_t dot_acc1 = vdupq_n_f32(0.0f);

    size_t i = 0;
    const size_t simd8_end = dim & ~static_cast<size_t>(7);
    for (; i < simd8_end; i += 8) {
        dot_acc0 = vmlaq_f32(dot_acc0, vld1q_f32(va + i),     vld1q_f32(vb + i));
        dot_acc1 = vmlaq_f32(dot_acc1, vld1q_f32(va + i + 4), vld1q_f32(vb + i + 4));
    }
    const size_t simd4_end = dim & ~static_cast<size_t>(3);
    for (; i < simd4_end; i += 4) {
        dot_acc0 = vmlaq_f32(dot_acc0, vld1q_f32(va + i), vld1q_f32(vb + i));
    }

    double dot = static_cast<double>(vaddvq_f32(vaddq_f32(dot_acc0, dot_acc1)));
    for (; i < dim; ++i) {
        dot += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
    }
    return dot;
}

inline double ComputeDotNorm_Neon::squared_norm_f16(const uint8_t *a, size_t dim) {
    const float16 *va = reinterpret_cast<const float16 *>(a);
    float32x4_t norm_acc0 = vdupq_n_f32(0.0f);
    float32x4_t norm_acc1 = vdupq_n_f32(0.0f);

    size_t i = 0;
    const size_t simd8_end = dim & ~static_cast<size_t>(7);
    const size_t simd4_end = dim & ~static_cast<size_t>(3);
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    for (; i < simd8_end; i += 8) {
        const float16x8_t a8 = vld1q_f16(reinterpret_cast<const float16_t *>(va + i));
        const float32x4_t a_lo = vcvt_f32_f16(vget_low_f16(a8));
        const float32x4_t a_hi = vcvt_f32_f16(vget_high_f16(a8));
        norm_acc0 = vmlaq_f32(norm_acc0, a_lo, a_lo);
        norm_acc1 = vmlaq_f32(norm_acc1, a_hi, a_hi);
    }
#endif
    for (; i < simd4_end; i += 4) {
        const float32x4_t a4_f32 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const float16_t *>(va + i)));
        norm_acc0 = vmlaq_f32(norm_acc0, a4_f32, a4_f32);
    }

    double norm = static_cast<double>(vaddvq_f32(vaddq_f32(norm_acc0, norm_acc1)));
    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        norm += ai * ai;
    }
    return norm;
}

inline double ComputeDotNorm_Neon::dot_f16(const uint8_t *a, const uint8_t *b, size_t dim) {
    const float16 *va = reinterpret_cast<const float16 *>(a);
    const float16 *vb = reinterpret_cast<const float16 *>(b);
    float32x4_t dot_acc0 = vdupq_n_f32(0.0f);
    float32x4_t dot_acc1 = vdupq_n_f32(0.0f);

    size_t i = 0;
    const size_t simd8_end = dim & ~static_cast<size_t>(7);
    const size_t simd4_end = dim & ~static_cast<size_t>(3);
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    for (; i < simd8_end; i += 8) {
        const float16x8_t a8 = vld1q_f16(reinterpret_cast<const float16_t *>(va + i));
        const float16x8_t b8 = vld1q_f16(reinterpret_cast<const float16_t *>(vb + i));
        const float32x4_t a_lo = vcvt_f32_f16(vget_low_f16(a8));
        const float32x4_t a_hi = vcvt_f32_f16(vget_high_f16(a8));
        const float32x4_t b_lo = vcvt_f32_f16(vget_low_f16(b8));
        const float32x4_t b_hi = vcvt_f32_f16(vget_high_f16(b8));

        dot_acc0 = vmlaq_f32(dot_acc0, a_lo, b_lo);
        dot_acc1 = vmlaq_f32(dot_acc1, a_hi, b_hi);
    }
#endif
    for (; i < simd4_end; i += 4) {
        const float16x4_t a4 = vld1_f16(reinterpret_cast<const float16_t *>(va + i));
        const float16x4_t b4 = vld1_f16(reinterpret_cast<const float16_t *>(vb + i));
        dot_acc0 = vmlaq_f32(dot_acc0, vcvt_f32_f16(a4), vcvt_f32_f16(b4));
    }

    double dot = static_cast<double>(vaddvq_f32(vaddq_f32(dot_acc0, dot_acc1)));
    for (; i < dim; ++i) {
        dot += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
    }
    return dot;
}

inline double ComputeDotNorm_Neon::squared_norm_i16(const uint8_t *a, size_t dim) {
    const int16_t *va = reinterpret_cast<const int16_t *>(a);
    int64x2_t norm_acc0 = vdupq_n_s64(0);
    int64x2_t norm_acc1 = vdupq_n_s64(0);
    int64x2_t norm_acc2 = vdupq_n_s64(0);
    int64x2_t norm_acc3 = vdupq_n_s64(0);

    size_t i = 0;
    const size_t simd8_end = dim & ~static_cast<size_t>(7);
    for (; i < simd8_end; i += 8) {
        const int16x8_t a8 = vld1q_s16(va + i);
        const int32x4_t a_lo = vmovl_s16(vget_low_s16(a8));
        const int32x4_t a_hi = vmovl_s16(vget_high_s16(a8));
        accumulate_mul_i32_as_i64_cos(a_lo, a_lo, &norm_acc0, &norm_acc1);
        accumulate_mul_i32_as_i64_cos(a_hi, a_hi, &norm_acc2, &norm_acc3);
    }

    double norm = static_cast<double>(hsum_s64x2(norm_acc0) + hsum_s64x2(norm_acc1) +
                                      hsum_s64x2(norm_acc2) + hsum_s64x2(norm_acc3));
    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        norm += ai * ai;
    }
    return norm;
}

inline double ComputeDotNorm_Neon::dot_i16(const uint8_t *a, const uint8_t *b, size_t dim) {
    const int16_t *va = reinterpret_cast<const int16_t *>(a);
    const int16_t *vb = reinterpret_cast<const int16_t *>(b);
    int64x2_t dot_acc0 = vdupq_n_s64(0);
    int64x2_t dot_acc1 = vdupq_n_s64(0);
    int64x2_t dot_acc2 = vdupq_n_s64(0);
    int64x2_t dot_acc3 = vdupq_n_s64(0);

    size_t i = 0;
    const size_t simd8_end = dim & ~static_cast<size_t>(7);
    for (; i < simd8_end; i += 8) {
        const int16x8_t a8 = vld1q_s16(va + i);
        const int16x8_t b8 = vld1q_s16(vb + i);
        const int32x4_t a_lo = vmovl_s16(vget_low_s16(a8));
        const int32x4_t a_hi = vmovl_s16(vget_high_s16(a8));
        const int32x4_t b_lo = vmovl_s16(vget_low_s16(b8));
        const int32x4_t b_hi = vmovl_s16(vget_high_s16(b8));

        accumulate_mul_i32_as_i64_cos(a_lo, b_lo, &dot_acc0, &dot_acc1);
        accumulate_mul_i32_as_i64_cos(a_hi, b_hi, &dot_acc2, &dot_acc3);
    }

    double dot = static_cast<double>(hsum_s64x2(dot_acc0) + hsum_s64x2(dot_acc1) +
                                     hsum_s64x2(dot_acc2) + hsum_s64x2(dot_acc3));
    for (; i < dim; ++i) {
        dot += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
    }
    return dot;
}

#else

inline double ComputeDotNorm_Neon::dot_f32(const uint8_t *, const uint8_t *, size_t) {
    throw std::runtime_error("NEON f32 not supported on this platform");
}

inline double ComputeDotNorm_Neon::dot_f16(const uint8_t *, const uint8_t *, size_t) {
    throw std::runtime_error("NEON f16 not supported on this platform");
}

inline double ComputeDotNorm_Neon::dot_i16(const uint8_t *, const uint8_t *, size_t) {
    throw std::runtime_error("NEON i16 not supported on this platform");
}

inline double ComputeDotNorm_Neon::squared_norm_f32(const uint8_t *, size_t) {
    throw std::runtime_error("NEON f32 not supported on this platform");
}

inline double ComputeDotNorm_Neon::squared_norm_f16(const uint8_t *, size_t) {
    throw std::runtime_error("NEON f16 not supported on this platform");
}

inline double ComputeDotNorm_Neon::squared_norm_i16(const uint8_t *, size_t) {
    throw std::runtime_error("NEON i16 not supported on this platform");
}

#endif

} // namespace sketch2
