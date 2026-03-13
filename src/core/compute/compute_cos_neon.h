#pragma once
#include "core/compute/compute.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace sketch2 {

// Computes cosine distance between two vectors using NEON.
class ComputeCos_Neon {
public:
    static double dist_f32(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_f16(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_i16(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_f32_with_query_norm(const uint8_t *a, const uint8_t *b, size_t dim, double query_norm_sq);
    static double dist_f16_with_query_norm(const uint8_t *a, const uint8_t *b, size_t dim, double query_norm_sq);
    static double dist_i16_with_query_norm(const uint8_t *a, const uint8_t *b, size_t dim, double query_norm_sq);
    static double squared_norm_f32(const uint8_t *a, size_t dim);
    static double squared_norm_f16(const uint8_t *a, size_t dim);
    static double squared_norm_i16(const uint8_t *a, size_t dim);
};

#if defined(__aarch64__)

inline double finalize_cosine_distance_neon(double dot, double norm_a, double norm_b) {
    if (norm_a == 0.0 && norm_b == 0.0) {
        return 0.0;
    }
    if (norm_a == 0.0 || norm_b == 0.0) {
        return 1.0;
    }
    const double cosine = std::clamp(dot / std::sqrt(norm_a * norm_b), -1.0, 1.0);
    return 1.0 - cosine;
}

inline int64_t hsum_s64x2_cos(int64x2_t v) {
    return vgetq_lane_s64(v, 0) + vgetq_lane_s64(v, 1);
}

inline void accumulate_mul_i32_as_i64_cos(int32x4_t a, int32x4_t b, int64x2_t* acc0, int64x2_t* acc1) {
    *acc0 = vaddq_s64(*acc0, vmull_s32(vget_low_s32(a), vget_low_s32(b)));
    *acc1 = vaddq_s64(*acc1, vmull_s32(vget_high_s32(a), vget_high_s32(b)));
}

inline double ComputeCos_Neon::dist_f32(const uint8_t *a, const uint8_t *b, size_t dim) {
    return dist_f32_with_query_norm(a, b, dim, squared_norm_f32(b, dim));
}

inline double ComputeCos_Neon::squared_norm_f32(const uint8_t *a, size_t dim) {
    const float *va = reinterpret_cast<const float *>(a);
    double norm = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        norm += ai * ai;
    }
    return norm;
}

inline double ComputeCos_Neon::dist_f32_with_query_norm(const uint8_t *a, const uint8_t *b, size_t dim,
        double query_norm_sq) {
    const float *va = reinterpret_cast<const float *>(a);
    const float *vb = reinterpret_cast<const float *>(b);
    float32x4_t dot_acc = vdupq_n_f32(0.0f);
    float32x4_t norm_a_acc = vdupq_n_f32(0.0f);

    size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        const float32x4_t a4 = vld1q_f32(va + i);
        const float32x4_t b4 = vld1q_f32(vb + i);
        dot_acc = vmlaq_f32(dot_acc, a4, b4);
        norm_a_acc = vmlaq_f32(norm_a_acc, a4, a4);
    }

    double dot = static_cast<double>(vaddvq_f32(dot_acc));
    double norm_a = static_cast<double>(vaddvq_f32(norm_a_acc));
    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        const double bi = static_cast<double>(vb[i]);
        dot += ai * bi;
        norm_a += ai * ai;
    }

    return finalize_cosine_distance_neon(dot, norm_a, query_norm_sq);
}

inline double ComputeCos_Neon::dist_f16(const uint8_t *a, const uint8_t *b, size_t dim) {
    return dist_f16_with_query_norm(a, b, dim, squared_norm_f16(b, dim));
}

inline double ComputeCos_Neon::squared_norm_f16(const uint8_t *a, size_t dim) {
    const float16 *va = reinterpret_cast<const float16 *>(a);
    double norm = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        norm += ai * ai;
    }
    return norm;
}

inline double ComputeCos_Neon::dist_f16_with_query_norm(const uint8_t *a, const uint8_t *b, size_t dim,
        double query_norm_sq) {
    const float16 *va = reinterpret_cast<const float16 *>(a);
    const float16 *vb = reinterpret_cast<const float16 *>(b);
    float32x4_t dot_acc0 = vdupq_n_f32(0.0f);
    float32x4_t dot_acc1 = vdupq_n_f32(0.0f);
    float32x4_t norm_a_acc0 = vdupq_n_f32(0.0f);
    float32x4_t norm_a_acc1 = vdupq_n_f32(0.0f);

    size_t i = 0;
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    for (; i + 8 <= dim; i += 8) {
        const float16x8_t a8 = vld1q_f16(reinterpret_cast<const float16_t *>(va + i));
        const float16x8_t b8 = vld1q_f16(reinterpret_cast<const float16_t *>(vb + i));
        const float32x4_t a_lo = vcvt_f32_f16(vget_low_f16(a8));
        const float32x4_t a_hi = vcvt_f32_f16(vget_high_f16(a8));
        const float32x4_t b_lo = vcvt_f32_f16(vget_low_f16(b8));
        const float32x4_t b_hi = vcvt_f32_f16(vget_high_f16(b8));

        dot_acc0 = vmlaq_f32(dot_acc0, a_lo, b_lo);
        dot_acc1 = vmlaq_f32(dot_acc1, a_hi, b_hi);
        norm_a_acc0 = vmlaq_f32(norm_a_acc0, a_lo, a_lo);
        norm_a_acc1 = vmlaq_f32(norm_a_acc1, a_hi, a_hi);
    }
#endif
    for (; i + 4 <= dim; i += 4) {
        const float16x4_t a4 = vld1_f16(reinterpret_cast<const float16_t *>(va + i));
        const float16x4_t b4 = vld1_f16(reinterpret_cast<const float16_t *>(vb + i));
        const float32x4_t a4_f32 = vcvt_f32_f16(a4);
        const float32x4_t b4_f32 = vcvt_f32_f16(b4);

        dot_acc0 = vmlaq_f32(dot_acc0, a4_f32, b4_f32);
        norm_a_acc0 = vmlaq_f32(norm_a_acc0, a4_f32, a4_f32);
    }

    double dot = static_cast<double>(vaddvq_f32(vaddq_f32(dot_acc0, dot_acc1)));
    double norm_a = static_cast<double>(vaddvq_f32(vaddq_f32(norm_a_acc0, norm_a_acc1)));
    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        const double bi = static_cast<double>(vb[i]);
        dot += ai * bi;
        norm_a += ai * ai;
    }

    return finalize_cosine_distance_neon(dot, norm_a, query_norm_sq);
}

inline double ComputeCos_Neon::dist_i16(const uint8_t *a, const uint8_t *b, size_t dim) {
    return dist_i16_with_query_norm(a, b, dim, squared_norm_i16(b, dim));
}

inline double ComputeCos_Neon::squared_norm_i16(const uint8_t *a, size_t dim) {
    const int16_t *va = reinterpret_cast<const int16_t *>(a);
    double norm = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        norm += ai * ai;
    }
    return norm;
}

inline double ComputeCos_Neon::dist_i16_with_query_norm(const uint8_t *a, const uint8_t *b, size_t dim,
        double query_norm_sq) {
    const int16_t *va = reinterpret_cast<const int16_t *>(a);
    const int16_t *vb = reinterpret_cast<const int16_t *>(b);
    int64x2_t dot_acc0 = vdupq_n_s64(0);
    int64x2_t dot_acc1 = vdupq_n_s64(0);
    int64x2_t norm_a_acc0 = vdupq_n_s64(0);
    int64x2_t norm_a_acc1 = vdupq_n_s64(0);

    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        const int16x8_t a8 = vld1q_s16(va + i);
        const int16x8_t b8 = vld1q_s16(vb + i);
        const int32x4_t a_lo = vmovl_s16(vget_low_s16(a8));
        const int32x4_t a_hi = vmovl_s16(vget_high_s16(a8));
        const int32x4_t b_lo = vmovl_s16(vget_low_s16(b8));
        const int32x4_t b_hi = vmovl_s16(vget_high_s16(b8));

        accumulate_mul_i32_as_i64_cos(a_lo, b_lo, &dot_acc0, &dot_acc1);
        accumulate_mul_i32_as_i64_cos(a_hi, b_hi, &dot_acc0, &dot_acc1);
        accumulate_mul_i32_as_i64_cos(a_lo, a_lo, &norm_a_acc0, &norm_a_acc1);
        accumulate_mul_i32_as_i64_cos(a_hi, a_hi, &norm_a_acc0, &norm_a_acc1);
    }

    double dot = static_cast<double>(hsum_s64x2_cos(dot_acc0) + hsum_s64x2_cos(dot_acc1));
    double norm_a = static_cast<double>(hsum_s64x2_cos(norm_a_acc0) + hsum_s64x2_cos(norm_a_acc1));
    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        const double bi = static_cast<double>(vb[i]);
        dot += ai * bi;
        norm_a += ai * ai;
    }

    return finalize_cosine_distance_neon(dot, norm_a, query_norm_sq);
}

#else

inline double ComputeCos_Neon::dist_f32(const uint8_t *, const uint8_t *, size_t) {
    throw std::runtime_error("NEON f32 not supported on this platform");
}

inline double ComputeCos_Neon::dist_f32_with_query_norm(const uint8_t *, const uint8_t *, size_t, double) {
    throw std::runtime_error("NEON f32 not supported on this platform");
}

inline double ComputeCos_Neon::dist_f16(const uint8_t *, const uint8_t *, size_t) {
    throw std::runtime_error("NEON f16 not supported on this platform");
}

inline double ComputeCos_Neon::dist_f16_with_query_norm(const uint8_t *, const uint8_t *, size_t, double) {
    throw std::runtime_error("NEON f16 not supported on this platform");
}

inline double ComputeCos_Neon::dist_i16(const uint8_t *, const uint8_t *, size_t) {
    throw std::runtime_error("NEON i16 not supported on this platform");
}

inline double ComputeCos_Neon::dist_i16_with_query_norm(const uint8_t *, const uint8_t *, size_t, double) {
    throw std::runtime_error("NEON i16 not supported on this platform");
}

inline double ComputeCos_Neon::squared_norm_f32(const uint8_t *, size_t) {
    throw std::runtime_error("NEON f32 not supported on this platform");
}

inline double ComputeCos_Neon::squared_norm_f16(const uint8_t *, size_t) {
    throw std::runtime_error("NEON f16 not supported on this platform");
}

inline double ComputeCos_Neon::squared_norm_i16(const uint8_t *, size_t) {
    throw std::runtime_error("NEON i16 not supported on this platform");
}

#endif

} // namespace sketch2
