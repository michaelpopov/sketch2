#pragma once
#include "core/compute/compute.h"
#include <cmath>
#include <cstdint>
#include <stdexcept>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace sketch2 {

// Computes L1 (Manhattan) distance between two vectors using NEON.
class ComputeL1_Neon {
public:
    static double dist_f32(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_f16(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_i16(const uint8_t *a, const uint8_t *b, size_t dim);
};

#if defined(__aarch64__)

inline double ComputeL1_Neon::dist_f32(const uint8_t *a, const uint8_t *b, size_t dim) {
    const float *va = reinterpret_cast<const float *>(a);
    const float *vb = reinterpret_cast<const float *>(b);
    float32x4_t acc = vdupq_n_f32(0.0f);

    size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        float32x4_t a4 = vld1q_f32(va + i);
        float32x4_t b4 = vld1q_f32(vb + i);
        float32x4_t diff = vsubq_f32(a4, b4);
        acc = vaddq_f32(acc, vabsq_f32(diff));
    }

    double sum = static_cast<double>(vaddvq_f32(acc));

    for (; i < dim; ++i) {
        sum += std::abs(va[i] - vb[i]);
    }
    return sum;
}

inline double ComputeL1_Neon::dist_f16(const uint8_t *a, const uint8_t *b, size_t dim) {
    const float16 *va = reinterpret_cast<const float16 *>(a);
    const float16 *vb = reinterpret_cast<const float16 *>(b);
    double sum = 0.0;
    size_t i = 0;

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    float16x8_t acc = vdupq_n_f16(0.0f);
    for (; i + 8 <= dim; i += 8) {
        float16x8_t a8 = vld1q_f16(reinterpret_cast<const float16_t *>(va + i));
        float16x8_t b8 = vld1q_f16(reinterpret_cast<const float16_t *>(vb + i));
        float16x8_t diff = vsubq_f16(a8, b8);
        acc = vaddq_f16(acc, vabsq_f16(diff));
    }
    // Manual horizontal sum as vaddvq_f16 might be missing in some environments
    float32x4_t acc_lo = vcvt_f32_f16(vget_low_f16(acc));
    float32x4_t acc_hi = vcvt_f32_f16(vget_high_f16(acc));
    sum = static_cast<double>(vaddvq_f32(vaddq_f32(acc_lo, acc_hi)));
#else
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= dim; i += 4) {
        float16x4_t a4 = vld1_f16(reinterpret_cast<const float16_t *>(va + i));
        float16x4_t b4 = vld1_f16(reinterpret_cast<const float16_t *>(vb + i));
        float32x4_t a4_f32 = vcvt_f32_f16(a4);
        float32x4_t b4_f32 = vcvt_f32_f16(b4);
        float32x4_t diff = vsubq_f32(a4_f32, b4_f32);
        acc = vaddq_f32(acc, vabsq_f32(diff));
    }
    sum = static_cast<double>(vaddvq_f32(acc));
#endif

    for (; i < dim; ++i) {
        sum += std::abs(static_cast<double>(va[i]) - static_cast<double>(vb[i]));
    }
    return sum;
}

inline double ComputeL1_Neon::dist_i16(const uint8_t *a, const uint8_t *b, size_t dim) {
    const int16_t *va = reinterpret_cast<const int16_t *>(a);
    const int16_t *vb = reinterpret_cast<const int16_t *>(b);
    uint32x4_t acc = vdupq_n_u32(0);

    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        int16x8_t a8 = vld1q_s16(va + i);
        int16x8_t b8 = vld1q_s16(vb + i);
        // vabdq_s16 returns int16x8_t on some systems/compilers
        uint16x8_t abs_diff = vreinterpretq_u16_s16(vabdq_s16(a8, b8));
        acc = vaddw_u16(acc, vget_low_u16(abs_diff));
        acc = vaddw_u16(acc, vget_high_u16(abs_diff));
    }

    uint64_t total_sum = static_cast<uint64_t>(vaddvq_u32(acc));

    for (; i < dim; ++i) {
        const int diff = static_cast<int>(va[i]) - static_cast<int>(vb[i]);
        total_sum += std::abs(diff);
    }
    return static_cast<double>(total_sum);
}

#else

inline double ComputeL1_Neon::dist_f32(const uint8_t *, const uint8_t *, size_t) {
    throw std::runtime_error("NEON f32 not supported on this platform");
}

inline double ComputeL1_Neon::dist_f16(const uint8_t *, const uint8_t *, size_t) {
    throw std::runtime_error("NEON f16 not supported on this platform");
}

inline double ComputeL1_Neon::dist_i16(const uint8_t *, const uint8_t *, size_t) {
    throw std::runtime_error("NEON i16 not supported on this platform");
}

#endif // __aarch64__

} // namespace sketch2
