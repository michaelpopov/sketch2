#pragma once
#include "core/compute/compute.h"
#include <cmath>
#include <cstdint>
#include <stdexcept>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace sketch2 {

// Computes squared L2 distance between two vectors using NEON.
class ComputeL2_Neon {
public:
    static double dist_f32(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_f16(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_i16(const uint8_t *a, const uint8_t *b, size_t dim);
};

#if defined(__aarch64__)

inline double ComputeL2_Neon::dist_f32(const uint8_t *a, const uint8_t *b, size_t dim) {
    const float *va = reinterpret_cast<const float *>(a);
    const float *vb = reinterpret_cast<const float *>(b);
    float32x4_t acc = vdupq_n_f32(0.0f);

    size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        const float32x4_t d = vsubq_f32(vld1q_f32(va + i), vld1q_f32(vb + i));
        acc = vmlaq_f32(acc, d, d);
    }

    double sum = static_cast<double>(vaddvq_f32(acc));
    for (; i < dim; ++i) {
        const double d = static_cast<double>(va[i]) - static_cast<double>(vb[i]);
        sum += d * d;
    }
    return sum;
}

inline double ComputeL2_Neon::dist_f16(const uint8_t *a, const uint8_t *b, size_t dim) {
    const float16 *va = reinterpret_cast<const float16 *>(a);
    const float16 *vb = reinterpret_cast<const float16 *>(b);
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double d = static_cast<double>(va[i]) - static_cast<double>(vb[i]);
        sum += d * d;
    }
    return sum;
}

inline double ComputeL2_Neon::dist_i16(const uint8_t *a, const uint8_t *b, size_t dim) {
    const int16_t *va = reinterpret_cast<const int16_t *>(a);
    const int16_t *vb = reinterpret_cast<const int16_t *>(b);
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
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
