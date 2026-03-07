#pragma once
#include "core/compute/compute_avx2_utils.h"
#include "core/compute/compute.h"
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace sketch2 {

// Computes squared L2 distance between two vectors.
class ComputeL2_AVX2 {
public:
    static double dist_f32(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_f16(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_i16(const uint8_t *a, const uint8_t *b, size_t dim);
};

#if defined(__AVX2__)

inline double ComputeL2_AVX2::dist_f32(const uint8_t *a, const uint8_t *b, size_t dim) {
    const float *va = reinterpret_cast<const float *>(a);
    const float *vb = reinterpret_cast<const float *>(b);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    const bool aligned = (((reinterpret_cast<uintptr_t>(va) | reinterpret_cast<uintptr_t>(vb)) & 31u) == 0u);

    size_t i = 0;
    if (aligned) {
        for (; i + 32 <= dim; i += 32) {
            const __m256 a0 = _mm256_load_ps(va + i);
            const __m256 b0 = _mm256_load_ps(vb + i);
            const __m256 a1 = _mm256_load_ps(va + i + 8);
            const __m256 b1 = _mm256_load_ps(vb + i + 8);
            const __m256 a2 = _mm256_load_ps(va + i + 16);
            const __m256 b2 = _mm256_load_ps(vb + i + 16);
            const __m256 a3 = _mm256_load_ps(va + i + 24);
            const __m256 b3 = _mm256_load_ps(vb + i + 24);

            const __m256 d0 = _mm256_sub_ps(a0, b0);
            const __m256 d1 = _mm256_sub_ps(a1, b1);
            const __m256 d2 = _mm256_sub_ps(a2, b2);
            const __m256 d3 = _mm256_sub_ps(a3, b3);
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0, d0));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(d1, d1));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(d2, d2));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(d3, d3));
        }
        for (; i + 8 <= dim; i += 8) {
            const __m256 d = _mm256_sub_ps(_mm256_load_ps(va + i), _mm256_load_ps(vb + i));
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d, d));
        }
    } else {
        for (; i + 32 <= dim; i += 32) {
            const __m256 a0 = _mm256_loadu_ps(va + i);
            const __m256 b0 = _mm256_loadu_ps(vb + i);
            const __m256 a1 = _mm256_loadu_ps(va + i + 8);
            const __m256 b1 = _mm256_loadu_ps(vb + i + 8);
            const __m256 a2 = _mm256_loadu_ps(va + i + 16);
            const __m256 b2 = _mm256_loadu_ps(vb + i + 16);
            const __m256 a3 = _mm256_loadu_ps(va + i + 24);
            const __m256 b3 = _mm256_loadu_ps(vb + i + 24);

            const __m256 d0 = _mm256_sub_ps(a0, b0);
            const __m256 d1 = _mm256_sub_ps(a1, b1);
            const __m256 d2 = _mm256_sub_ps(a2, b2);
            const __m256 d3 = _mm256_sub_ps(a3, b3);
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0, d0));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(d1, d1));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(d2, d2));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(d3, d3));
        }
        for (; i + 8 <= dim; i += 8) {
            const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(va + i), _mm256_loadu_ps(vb + i));
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d, d));
        }
    }

    const __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    double sum = hsum_ps_256(acc);
    for (; i < dim; ++i) {
        const double d = static_cast<double>(va[i]) - static_cast<double>(vb[i]);
        sum += d * d;
    }
    return sum;
}

inline double ComputeL2_AVX2::dist_f16(const uint8_t *a, const uint8_t *b, size_t dim) {
    const float16 *va = reinterpret_cast<const float16 *>(a);
    const float16 *vb = reinterpret_cast<const float16 *>(b);
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double d = static_cast<double>(va[i]) - static_cast<double>(vb[i]);
        sum += d * d;
    }
    return sum;
}

inline double ComputeL2_AVX2::dist_i16(const uint8_t *a, const uint8_t *b, size_t dim) {
    const int16_t *va = reinterpret_cast<const int16_t *>(a);
    const int16_t *vb = reinterpret_cast<const int16_t *>(b);
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const int64_t d = static_cast<int64_t>(va[i]) - static_cast<int64_t>(vb[i]);
        sum += static_cast<double>(d * d);
    }
    return sum;
}

#endif

} // namespace sketch2
