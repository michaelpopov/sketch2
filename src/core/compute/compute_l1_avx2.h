#pragma once
#include "core/compute/compute.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace sketch2 {

// Computes L1 (Manhattan) distance between two vectors.
class ComputeL1_AVX2 {
public:
    static double dist_f32(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_f16(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_i16(const uint8_t *a, const uint8_t *b, size_t dim);
};

#if defined(__AVX2__)

inline double ComputeL1_AVX2::dist_f32(const uint8_t *a, const uint8_t *b, size_t dim) {
    const float *va = reinterpret_cast<const float *>(a);
    const float *vb = reinterpret_cast<const float *>(b);
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 acc = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        const __m256 a8 = _mm256_loadu_ps(va + i);
        const __m256 b8 = _mm256_loadu_ps(vb + i);
        const __m256 d8 = _mm256_sub_ps(a8, b8);
        const __m256 abs_d8 = _mm256_andnot_ps(sign_mask, d8);
        acc = _mm256_add_ps(acc, abs_d8);
    }

    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, acc);
    double sum = static_cast<double>(lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5]
                                     + lanes[6] + lanes[7]);

    for (; i < dim; ++i) {
        sum += std::abs(va[i] - vb[i]);
    }
    return sum;
}

inline double ComputeL1_AVX2::dist_f16(const uint8_t *a, const uint8_t *b, size_t dim) {
    const float16 *va = reinterpret_cast<const float16 *>(a);
    const float16 *vb = reinterpret_cast<const float16 *>(b);
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        sum += std::abs(va[i] - vb[i]);
    }
    return sum;
}

inline double ComputeL1_AVX2::dist_i16(const uint8_t *a, const uint8_t *b, size_t dim) {
    const int16_t *va = reinterpret_cast<const int16_t *>(a);
    const int16_t *vb = reinterpret_cast<const int16_t *>(b);

    const __m256i zero = _mm256_setzero_si256();
    __m256i acc64 = _mm256_setzero_si256();

    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        const __m256i a16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(va + i));
        const __m256i b16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(vb + i));

        const __m128i a_lo16 = _mm256_castsi256_si128(a16);
        const __m128i a_hi16 = _mm256_extracti128_si256(a16, 1);
        const __m128i b_lo16 = _mm256_castsi256_si128(b16);
        const __m128i b_hi16 = _mm256_extracti128_si256(b16, 1);

        const __m256i a_lo32 = _mm256_cvtepi16_epi32(a_lo16);
        const __m256i a_hi32 = _mm256_cvtepi16_epi32(a_hi16);
        const __m256i b_lo32 = _mm256_cvtepi16_epi32(b_lo16);
        const __m256i b_hi32 = _mm256_cvtepi16_epi32(b_hi16);

        const __m256i abs_lo32 = _mm256_abs_epi32(_mm256_sub_epi32(a_lo32, b_lo32));
        const __m256i abs_hi32 = _mm256_abs_epi32(_mm256_sub_epi32(a_hi32, b_hi32));

        acc64 = _mm256_add_epi64(acc64, _mm256_unpacklo_epi32(abs_lo32, zero));
        acc64 = _mm256_add_epi64(acc64, _mm256_unpackhi_epi32(abs_lo32, zero));
        acc64 = _mm256_add_epi64(acc64, _mm256_unpacklo_epi32(abs_hi32, zero));
        acc64 = _mm256_add_epi64(acc64, _mm256_unpackhi_epi32(abs_hi32, zero));
    }

    alignas(32) uint64_t lanes[4];
    _mm256_store_si256(reinterpret_cast<__m256i *>(lanes), acc64);
    double sum = static_cast<double>(lanes[0] + lanes[1] + lanes[2] + lanes[3]);

    for (; i < dim; ++i) {
        const int diff = static_cast<int>(va[i]) - static_cast<int>(vb[i]);
        sum += std::abs(diff);
    }
    return sum;
}

#endif

} // namespace sketch2
