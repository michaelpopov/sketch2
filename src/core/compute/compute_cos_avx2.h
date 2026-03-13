#pragma once
#include "core/compute/compute_avx2_utils.h"
#include "core/compute/compute.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace sketch2 {

// Computes cosine distance between two vectors using AVX2.
class ComputeCos_AVX2 {
public:
    static double dist_f32(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_f16(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_i16(const uint8_t *a, const uint8_t *b, size_t dim);
};

#if defined(__AVX2__)

inline double finalize_cosine_distance_avx2(double dot, double norm_a, double norm_b) {
    if (norm_a == 0.0 && norm_b == 0.0) {
        return 0.0;
    }
    if (norm_a == 0.0 || norm_b == 0.0) {
        return 1.0;
    }
    const double cosine = std::clamp(dot / std::sqrt(norm_a * norm_b), -1.0, 1.0);
    return 1.0 - cosine;
}

inline double hsum_epi64_256_cos(__m256i v) {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    const __m128i sum = _mm_add_epi64(lo, hi);
    alignas(16) int64_t lanes[2];
    _mm_store_si128(reinterpret_cast<__m128i *>(lanes), sum);
    return static_cast<double>(lanes[0]) + static_cast<double>(lanes[1]);
}

inline __m256i accumulate_mul_i32_as_i64_cos(__m256i acc, __m256i a32, __m256i b32) {
    const __m256i a_odd = _mm256_shuffle_epi32(a32, _MM_SHUFFLE(3, 3, 1, 1));
    const __m256i b_odd = _mm256_shuffle_epi32(b32, _MM_SHUFFLE(3, 3, 1, 1));
    const __m256i even_prod = _mm256_mul_epi32(a32, b32);
    const __m256i odd_prod = _mm256_mul_epi32(a_odd, b_odd);
    return _mm256_add_epi64(acc, _mm256_add_epi64(even_prod, odd_prod));
}

inline void accumulate_i16_products_as_i64_cos(__m256i a16, __m256i b16, __m256i* dot_acc,
        __m256i* norm_a_acc, __m256i* norm_b_acc) {
    const __m128i a_lo16 = _mm256_castsi256_si128(a16);
    const __m128i a_hi16 = _mm256_extracti128_si256(a16, 1);
    const __m128i b_lo16 = _mm256_castsi256_si128(b16);
    const __m128i b_hi16 = _mm256_extracti128_si256(b16, 1);

    const __m256i a_lo32 = _mm256_cvtepi16_epi32(a_lo16);
    const __m256i a_hi32 = _mm256_cvtepi16_epi32(a_hi16);
    const __m256i b_lo32 = _mm256_cvtepi16_epi32(b_lo16);
    const __m256i b_hi32 = _mm256_cvtepi16_epi32(b_hi16);

    *dot_acc = accumulate_mul_i32_as_i64_cos(*dot_acc, a_lo32, b_lo32);
    *dot_acc = accumulate_mul_i32_as_i64_cos(*dot_acc, a_hi32, b_hi32);
    *norm_a_acc = accumulate_mul_i32_as_i64_cos(*norm_a_acc, a_lo32, a_lo32);
    *norm_a_acc = accumulate_mul_i32_as_i64_cos(*norm_a_acc, a_hi32, a_hi32);
    *norm_b_acc = accumulate_mul_i32_as_i64_cos(*norm_b_acc, b_lo32, b_lo32);
    *norm_b_acc = accumulate_mul_i32_as_i64_cos(*norm_b_acc, b_hi32, b_hi32);
}

inline double ComputeCos_AVX2::dist_f32(const uint8_t *a, const uint8_t *b, size_t dim) {
    const float *va = reinterpret_cast<const float *>(a);
    const float *vb = reinterpret_cast<const float *>(b);
    __m256 dot0 = _mm256_setzero_ps();
    __m256 dot1 = _mm256_setzero_ps();
    __m256 dot2 = _mm256_setzero_ps();
    __m256 dot3 = _mm256_setzero_ps();
    __m256 norm_a0 = _mm256_setzero_ps();
    __m256 norm_a1 = _mm256_setzero_ps();
    __m256 norm_a2 = _mm256_setzero_ps();
    __m256 norm_a3 = _mm256_setzero_ps();
    __m256 norm_b0 = _mm256_setzero_ps();
    __m256 norm_b1 = _mm256_setzero_ps();
    __m256 norm_b2 = _mm256_setzero_ps();
    __m256 norm_b3 = _mm256_setzero_ps();
    const bool aligned =
        (((reinterpret_cast<uintptr_t>(va) | reinterpret_cast<uintptr_t>(vb)) & (kAvx2VectorAlignment - 1u)) == 0u);

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

            dot0 = fmadd_ps(a0, b0, dot0);
            dot1 = fmadd_ps(a1, b1, dot1);
            dot2 = fmadd_ps(a2, b2, dot2);
            dot3 = fmadd_ps(a3, b3, dot3);
            norm_a0 = fmadd_ps(a0, a0, norm_a0);
            norm_a1 = fmadd_ps(a1, a1, norm_a1);
            norm_a2 = fmadd_ps(a2, a2, norm_a2);
            norm_a3 = fmadd_ps(a3, a3, norm_a3);
            norm_b0 = fmadd_ps(b0, b0, norm_b0);
            norm_b1 = fmadd_ps(b1, b1, norm_b1);
            norm_b2 = fmadd_ps(b2, b2, norm_b2);
            norm_b3 = fmadd_ps(b3, b3, norm_b3);
        }
        for (; i + 8 <= dim; i += 8) {
            const __m256 a8 = _mm256_load_ps(va + i);
            const __m256 b8 = _mm256_load_ps(vb + i);
            dot0 = fmadd_ps(a8, b8, dot0);
            norm_a0 = fmadd_ps(a8, a8, norm_a0);
            norm_b0 = fmadd_ps(b8, b8, norm_b0);
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

            dot0 = fmadd_ps(a0, b0, dot0);
            dot1 = fmadd_ps(a1, b1, dot1);
            dot2 = fmadd_ps(a2, b2, dot2);
            dot3 = fmadd_ps(a3, b3, dot3);
            norm_a0 = fmadd_ps(a0, a0, norm_a0);
            norm_a1 = fmadd_ps(a1, a1, norm_a1);
            norm_a2 = fmadd_ps(a2, a2, norm_a2);
            norm_a3 = fmadd_ps(a3, a3, norm_a3);
            norm_b0 = fmadd_ps(b0, b0, norm_b0);
            norm_b1 = fmadd_ps(b1, b1, norm_b1);
            norm_b2 = fmadd_ps(b2, b2, norm_b2);
            norm_b3 = fmadd_ps(b3, b3, norm_b3);
        }
        for (; i + 8 <= dim; i += 8) {
            const __m256 a8 = _mm256_loadu_ps(va + i);
            const __m256 b8 = _mm256_loadu_ps(vb + i);
            dot0 = fmadd_ps(a8, b8, dot0);
            norm_a0 = fmadd_ps(a8, a8, norm_a0);
            norm_b0 = fmadd_ps(b8, b8, norm_b0);
        }
    }

    const __m256 dot = _mm256_add_ps(_mm256_add_ps(dot0, dot1), _mm256_add_ps(dot2, dot3));
    const __m256 norm_a = _mm256_add_ps(_mm256_add_ps(norm_a0, norm_a1), _mm256_add_ps(norm_a2, norm_a3));
    const __m256 norm_b = _mm256_add_ps(_mm256_add_ps(norm_b0, norm_b1), _mm256_add_ps(norm_b2, norm_b3));

    double dot_sum = hsum_ps_256(dot);
    double norm_a_sum = hsum_ps_256(norm_a);
    double norm_b_sum = hsum_ps_256(norm_b);

    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        const double bi = static_cast<double>(vb[i]);
        dot_sum += ai * bi;
        norm_a_sum += ai * ai;
        norm_b_sum += bi * bi;
    }

    return finalize_cosine_distance_avx2(dot_sum, norm_a_sum, norm_b_sum);
}

inline double ComputeCos_AVX2::dist_f16(const uint8_t *a, const uint8_t *b, size_t dim) {
    const float16 *va = reinterpret_cast<const float16 *>(a);
    const float16 *vb = reinterpret_cast<const float16 *>(b);
    __m256 dot0 = _mm256_setzero_ps();
    __m256 dot1 = _mm256_setzero_ps();
    __m256 dot2 = _mm256_setzero_ps();
    __m256 dot3 = _mm256_setzero_ps();
    __m256 norm_a0 = _mm256_setzero_ps();
    __m256 norm_a1 = _mm256_setzero_ps();
    __m256 norm_a2 = _mm256_setzero_ps();
    __m256 norm_a3 = _mm256_setzero_ps();
    __m256 norm_b0 = _mm256_setzero_ps();
    __m256 norm_b1 = _mm256_setzero_ps();
    __m256 norm_b2 = _mm256_setzero_ps();
    __m256 norm_b3 = _mm256_setzero_ps();
    const bool aligned =
        (((reinterpret_cast<uintptr_t>(va) | reinterpret_cast<uintptr_t>(vb)) & (kHalfVectorAlignment - 1u)) == 0u);

    size_t i = 0;
    if (aligned) {
        for (; i + 32 <= dim; i += 32) {
            const __m256 a0 = load_f16x8_ps_aligned(va + i);
            const __m256 b0 = load_f16x8_ps_aligned(vb + i);
            const __m256 a1 = load_f16x8_ps_aligned(va + i + 8);
            const __m256 b1 = load_f16x8_ps_aligned(vb + i + 8);
            const __m256 a2 = load_f16x8_ps_aligned(va + i + 16);
            const __m256 b2 = load_f16x8_ps_aligned(vb + i + 16);
            const __m256 a3 = load_f16x8_ps_aligned(va + i + 24);
            const __m256 b3 = load_f16x8_ps_aligned(vb + i + 24);

            dot0 = fmadd_ps(a0, b0, dot0);
            dot1 = fmadd_ps(a1, b1, dot1);
            dot2 = fmadd_ps(a2, b2, dot2);
            dot3 = fmadd_ps(a3, b3, dot3);
            norm_a0 = fmadd_ps(a0, a0, norm_a0);
            norm_a1 = fmadd_ps(a1, a1, norm_a1);
            norm_a2 = fmadd_ps(a2, a2, norm_a2);
            norm_a3 = fmadd_ps(a3, a3, norm_a3);
            norm_b0 = fmadd_ps(b0, b0, norm_b0);
            norm_b1 = fmadd_ps(b1, b1, norm_b1);
            norm_b2 = fmadd_ps(b2, b2, norm_b2);
            norm_b3 = fmadd_ps(b3, b3, norm_b3);
        }
        for (; i + 8 <= dim; i += 8) {
            const __m256 a8 = load_f16x8_ps_aligned(va + i);
            const __m256 b8 = load_f16x8_ps_aligned(vb + i);
            dot0 = fmadd_ps(a8, b8, dot0);
            norm_a0 = fmadd_ps(a8, a8, norm_a0);
            norm_b0 = fmadd_ps(b8, b8, norm_b0);
        }
    } else {
        for (; i + 32 <= dim; i += 32) {
            const __m256 a0 = load_f16x8_ps(va + i);
            const __m256 b0 = load_f16x8_ps(vb + i);
            const __m256 a1 = load_f16x8_ps(va + i + 8);
            const __m256 b1 = load_f16x8_ps(vb + i + 8);
            const __m256 a2 = load_f16x8_ps(va + i + 16);
            const __m256 b2 = load_f16x8_ps(vb + i + 16);
            const __m256 a3 = load_f16x8_ps(va + i + 24);
            const __m256 b3 = load_f16x8_ps(vb + i + 24);

            dot0 = fmadd_ps(a0, b0, dot0);
            dot1 = fmadd_ps(a1, b1, dot1);
            dot2 = fmadd_ps(a2, b2, dot2);
            dot3 = fmadd_ps(a3, b3, dot3);
            norm_a0 = fmadd_ps(a0, a0, norm_a0);
            norm_a1 = fmadd_ps(a1, a1, norm_a1);
            norm_a2 = fmadd_ps(a2, a2, norm_a2);
            norm_a3 = fmadd_ps(a3, a3, norm_a3);
            norm_b0 = fmadd_ps(b0, b0, norm_b0);
            norm_b1 = fmadd_ps(b1, b1, norm_b1);
            norm_b2 = fmadd_ps(b2, b2, norm_b2);
            norm_b3 = fmadd_ps(b3, b3, norm_b3);
        }
        for (; i + 8 <= dim; i += 8) {
            const __m256 a8 = load_f16x8_ps(va + i);
            const __m256 b8 = load_f16x8_ps(vb + i);
            dot0 = fmadd_ps(a8, b8, dot0);
            norm_a0 = fmadd_ps(a8, a8, norm_a0);
            norm_b0 = fmadd_ps(b8, b8, norm_b0);
        }
    }

    const __m256 dot = _mm256_add_ps(_mm256_add_ps(dot0, dot1), _mm256_add_ps(dot2, dot3));
    const __m256 norm_a = _mm256_add_ps(_mm256_add_ps(norm_a0, norm_a1), _mm256_add_ps(norm_a2, norm_a3));
    const __m256 norm_b = _mm256_add_ps(_mm256_add_ps(norm_b0, norm_b1), _mm256_add_ps(norm_b2, norm_b3));

    double dot_sum = hsum_ps_256(dot);
    double norm_a_sum = hsum_ps_256(norm_a);
    double norm_b_sum = hsum_ps_256(norm_b);

    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        const double bi = static_cast<double>(vb[i]);
        dot_sum += ai * bi;
        norm_a_sum += ai * ai;
        norm_b_sum += bi * bi;
    }

    return finalize_cosine_distance_avx2(dot_sum, norm_a_sum, norm_b_sum);
}

inline double ComputeCos_AVX2::dist_i16(const uint8_t *a, const uint8_t *b, size_t dim) {
    const int16_t *va = reinterpret_cast<const int16_t *>(a);
    const int16_t *vb = reinterpret_cast<const int16_t *>(b);
    __m256i dot_acc = _mm256_setzero_si256();
    __m256i norm_a_acc = _mm256_setzero_si256();
    __m256i norm_b_acc = _mm256_setzero_si256();
    const bool aligned =
        (((reinterpret_cast<uintptr_t>(va) | reinterpret_cast<uintptr_t>(vb)) & (kAvx2VectorAlignment - 1u)) == 0u);

    size_t i = 0;
    if (aligned) {
        for (; i + 16 <= dim; i += 16) {
            const __m256i a16 = _mm256_load_si256(reinterpret_cast<const __m256i *>(va + i));
            const __m256i b16 = _mm256_load_si256(reinterpret_cast<const __m256i *>(vb + i));
            accumulate_i16_products_as_i64_cos(a16, b16, &dot_acc, &norm_a_acc, &norm_b_acc);
        }
    } else {
        for (; i + 16 <= dim; i += 16) {
            const __m256i a16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(va + i));
            const __m256i b16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(vb + i));
            accumulate_i16_products_as_i64_cos(a16, b16, &dot_acc, &norm_a_acc, &norm_b_acc);
        }
    }

    double dot = hsum_epi64_256_cos(dot_acc);
    double norm_a = hsum_epi64_256_cos(norm_a_acc);
    double norm_b = hsum_epi64_256_cos(norm_b_acc);

    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        const double bi = static_cast<double>(vb[i]);
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    return finalize_cosine_distance_avx2(dot, norm_a, norm_b);
}

#endif

} // namespace sketch2
