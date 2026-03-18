// Implements AVX2-optimized L2-distance kernels.

#pragma once
#include "core/compute/compute_avx2_utils.h"
#include "core/compute/compute.h"
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace sketch2 {

// Computes squared L2 distance between two vectors.
// ComputeL2_AVX2 exists to provide AVX2-specialized squared-L2 kernels for x86
// scan workloads. It exposes typed entry points compatible with the generic dispatcher.
class ComputeL2_AVX2 {
public:
    SKETCH_AVX2_TARGET static double dist_f32(const uint8_t *a, const uint8_t *b, size_t dim);
    SKETCH_AVX2_TARGET static double dist_f16(const uint8_t *a, const uint8_t *b, size_t dim);
    SKETCH_AVX2_TARGET static double dist_i16(const uint8_t *a, const uint8_t *b, size_t dim);
};

#if defined(SKETCH_ENABLE_AVX2) && SKETCH_ENABLE_AVX2 && (defined(__x86_64__) || defined(__i386__))

SKETCH_AVX2_TARGET inline __m256i accumulate_squared_i32_as_i64_l2(__m256i acc, __m256i diff32) {
    const __m256i odd32 = _mm256_shuffle_epi32(diff32, _MM_SHUFFLE(3, 3, 1, 1));
    const __m256i even_sq64 = _mm256_mul_epi32(diff32, diff32);
    const __m256i odd_sq64 = _mm256_mul_epi32(odd32, odd32);
    return _mm256_add_epi64(acc, _mm256_add_epi64(even_sq64, odd_sq64));
}

SKETCH_AVX2_TARGET inline __m256i accumulate_squared_i16_as_i64_l2(__m256i acc, __m256i a16, __m256i b16) {
    const __m128i a_lo16 = _mm256_castsi256_si128(a16);
    const __m128i a_hi16 = _mm256_extracti128_si256(a16, 1);
    const __m128i b_lo16 = _mm256_castsi256_si128(b16);
    const __m128i b_hi16 = _mm256_extracti128_si256(b16, 1);

    const __m256i d_lo32 = _mm256_sub_epi32(_mm256_cvtepi16_epi32(a_lo16), _mm256_cvtepi16_epi32(b_lo16));
    const __m256i d_hi32 = _mm256_sub_epi32(_mm256_cvtepi16_epi32(a_hi16), _mm256_cvtepi16_epi32(b_hi16));

    acc = accumulate_squared_i32_as_i64_l2(acc, d_lo32);
    acc = accumulate_squared_i32_as_i64_l2(acc, d_hi32);
    return acc;
}

SKETCH_AVX2_TARGET inline double ComputeL2_AVX2::dist_f32(const uint8_t *a, const uint8_t *b, size_t dim) {
    const float *va = reinterpret_cast<const float *>(a);
    const float *vb = reinterpret_cast<const float *>(b);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    size_t i = 0;
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
        acc0 = fmadd_ps(d0, d0, acc0);
        acc1 = fmadd_ps(d1, d1, acc1);
        acc2 = fmadd_ps(d2, d2, acc2);
        acc3 = fmadd_ps(d3, d3, acc3);
    }
    for (; i + 8 <= dim; i += 8) {
        const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(va + i), _mm256_loadu_ps(vb + i));
        acc0 = fmadd_ps(d, d, acc0);
    }

    const __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    double sum = hsum_ps_256(acc);
    for (; i < dim; ++i) {
        const double d = static_cast<double>(va[i]) - static_cast<double>(vb[i]);
        sum += d * d;
    }
    return sum;
}

SKETCH_AVX2_TARGET inline double ComputeL2_AVX2::dist_f16(const uint8_t *a, const uint8_t *b, size_t dim) {
    const float16 *va = reinterpret_cast<const float16 *>(a);
    const float16 *vb = reinterpret_cast<const float16 *>(b);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 32 <= dim; i += 32) {
        const __m256 a0 = load_f16x8_ps(va + i);
        const __m256 b0 = load_f16x8_ps(vb + i);
        const __m256 a1 = load_f16x8_ps(va + i + 8);
        const __m256 b1 = load_f16x8_ps(vb + i + 8);
        const __m256 a2 = load_f16x8_ps(va + i + 16);
        const __m256 b2 = load_f16x8_ps(vb + i + 16);
        const __m256 a3 = load_f16x8_ps(va + i + 24);
        const __m256 b3 = load_f16x8_ps(vb + i + 24);

        const __m256 d0 = _mm256_sub_ps(a0, b0);
        const __m256 d1 = _mm256_sub_ps(a1, b1);
        const __m256 d2 = _mm256_sub_ps(a2, b2);
        const __m256 d3 = _mm256_sub_ps(a3, b3);
        acc0 = fmadd_ps(d0, d0, acc0);
        acc1 = fmadd_ps(d1, d1, acc1);
        acc2 = fmadd_ps(d2, d2, acc2);
        acc3 = fmadd_ps(d3, d3, acc3);
    }
    for (; i + 8 <= dim; i += 8) {
        const __m256 d = _mm256_sub_ps(load_f16x8_ps(va + i), load_f16x8_ps(vb + i));
        acc0 = fmadd_ps(d, d, acc0);
    }

    const __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    double sum = hsum_ps_256(acc);
    for (; i < dim; ++i) {
        const double d = static_cast<double>(va[i]) - static_cast<double>(vb[i]);
        sum += d * d;
    }
    return sum;
}

SKETCH_AVX2_TARGET inline double ComputeL2_AVX2::dist_i16(const uint8_t *a, const uint8_t *b, size_t dim) {
    const int16_t *va = reinterpret_cast<const int16_t *>(a);
    const int16_t *vb = reinterpret_cast<const int16_t *>(b);
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();

    size_t i = 0;
    for (; i + 32 <= dim; i += 32) {
        const __m256i a16_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(va + i));
        const __m256i b16_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(vb + i));
        const __m256i a16_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(va + i + 16));
        const __m256i b16_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(vb + i + 16));

        acc0 = accumulate_squared_i16_as_i64_l2(acc0, a16_0, b16_0);
        acc1 = accumulate_squared_i16_as_i64_l2(acc1, a16_1, b16_1);
    }
    for (; i + 16 <= dim; i += 16) {
        const __m256i a16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(va + i));
        const __m256i b16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(vb + i));
        acc0 = accumulate_squared_i16_as_i64_l2(acc0, a16, b16);
    }

    double sum = hsum_epi64_256(_mm256_add_epi64(acc0, acc1));
    for (; i < dim; ++i) {
        const int64_t d = static_cast<int64_t>(va[i]) - static_cast<int64_t>(vb[i]);
        sum += static_cast<double>(d * d);
    }
    return sum;
}

#endif

} // namespace sketch2
