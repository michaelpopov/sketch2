// Provides AVX2 helper utilities shared by the vectorized compute implementations.

#pragma once
#include "core/utils/arch_detection.h"

#if SKETCH_HAS_AVX2
#include <immintrin.h>

namespace sketch2 {

SKETCH_AVX2_TARGET inline double hsum_ps_256(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    const __m128 sum_hi = _mm_movehl_ps(sum, sum);
    sum = _mm_add_ps(sum, sum_hi);
    const __m128 sum_shuf = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 1, 1, 1));
    sum = _mm_add_ss(sum, sum_shuf);
    return static_cast<double>(_mm_cvtss_f32(sum));
}

SKETCH_AVX2_TARGET inline double hsum_epi64_256(__m256i v) {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    const __m128i sum = _mm_add_epi64(lo, hi);
    alignas(16) int64_t lanes[2];
    _mm_store_si128(reinterpret_cast<__m128i *>(lanes), sum);
    return static_cast<double>(lanes[0]) + static_cast<double>(lanes[1]);
}

// AVX2 kernels are targeted with FMA, so use fused multiply-add directly.
SKETCH_AVX2_TARGET inline __m256 fmadd_ps(__m256 a, __m256 b, __m256 acc) {
    return _mm256_fmadd_ps(a, b, acc);
}

SKETCH_AVX2_TARGET inline __m256 load_f16x8_ps(const void *ptr) {
    return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr)));
}

} // namespace sketch2

#endif
