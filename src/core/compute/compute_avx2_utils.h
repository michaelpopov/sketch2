// Provides AVX2 helper utilities shared by the vectorized compute implementations.

#pragma once

#ifndef SKETCH_AVX2_TARGET
#define SKETCH_AVX2_TARGET
#endif

#if defined(SKETCH_ENABLE_AVX2) && SKETCH_ENABLE_AVX2 && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>

#if defined(__GNUC__) || defined(__clang__)
#undef SKETCH_AVX2_TARGET
#define SKETCH_AVX2_TARGET __attribute__((target("avx2,f16c,fma")))
#endif

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

// AVX2 kernels are targeted with FMA, so use fused multiply-add directly.
SKETCH_AVX2_TARGET inline __m256 fmadd_ps(__m256 a, __m256 b, __m256 acc) {
    return _mm256_fmadd_ps(a, b, acc);
}

SKETCH_AVX2_TARGET inline __m256 load_f16x8_ps(const void *ptr) {
    return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr)));
}

} // namespace sketch2

#endif
