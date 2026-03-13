#pragma once

#if defined(__AVX2__)
#include <immintrin.h>

namespace sketch2 {

// Byte alignment required for aligned AVX2 256-bit loads/stores.
inline constexpr uintptr_t kAvx2VectorAlignment = 32u;

// Byte alignment required for aligned 128-bit half-precision loads before F16C widening.
inline constexpr uintptr_t kHalfVectorAlignment = 16u;

inline double hsum_ps_256(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    const __m128 sum_hi = _mm_movehl_ps(sum, sum);
    sum = _mm_add_ps(sum, sum_hi);
    const __m128 sum_shuf = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 1, 1, 1));
    sum = _mm_add_ss(sum, sum_shuf);
    return static_cast<double>(_mm_cvtss_f32(sum));
}

// Keep AVX2 kernels buildable without FMA while still using fused multiply-add when enabled.
inline __m256 fmadd_ps(__m256 a, __m256 b, __m256 acc) {
#if defined(__FMA__)
    return _mm256_fmadd_ps(a, b, acc);
#else
    return _mm256_add_ps(acc, _mm256_mul_ps(a, b));
#endif
}

inline __m256 load_f16x8_ps(const void *ptr) {
    return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr)));
}

inline __m256 load_f16x8_ps_aligned(const void *ptr) {
    return _mm256_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i *>(ptr)));
}

} // namespace sketch2

#endif
