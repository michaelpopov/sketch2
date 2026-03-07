#pragma once

#if defined(__AVX2__)
#include <immintrin.h>

namespace sketch2 {

inline double hsum_ps_256(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return static_cast<double>(_mm_cvtss_f32(sum));
}

} // namespace sketch2

#endif
