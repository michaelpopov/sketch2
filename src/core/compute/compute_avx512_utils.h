// Provides AVX-512 helper utilities shared by the vectorized compute implementations.

#pragma once
#include "core/utils/arch_detection.h"

#if SKETCH_HAS_AVX512
#include <immintrin.h>

namespace sketch2 {

SKETCH_AVX512F_TARGET inline double hsum_ps_512(__m512 v) {
    return static_cast<double>(_mm512_reduce_add_ps(v));
}

SKETCH_AVX512F_TARGET inline double hsum_epi64_512(__m512i v) {
    return static_cast<double>(_mm512_reduce_add_epi64(v));
}

// AVX-512F provides fused multiply-add on 16 f32 lanes, which is the basic
// accumulation primitive used by the L2 and cosine kernels.
SKETCH_AVX512F_TARGET inline __m512 fmadd_ps_512(__m512 a, __m512 b, __m512 acc) {
    return _mm512_fmadd_ps(a, b, acc);
}

// Clearing the sign bit with integer logic keeps the AVX-512F backend from
// accidentally depending on AVX-512DQ just to compute absolute values.
SKETCH_AVX512F_TARGET inline __m512 abs_ps_512(__m512 v) {
    const __m512i mask = _mm512_set1_epi32(0x7fffffff);
    return _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(v), mask));
}

// AVX-512F can widen 16 packed half values into one 512-bit float vector
// without requiring AVX512-FP16 arithmetic in the kernel itself.
SKETCH_AVX512F_TARGET inline __m512 load_f16x16_ps(const void *ptr) {
    return _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr)));
}

SKETCH_AVX512F_TARGET inline __m512i load_i16x16_i32(const void *ptr) {
    return _mm512_cvtepi16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr)));
}

// Non-negative 32-bit values can be widened into 64-bit lanes by interleaving
// them with zeros before the integer accumulation step.
SKETCH_AVX512F_TARGET inline void accumulate_u32_as_i64(__m512i v32, __m512i zero,
        __m512i *acc_lo, __m512i *acc_hi) {
    *acc_lo = _mm512_add_epi64(*acc_lo, _mm512_unpacklo_epi32(v32, zero));
    *acc_hi = _mm512_add_epi64(*acc_hi, _mm512_unpackhi_epi32(v32, zero));
}

// Signed 32-bit products need sign extension before they are merged into the
// exact 64-bit dot-product accumulation.
SKETCH_AVX512F_TARGET inline void accumulate_i32_as_i64(__m512i v32, __m512i *acc_lo, __m512i *acc_hi) {
    const __m512i sign = _mm512_srai_epi32(v32, 31);
    *acc_lo = _mm512_add_epi64(*acc_lo, _mm512_unpacklo_epi32(v32, sign));
    *acc_hi = _mm512_add_epi64(*acc_hi, _mm512_unpackhi_epi32(v32, sign));
}

} // namespace sketch2

#endif
