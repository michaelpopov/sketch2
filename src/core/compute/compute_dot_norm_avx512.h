// Implements AVX-512-optimized shared dot-product and squared-norm kernels.

#pragma once

#include "core/compute/compute_avx512_utils.h"
#include "core/compute/compute.h"

#include <cstdint>

namespace sketch2 {

// ComputeDotNorm_AVX512 owns the AVX-512-specialized shared dot/norm kernels
// used by both cosine and L2 scan paths.
class ComputeDotNorm_AVX512 {
public:
    SKETCH_AVX512F_TARGET static double squared_norm_f32(const uint8_t *a, size_t dim);
    SKETCH_AVX512F_TARGET static double squared_norm_f16(const uint8_t *a, size_t dim);
    SKETCH_AVX512F_TARGET static double squared_norm_i16(const uint8_t *a, size_t dim);
    SKETCH_AVX512F_TARGET static double dot_f32(const uint8_t *a, const uint8_t *b, size_t dim);
    SKETCH_AVX512F_TARGET static double dot_f16(const uint8_t *a, const uint8_t *b, size_t dim);
    SKETCH_AVX512F_TARGET static double dot_i16(const uint8_t *a, const uint8_t *b, size_t dim);
};

// ComputeDotNorm_AVX512_VNNI keeps a distinct runtime backend entrypoint for
// CPUs that advertise VNNI, while reusing the same AVX-512F shared kernels.
class ComputeDotNorm_AVX512_VNNI {
public:
    SKETCH_AVX512VNNI_TARGET static double squared_norm_f32(const uint8_t *a, size_t dim);
    SKETCH_AVX512VNNI_TARGET static double squared_norm_f16(const uint8_t *a, size_t dim);
    SKETCH_AVX512VNNI_TARGET static double squared_norm_i16(const uint8_t *a, size_t dim);
    SKETCH_AVX512VNNI_TARGET static double dot_f32(const uint8_t *a, const uint8_t *b, size_t dim);
    SKETCH_AVX512VNNI_TARGET static double dot_f16(const uint8_t *a, const uint8_t *b, size_t dim);
    SKETCH_AVX512VNNI_TARGET static double dot_i16(const uint8_t *a, const uint8_t *b, size_t dim);
};

#if ((defined(SKETCH_ENABLE_AVX512F) && SKETCH_ENABLE_AVX512F) || \
     (defined(SKETCH_ENABLE_AVX512VNNI) && SKETCH_ENABLE_AVX512VNNI)) && \
    (defined(__x86_64__) || defined(__i386__))

SKETCH_AVX512F_TARGET inline double ComputeDotNorm_AVX512::squared_norm_f32(const uint8_t *a, size_t dim) {
    const float *va = reinterpret_cast<const float *>(a);
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 64 <= dim; i += 64) {
        const __m512 a0 = _mm512_loadu_ps(va + i);
        const __m512 a1 = _mm512_loadu_ps(va + i + 16);
        const __m512 a2 = _mm512_loadu_ps(va + i + 32);
        const __m512 a3 = _mm512_loadu_ps(va + i + 48);
        acc0 = fmadd_ps_512(a0, a0, acc0);
        acc1 = fmadd_ps_512(a1, a1, acc1);
        acc2 = fmadd_ps_512(a2, a2, acc2);
        acc3 = fmadd_ps_512(a3, a3, acc3);
    }
    for (; i + 16 <= dim; i += 16) {
        const __m512 a16 = _mm512_loadu_ps(va + i);
        acc0 = fmadd_ps_512(a16, a16, acc0);
    }

    const __m512 acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    double norm = hsum_ps_512(acc);
    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        norm += ai * ai;
    }
    return norm;
}

SKETCH_AVX512F_TARGET inline double ComputeDotNorm_AVX512::squared_norm_f16(const uint8_t *a, size_t dim) {
    const float16 *va = reinterpret_cast<const float16 *>(a);
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 64 <= dim; i += 64) {
        const __m512 a0 = load_f16x16_ps(va + i);
        const __m512 a1 = load_f16x16_ps(va + i + 16);
        const __m512 a2 = load_f16x16_ps(va + i + 32);
        const __m512 a3 = load_f16x16_ps(va + i + 48);
        acc0 = fmadd_ps_512(a0, a0, acc0);
        acc1 = fmadd_ps_512(a1, a1, acc1);
        acc2 = fmadd_ps_512(a2, a2, acc2);
        acc3 = fmadd_ps_512(a3, a3, acc3);
    }
    for (; i + 16 <= dim; i += 16) {
        const __m512 a16 = load_f16x16_ps(va + i);
        acc0 = fmadd_ps_512(a16, a16, acc0);
    }

    const __m512 acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    double norm = hsum_ps_512(acc);
    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        norm += ai * ai;
    }
    return norm;
}

SKETCH_AVX512F_TARGET inline double ComputeDotNorm_AVX512::dot_f32(const uint8_t *a, const uint8_t *b, size_t dim) {
    const float *va = reinterpret_cast<const float *>(a);
    const float *vb = reinterpret_cast<const float *>(b);
    __m512 dot0 = _mm512_setzero_ps();
    __m512 dot1 = _mm512_setzero_ps();
    __m512 dot2 = _mm512_setzero_ps();
    __m512 dot3 = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 64 <= dim; i += 64) {
        const __m512 a0 = _mm512_loadu_ps(va + i);
        const __m512 b0 = _mm512_loadu_ps(vb + i);
        const __m512 a1 = _mm512_loadu_ps(va + i + 16);
        const __m512 b1 = _mm512_loadu_ps(vb + i + 16);
        const __m512 a2 = _mm512_loadu_ps(va + i + 32);
        const __m512 b2 = _mm512_loadu_ps(vb + i + 32);
        const __m512 a3 = _mm512_loadu_ps(va + i + 48);
        const __m512 b3 = _mm512_loadu_ps(vb + i + 48);
        dot0 = fmadd_ps_512(a0, b0, dot0);
        dot1 = fmadd_ps_512(a1, b1, dot1);
        dot2 = fmadd_ps_512(a2, b2, dot2);
        dot3 = fmadd_ps_512(a3, b3, dot3);
    }
    for (; i + 16 <= dim; i += 16) {
        dot0 = fmadd_ps_512(_mm512_loadu_ps(va + i), _mm512_loadu_ps(vb + i), dot0);
    }

    const __m512 dot = _mm512_add_ps(_mm512_add_ps(dot0, dot1), _mm512_add_ps(dot2, dot3));
    double dot_sum = hsum_ps_512(dot);
    for (; i < dim; ++i) {
        dot_sum += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
    }
    return dot_sum;
}

SKETCH_AVX512F_TARGET inline double ComputeDotNorm_AVX512::dot_f16(const uint8_t *a, const uint8_t *b, size_t dim) {
    const float16 *va = reinterpret_cast<const float16 *>(a);
    const float16 *vb = reinterpret_cast<const float16 *>(b);
    __m512 dot0 = _mm512_setzero_ps();
    __m512 dot1 = _mm512_setzero_ps();
    __m512 dot2 = _mm512_setzero_ps();
    __m512 dot3 = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 64 <= dim; i += 64) {
        const __m512 a0 = load_f16x16_ps(va + i);
        const __m512 b0 = load_f16x16_ps(vb + i);
        const __m512 a1 = load_f16x16_ps(va + i + 16);
        const __m512 b1 = load_f16x16_ps(vb + i + 16);
        const __m512 a2 = load_f16x16_ps(va + i + 32);
        const __m512 b2 = load_f16x16_ps(vb + i + 32);
        const __m512 a3 = load_f16x16_ps(va + i + 48);
        const __m512 b3 = load_f16x16_ps(vb + i + 48);
        dot0 = fmadd_ps_512(a0, b0, dot0);
        dot1 = fmadd_ps_512(a1, b1, dot1);
        dot2 = fmadd_ps_512(a2, b2, dot2);
        dot3 = fmadd_ps_512(a3, b3, dot3);
    }
    for (; i + 16 <= dim; i += 16) {
        dot0 = fmadd_ps_512(load_f16x16_ps(va + i), load_f16x16_ps(vb + i), dot0);
    }

    const __m512 dot = _mm512_add_ps(_mm512_add_ps(dot0, dot1), _mm512_add_ps(dot2, dot3));
    double dot_sum = hsum_ps_512(dot);
    for (; i < dim; ++i) {
        dot_sum += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
    }
    return dot_sum;
}

SKETCH_AVX512F_TARGET inline double ComputeDotNorm_AVX512::squared_norm_i16(const uint8_t *a, size_t dim) {
    const int16_t *va = reinterpret_cast<const int16_t *>(a);
    const __m512i zero = _mm512_setzero_si512();
    __m512i acc_lo = _mm512_setzero_si512();
    __m512i acc_hi = _mm512_setzero_si512();

    size_t i = 0;
    for (; i + 32 <= dim; i += 32) {
        const __m512i a0 = load_i16x16_i32(va + i);
        const __m512i a1 = load_i16x16_i32(va + i + 16);
        accumulate_u32_as_i64(_mm512_mullo_epi32(a0, a0), zero, &acc_lo, &acc_hi);
        accumulate_u32_as_i64(_mm512_mullo_epi32(a1, a1), zero, &acc_lo, &acc_hi);
    }
    for (; i + 16 <= dim; i += 16) {
        const __m512i a0 = load_i16x16_i32(va + i);
        accumulate_u32_as_i64(_mm512_mullo_epi32(a0, a0), zero, &acc_lo, &acc_hi);
    }

    double norm = hsum_epi64_512(_mm512_add_epi64(acc_lo, acc_hi));
    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        norm += ai * ai;
    }
    return norm;
}

SKETCH_AVX512F_TARGET inline double ComputeDotNorm_AVX512::dot_i16(const uint8_t *a, const uint8_t *b, size_t dim) {
    const int16_t *va = reinterpret_cast<const int16_t *>(a);
    const int16_t *vb = reinterpret_cast<const int16_t *>(b);
    __m512i acc_lo = _mm512_setzero_si512();
    __m512i acc_hi = _mm512_setzero_si512();

    size_t i = 0;
    for (; i + 32 <= dim; i += 32) {
        const __m512i a0 = load_i16x16_i32(va + i);
        const __m512i b0 = load_i16x16_i32(vb + i);
        const __m512i a1 = load_i16x16_i32(va + i + 16);
        const __m512i b1 = load_i16x16_i32(vb + i + 16);
        accumulate_i32_as_i64(_mm512_mullo_epi32(a0, b0), &acc_lo, &acc_hi);
        accumulate_i32_as_i64(_mm512_mullo_epi32(a1, b1), &acc_lo, &acc_hi);
    }
    for (; i + 16 <= dim; i += 16) {
        const __m512i a0 = load_i16x16_i32(va + i);
        const __m512i b0 = load_i16x16_i32(vb + i);
        accumulate_i32_as_i64(_mm512_mullo_epi32(a0, b0), &acc_lo, &acc_hi);
    }

    double dot = hsum_epi64_512(_mm512_add_epi64(acc_lo, acc_hi));
    for (; i < dim; ++i) {
        dot += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
    }
    return dot;
}

#if defined(SKETCH_ENABLE_AVX512VNNI) && SKETCH_ENABLE_AVX512VNNI && (defined(__x86_64__) || defined(__i386__))

SKETCH_AVX512VNNI_TARGET inline double ComputeDotNorm_AVX512_VNNI::squared_norm_f32(const uint8_t *a, size_t dim) {
    return ComputeDotNorm_AVX512::squared_norm_f32(a, dim);
}

SKETCH_AVX512VNNI_TARGET inline double ComputeDotNorm_AVX512_VNNI::squared_norm_f16(const uint8_t *a, size_t dim) {
    return ComputeDotNorm_AVX512::squared_norm_f16(a, dim);
}

SKETCH_AVX512VNNI_TARGET inline double ComputeDotNorm_AVX512_VNNI::dot_f32(const uint8_t *a, const uint8_t *b, size_t dim) {
    return ComputeDotNorm_AVX512::dot_f32(a, b, dim);
}

SKETCH_AVX512VNNI_TARGET inline double ComputeDotNorm_AVX512_VNNI::dot_f16(const uint8_t *a, const uint8_t *b, size_t dim) {
    return ComputeDotNorm_AVX512::dot_f16(a, b, dim);
}

SKETCH_AVX512VNNI_TARGET inline double ComputeDotNorm_AVX512_VNNI::squared_norm_i16(const uint8_t *a, size_t dim) {
    return ComputeDotNorm_AVX512::squared_norm_i16(a, dim);
}

SKETCH_AVX512VNNI_TARGET inline double ComputeDotNorm_AVX512_VNNI::dot_i16(const uint8_t *a, const uint8_t *b, size_t dim) {
    return ComputeDotNorm_AVX512::dot_i16(a, b, dim);
}

#endif // SKETCH_ENABLE_AVX512VNNI

#endif // SKETCH_ENABLE_AVX512F || SKETCH_ENABLE_AVX512VNNI

} // namespace sketch2
