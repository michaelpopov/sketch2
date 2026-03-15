// Implements AVX-512F-optimized L2-distance kernels.

#pragma once
#include "core/compute/compute_avx512_utils.h"
#include "core/compute/compute.h"
#include <cstdint>

namespace sketch2 {

// ComputeL2_AVX512 exists to provide 512-bit float kernels for squared-L2
// distance. The i16 path stays native by widening to i32 lanes and then
// accumulating exact 64-bit sums.
class ComputeL2_AVX512 {
public:
    SKETCH_AVX512F_TARGET static double dist_f32(const uint8_t *a, const uint8_t *b, size_t dim);
    SKETCH_AVX512F_TARGET static double dist_f16(const uint8_t *a, const uint8_t *b, size_t dim);
    SKETCH_AVX512F_TARGET static double dist_i16(const uint8_t *a, const uint8_t *b, size_t dim);
};

// ComputeL2_AVX512_VNNI keeps a distinct runtime backend entrypoint for CPUs
// that advertise VNNI, while reusing the same AVX-512F kernels.
class ComputeL2_AVX512_VNNI {
public:
    SKETCH_AVX512VNNI_TARGET static double dist_f32(const uint8_t *a, const uint8_t *b, size_t dim);
    SKETCH_AVX512VNNI_TARGET static double dist_f16(const uint8_t *a, const uint8_t *b, size_t dim);
    SKETCH_AVX512VNNI_TARGET static double dist_i16(const uint8_t *a, const uint8_t *b, size_t dim);
};

#if ((defined(SKETCH_ENABLE_AVX512F) && SKETCH_ENABLE_AVX512F) || \
     (defined(SKETCH_ENABLE_AVX512VNNI) && SKETCH_ENABLE_AVX512VNNI)) && \
    (defined(__x86_64__) || defined(__i386__))

// The AVX-512 f32 kernel mirrors the AVX2 structure but doubles the vector
// width so each loop iteration consumes 16 floats per register.
SKETCH_AVX512F_TARGET inline double ComputeL2_AVX512::dist_f32(const uint8_t *a, const uint8_t *b, size_t dim) {
    const float *va = reinterpret_cast<const float *>(a);
    const float *vb = reinterpret_cast<const float *>(b);
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();

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

        const __m512 d0 = _mm512_sub_ps(a0, b0);
        const __m512 d1 = _mm512_sub_ps(a1, b1);
        const __m512 d2 = _mm512_sub_ps(a2, b2);
        const __m512 d3 = _mm512_sub_ps(a3, b3);
        acc0 = fmadd_ps_512(d0, d0, acc0);
        acc1 = fmadd_ps_512(d1, d1, acc1);
        acc2 = fmadd_ps_512(d2, d2, acc2);
        acc3 = fmadd_ps_512(d3, d3, acc3);
    }
    for (; i + 16 <= dim; i += 16) {
        const __m512 d = _mm512_sub_ps(_mm512_loadu_ps(va + i), _mm512_loadu_ps(vb + i));
        acc0 = fmadd_ps_512(d, d, acc0);
    }

    const __m512 acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    double sum = hsum_ps_512(acc);
    for (; i < dim; ++i) {
        const double d = static_cast<double>(va[i]) - static_cast<double>(vb[i]);
        sum += d * d;
    }
    return sum;
}

// The AVX-512 f16 kernel widens 16 half values at a time and keeps the rest of
// the L2 arithmetic identical to the established f16 -> f32 model.
SKETCH_AVX512F_TARGET inline double ComputeL2_AVX512::dist_f16(const uint8_t *a, const uint8_t *b, size_t dim) {
    const float16 *va = reinterpret_cast<const float16 *>(a);
    const float16 *vb = reinterpret_cast<const float16 *>(b);
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();

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

        const __m512 d0 = _mm512_sub_ps(a0, b0);
        const __m512 d1 = _mm512_sub_ps(a1, b1);
        const __m512 d2 = _mm512_sub_ps(a2, b2);
        const __m512 d3 = _mm512_sub_ps(a3, b3);
        acc0 = fmadd_ps_512(d0, d0, acc0);
        acc1 = fmadd_ps_512(d1, d1, acc1);
        acc2 = fmadd_ps_512(d2, d2, acc2);
        acc3 = fmadd_ps_512(d3, d3, acc3);
    }
    for (; i + 16 <= dim; i += 16) {
        const __m512 d = _mm512_sub_ps(load_f16x16_ps(va + i), load_f16x16_ps(vb + i));
        acc0 = fmadd_ps_512(d, d, acc0);
    }

    const __m512 acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    double sum = hsum_ps_512(acc);
    for (; i < dim; ++i) {
        const double d = static_cast<double>(va[i]) - static_cast<double>(vb[i]);
        sum += d * d;
    }
    return sum;
}

// Squared-L2 needs exact 64-bit accumulation because the squared 17-bit diff
// can exceed signed 32-bit even though it still fits in 32 bits unsigned.
SKETCH_AVX512F_TARGET inline double ComputeL2_AVX512::dist_i16(const uint8_t *a, const uint8_t *b, size_t dim) {
    const int16_t *va = reinterpret_cast<const int16_t *>(a);
    const int16_t *vb = reinterpret_cast<const int16_t *>(b);
    const __m512i zero = _mm512_setzero_si512();
    __m512i acc_lo = _mm512_setzero_si512();
    __m512i acc_hi = _mm512_setzero_si512();

    size_t i = 0;
    for (; i + 32 <= dim; i += 32) {
        const __m512i a0 = load_i16x16_i32(va + i);
        const __m512i b0 = load_i16x16_i32(vb + i);
        const __m512i a1 = load_i16x16_i32(va + i + 16);
        const __m512i b1 = load_i16x16_i32(vb + i + 16);
        const __m512i d0 = _mm512_sub_epi32(a0, b0);
        const __m512i d1 = _mm512_sub_epi32(a1, b1);

        accumulate_u32_as_i64(_mm512_mullo_epi32(d0, d0), zero, &acc_lo, &acc_hi);
        accumulate_u32_as_i64(_mm512_mullo_epi32(d1, d1), zero, &acc_lo, &acc_hi);
    }
    for (; i + 16 <= dim; i += 16) {
        const __m512i a0 = load_i16x16_i32(va + i);
        const __m512i b0 = load_i16x16_i32(vb + i);
        const __m512i d0 = _mm512_sub_epi32(a0, b0);

        accumulate_u32_as_i64(_mm512_mullo_epi32(d0, d0), zero, &acc_lo, &acc_hi);
    }

    double sum = hsum_epi64_512(_mm512_add_epi64(acc_lo, acc_hi));
    for (; i < dim; ++i) {
        const int64_t d = static_cast<int64_t>(va[i]) - static_cast<int64_t>(vb[i]);
        sum += static_cast<double>(d * d);
    }
    return sum;
}

#if defined(SKETCH_ENABLE_AVX512VNNI) && SKETCH_ENABLE_AVX512VNNI && (defined(__x86_64__) || defined(__i386__))

SKETCH_AVX512VNNI_TARGET inline double ComputeL2_AVX512_VNNI::dist_f32(const uint8_t *a, const uint8_t *b, size_t dim) {
    return ComputeL2_AVX512::dist_f32(a, b, dim);
}

SKETCH_AVX512VNNI_TARGET inline double ComputeL2_AVX512_VNNI::dist_f16(const uint8_t *a, const uint8_t *b, size_t dim) {
    return ComputeL2_AVX512::dist_f16(a, b, dim);
}

SKETCH_AVX512VNNI_TARGET inline double ComputeL2_AVX512_VNNI::dist_i16(const uint8_t *a, const uint8_t *b, size_t dim) {
    return ComputeL2_AVX512::dist_i16(a, b, dim);
}

#endif // SKETCH_ENABLE_AVX512VNNI

#endif // SKETCH_ENABLE_AVX512F || SKETCH_ENABLE_AVX512VNNI

} // namespace sketch2
