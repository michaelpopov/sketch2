// Implements AVX-512F-optimized DOT-distance kernels.

#pragma once
#include "core/compute/compute_avx512_utils.h"
#include "core/compute/compute.h"
#include <cmath>
#include <cstdint>

namespace sketch2 {

// ComputeDOT_AVX512 exists to provide 512-bit float kernels for x86 hosts that
// support AVX-512F. The i16 path also stays native by widening to i32 lanes and
// accumulating in i64 so full-range sums remain exact.
class ComputeDOT_AVX512 {
public:
    SKETCH_AVX512F_TARGET static double dist_f32(const uint8_t *a, const uint8_t *b, size_t dim);
    SKETCH_AVX512F_TARGET static double dist_f16(const uint8_t *a, const uint8_t *b, size_t dim);
    SKETCH_AVX512F_TARGET static double dist_i16(const uint8_t *a, const uint8_t *b, size_t dim);
};

// ComputeDOT_AVX512_VNNI keeps a distinct runtime backend entrypoint for CPUs
// that advertise VNNI, while reusing the same AVX-512F kernels.
class ComputeDOT_AVX512_VNNI {
public:
    SKETCH_AVX512VNNI_TARGET static double dist_f32(const uint8_t *a, const uint8_t *b, size_t dim);
    SKETCH_AVX512VNNI_TARGET static double dist_f16(const uint8_t *a, const uint8_t *b, size_t dim);
    SKETCH_AVX512VNNI_TARGET static double dist_i16(const uint8_t *a, const uint8_t *b, size_t dim);
};

#if ((defined(SKETCH_ENABLE_AVX512F) && SKETCH_ENABLE_AVX512F) || \
     (defined(SKETCH_ENABLE_AVX512VNNI) && SKETCH_ENABLE_AVX512VNNI)) && \
    (defined(__x86_64__) || defined(__i386__))

// The f32 kernel keeps four independent accumulators so wide loads can overlap
// latency while the final scalar tail preserves the existing semantics.
SKETCH_AVX512F_TARGET inline double ComputeDOT_AVX512::dist_f32(const uint8_t *a, const uint8_t *b, size_t dim) {
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

        acc0 = _mm512_fmadd_ps(a0, b0, acc0);
        acc1 = _mm512_fmadd_ps(a1, b1, acc1);
        acc2 = _mm512_fmadd_ps(a2, b2, acc2);
        acc3 = _mm512_fmadd_ps(a3, b3, acc3);
    }
    for (; i + 16 <= dim; i += 16) {
        const __m512 a16 = _mm512_loadu_ps(va + i);
        const __m512 b16 = _mm512_loadu_ps(vb + i);
        acc0 = _mm512_fmadd_ps(a16, b16, acc0);
    }

    const __m512 acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    double sum = hsum_ps_512(acc);
    for (; i < dim; ++i) {
        sum += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
    }
    return sum;
}

// The f16 kernel follows the current arithmetic model exactly: widen half
// inputs to f32 in registers, accumulate in f32 lanes, then finish in scalar.
SKETCH_AVX512F_TARGET inline double ComputeDOT_AVX512::dist_f16(const uint8_t *a, const uint8_t *b, size_t dim) {
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

        acc0 = _mm512_fmadd_ps(a0, b0, acc0);
        acc1 = _mm512_fmadd_ps(a1, b1, acc1);
        acc2 = _mm512_fmadd_ps(a2, b2, acc2);
        acc3 = _mm512_fmadd_ps(a3, b3, acc3);
    }
    for (; i + 16 <= dim; i += 16) {
        const __m512 a16 = load_f16x16_ps(va + i);
        const __m512 b16 = load_f16x16_ps(vb + i);
        acc0 = _mm512_fmadd_ps(a16, b16, acc0);
    }

    const __m512 acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    double sum = hsum_ps_512(acc);
    for (; i < dim; ++i) {
        sum += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
    }
    return sum;
}

// The i16 path deliberately widens into 32-bit lanes and then accumulates in
// 32-bit integers.
SKETCH_AVX512F_TARGET inline double ComputeDOT_AVX512::dist_i16(const uint8_t *a, const uint8_t *b, size_t dim) {
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

    double sum = hsum_epi64_512(_mm512_add_epi64(acc_lo, acc_hi));
    for (; i < dim; ++i) {
        sum += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
    }
    return sum;
}

#if defined(SKETCH_ENABLE_AVX512VNNI) && SKETCH_ENABLE_AVX512VNNI && (defined(__x86_64__) || defined(__i386__))

SKETCH_AVX512VNNI_TARGET inline double ComputeDOT_AVX512_VNNI::dist_f32(const uint8_t *a, const uint8_t *b, size_t dim) {
    return ComputeDOT_AVX512::dist_f32(a, b, dim);
}

SKETCH_AVX512VNNI_TARGET inline double ComputeDOT_AVX512_VNNI::dist_f16(const uint8_t *a, const uint8_t *b, size_t dim) {
    return ComputeDOT_AVX512::dist_f16(a, b, dim);
}

SKETCH_AVX512VNNI_TARGET inline double ComputeDOT_AVX512_VNNI::dist_i16(const uint8_t *a, const uint8_t *b, size_t dim) {
    const int16_t *va = reinterpret_cast<const int16_t *>(a);
    const int16_t *vb = reinterpret_cast<const int16_t *>(b);
    __m512i acc_lo = _mm512_setzero_si512();
    __m512i acc_hi = _mm512_setzero_si512();

    size_t i = 0;
    for (; i + 64 <= dim; i += 64) {
        const __m512i a0 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(va + i));
        const __m512i b0 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(vb + i));
        const __m512i a1 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(va + i + 32));
        const __m512i b1 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(vb + i + 32));

        accumulate_i32_as_i64(_mm512_madd_epi16(a0, b0), &acc_lo, &acc_hi);
        accumulate_i32_as_i64(_mm512_madd_epi16(a1, b1), &acc_lo, &acc_hi);
    }
    for (; i + 32 <= dim; i += 32) {
        const __m512i a0 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(va + i));
        const __m512i b0 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(vb + i));
        accumulate_i32_as_i64(_mm512_madd_epi16(a0, b0), &acc_lo, &acc_hi);
    }

    double sum = hsum_epi64_512(_mm512_add_epi64(acc_lo, acc_hi));
    for (; i < dim; ++i) {
        sum += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
    }
    return sum;
}

#endif // SKETCH_ENABLE_AVX512VNNI

#endif // SKETCH_ENABLE_AVX512F || SKETCH_ENABLE_AVX512VNNI

} // namespace sketch2
