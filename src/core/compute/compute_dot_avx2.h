// Implements AVX2-optimized DOT-distance kernels.

#pragma once
#include "core/compute/compute_avx2_utils.h"
#include "core/compute/compute.h"
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace sketch2 {

// Computes DOT (Manhattan) distance between two vectors.
// ComputeDOT_AVX2 exists to provide AVX2-specialized DOT kernels for x86 scan
// workloads. It exposes typed entry points that match the portable DOT interface.
class ComputeDOT_AVX2 {
public:
    SKETCH_AVX2_TARGET static double dist_f32(const uint8_t *a, const uint8_t *b, size_t dim);
    SKETCH_AVX2_TARGET static double dist_f16(const uint8_t *a, const uint8_t *b, size_t dim);
    SKETCH_AVX2_TARGET static double dist_i16(const uint8_t *a, const uint8_t *b, size_t dim);

private:
    SKETCH_AVX2_TARGET static double dist_f32_8(const uint8_t *a, const uint8_t *b, size_t dim);
    SKETCH_AVX2_TARGET static double dist_i16_16(const uint8_t *a, const uint8_t *b, size_t dim);
};

#if defined(SKETCH_ENABLE_AVX2) && SKETCH_ENABLE_AVX2 && (defined(__x86_64__) || defined(__i386__))

SKETCH_AVX2_TARGET inline double ComputeDOT_AVX2::dist_f32(const uint8_t *a, const uint8_t *b, size_t dim) {
    if (dim % 8 == 0) {
        return dist_f32_8(a, b, dim);
    }

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

        acc0 = fmadd_ps(a0, b0, acc0);
        acc1 = fmadd_ps(a1, b1, acc1);
        acc2 = fmadd_ps(a2, b2, acc2);
        acc3 = fmadd_ps(a3, b3, acc3);
    }
    for (; i + 8 <= dim; i += 8) {
        const __m256 a8 = _mm256_loadu_ps(va + i);
        const __m256 b8 = _mm256_loadu_ps(vb + i);
        acc0 = fmadd_ps(a8, b8, acc0);
    }

    const __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    double sum = hsum_ps_256(acc);

    for (; i < dim; ++i) {
        sum += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
    }
    return sum;
}

SKETCH_AVX2_TARGET inline double ComputeDOT_AVX2::dist_f32_8(const uint8_t *a, const uint8_t *b, size_t dim) {
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

        acc0 = fmadd_ps(a0, b0, acc0);
        acc1 = fmadd_ps(a1, b1, acc1);
        acc2 = fmadd_ps(a2, b2, acc2);
        acc3 = fmadd_ps(a3, b3, acc3);
    }
    for (; i < dim; i += 8) {
        const __m256 a8 = _mm256_loadu_ps(va + i);
        const __m256 b8 = _mm256_loadu_ps(vb + i);
        acc0 = fmadd_ps(a8, b8, acc0);
    }

    const __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    return hsum_ps_256(acc);
}

SKETCH_AVX2_TARGET inline double ComputeDOT_AVX2::dist_f16(const uint8_t *a, const uint8_t *b, size_t dim) {
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

        acc0 = fmadd_ps(a0, b0, acc0);
        acc1 = fmadd_ps(a1, b1, acc1);
        acc2 = fmadd_ps(a2, b2, acc2);
        acc3 = fmadd_ps(a3, b3, acc3);
    }
    for (; i + 8 <= dim; i += 8) {
        const __m256 a8 = load_f16x8_ps(va + i);
        const __m256 b8 = load_f16x8_ps(vb + i);
        acc0 = fmadd_ps(a8, b8, acc0);
    }

    const __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    double sum = hsum_ps_256(acc);
    for (; i < dim; ++i) {
        sum += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
    }
    return sum;
}

SKETCH_AVX2_TARGET inline double ComputeDOT_AVX2::dist_i16(const uint8_t *a, const uint8_t *b, size_t dim) {
    if (dim % 16 == 0) {
        return dist_i16_16(a, b, dim);
    }

    const int16_t *va = reinterpret_cast<const int16_t *>(a);
    const int16_t *vb = reinterpret_cast<const int16_t *>(b);
    __m256i acc_lo = _mm256_setzero_si256();
    __m256i acc_hi = _mm256_setzero_si256();

    size_t i = 0;
    for (; i + 32 <= dim; i += 32) {
        const __m256i a16_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(va + i));
        const __m256i b16_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(vb + i));
        const __m256i a16_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(va + i + 16));
        const __m256i b16_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(vb + i + 16));

        accumulate_i32_as_i64(_mm256_madd_epi16(a16_0, b16_0), &acc_lo, &acc_hi);
        accumulate_i32_as_i64(_mm256_madd_epi16(a16_1, b16_1), &acc_lo, &acc_hi);
    }
    for (; i + 16 <= dim; i += 16) {
        const __m256i a16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(va + i));
        const __m256i b16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(vb + i));
        accumulate_i32_as_i64(_mm256_madd_epi16(a16, b16), &acc_lo, &acc_hi);
    }

    double sum = hsum_epi64_256(_mm256_add_epi64(acc_lo, acc_hi));

    for (; i < dim; ++i) {
        sum += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
    }
    return sum;
}

SKETCH_AVX2_TARGET inline double ComputeDOT_AVX2::dist_i16_16(const uint8_t *a, const uint8_t *b, size_t dim) {
    const int16_t *va = reinterpret_cast<const int16_t *>(a);
    const int16_t *vb = reinterpret_cast<const int16_t *>(b);
    __m256i acc_lo = _mm256_setzero_si256();
    __m256i acc_hi = _mm256_setzero_si256();

    size_t i = 0;
    for (; i + 32 <= dim; i += 32) {
        const __m256i a16_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(va + i));
        const __m256i b16_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(vb + i));
        const __m256i a16_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(va + i + 16));
        const __m256i b16_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(vb + i + 16));

        accumulate_i32_as_i64(_mm256_madd_epi16(a16_0, b16_0), &acc_lo, &acc_hi);
        accumulate_i32_as_i64(_mm256_madd_epi16(a16_1, b16_1), &acc_lo, &acc_hi);
    }
    for (; i < dim; i += 16) {
        const __m256i a16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(va + i));
        const __m256i b16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(vb + i));
        accumulate_i32_as_i64(_mm256_madd_epi16(a16, b16), &acc_lo, &acc_hi);
    }

    return hsum_epi64_256(_mm256_add_epi64(acc_lo, acc_hi));
}

#endif

} // namespace sketch2
