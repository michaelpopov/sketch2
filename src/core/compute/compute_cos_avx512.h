// Implements AVX-512F-optimized cosine-distance kernels.

#pragma once
#include "core/compute/compute_avx512_utils.h"
#include "core/compute/compute.h"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace sketch2 {

// ComputeCos_AVX512 exists to provide 512-bit float kernels for x86 cosine
// workloads. The i16 path also stays native by widening into i32 lanes and
// accumulating in i64 so full-range norms and dot products remain exact.
class ComputeCos_AVX512 {
public:
    SKETCH_AVX512F_TARGET static double dist_f32(const uint8_t *a, const uint8_t *b, size_t dim);
    SKETCH_AVX512F_TARGET static double dist_f16(const uint8_t *a, const uint8_t *b, size_t dim);
    SKETCH_AVX512F_TARGET static double dist_i16(const uint8_t *a, const uint8_t *b, size_t dim);
    SKETCH_AVX512F_TARGET static double dist_f32_with_query_norm(const uint8_t *a, const uint8_t *b, size_t dim, double query_norm_sq);
    SKETCH_AVX512F_TARGET static double dist_f16_with_query_norm(const uint8_t *a, const uint8_t *b, size_t dim, double query_norm_sq);
    SKETCH_AVX512F_TARGET static double dist_i16_with_query_norm(const uint8_t *a, const uint8_t *b, size_t dim, double query_norm_sq);
    SKETCH_AVX512F_TARGET static double squared_norm_f32(const uint8_t *a, size_t dim);
    SKETCH_AVX512F_TARGET static double squared_norm_f16(const uint8_t *a, size_t dim);
    SKETCH_AVX512F_TARGET static double squared_norm_i16(const uint8_t *a, size_t dim);
    SKETCH_AVX512F_TARGET static double dot_f32(const uint8_t *a, const uint8_t *b, size_t dim);
    SKETCH_AVX512F_TARGET static double dot_f16(const uint8_t *a, const uint8_t *b, size_t dim);
    SKETCH_AVX512F_TARGET static double dot_i16(const uint8_t *a, const uint8_t *b, size_t dim);
};

// ComputeCos_AVX512_VNNI keeps a distinct runtime backend entrypoint for CPUs
// that advertise VNNI, while reusing the same AVX-512F kernels.
class ComputeCos_AVX512_VNNI {
public:
    SKETCH_AVX512VNNI_TARGET static double dist_f32(const uint8_t *a, const uint8_t *b, size_t dim);
    SKETCH_AVX512VNNI_TARGET static double dist_f16(const uint8_t *a, const uint8_t *b, size_t dim);
    SKETCH_AVX512VNNI_TARGET static double dist_i16(const uint8_t *a, const uint8_t *b, size_t dim);
    SKETCH_AVX512VNNI_TARGET static double dist_f32_with_query_norm(const uint8_t *a, const uint8_t *b, size_t dim, double query_norm_sq);
    SKETCH_AVX512VNNI_TARGET static double dist_f16_with_query_norm(const uint8_t *a, const uint8_t *b, size_t dim, double query_norm_sq);
    SKETCH_AVX512VNNI_TARGET static double dist_i16_with_query_norm(const uint8_t *a, const uint8_t *b, size_t dim, double query_norm_sq);
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

SKETCH_AVX512F_TARGET inline double finalize_cosine_distance_avx512(double dot, double norm_a, double norm_b) {
    if (norm_a == 0.0 && norm_b == 0.0) {
        return 0.0;
    }
    if (norm_a == 0.0 || norm_b == 0.0) {
        return 1.0;
    }
    const double cosine = std::clamp(dot / std::sqrt(norm_a * norm_b), -1.0, 1.0);
    return 1.0 - cosine;
}

// The native f32 norm path uses 16-wide FMA accumulation so the scanner can
// keep query setup on the same backend family as the candidate loop.
SKETCH_AVX512F_TARGET inline double ComputeCos_AVX512::squared_norm_f32(const uint8_t *a, size_t dim) {
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

// The native f16 norm path still widens to f32 lanes first, preserving the
// current numerical model while using 16 half inputs per vector instruction.
SKETCH_AVX512F_TARGET inline double ComputeCos_AVX512::squared_norm_f16(const uint8_t *a, size_t dim) {
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

// The f32 dot path is the core cosine primitive used both directly and by the
// inverse-norm scanner path.
SKETCH_AVX512F_TARGET inline double ComputeCos_AVX512::dot_f32(const uint8_t *a, const uint8_t *b, size_t dim) {
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

// The f16 dot path widens the packed halves into one zmm register per chunk,
// then reuses the same 16-lane fused multiply-add accumulation as f32.
SKETCH_AVX512F_TARGET inline double ComputeCos_AVX512::dot_f16(const uint8_t *a, const uint8_t *b, size_t dim) {
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

SKETCH_AVX512F_TARGET inline double ComputeCos_AVX512::dist_f32(const uint8_t *a, const uint8_t *b, size_t dim) {
    return dist_f32_with_query_norm(a, b, dim, squared_norm_f32(b, dim));
}

SKETCH_AVX512F_TARGET inline double ComputeCos_AVX512::dist_f16(const uint8_t *a, const uint8_t *b, size_t dim) {
    return dist_f16_with_query_norm(a, b, dim, squared_norm_f16(b, dim));
}

// The query-norm f32 path computes the candidate norm and the dot product in
// one walk so cosine scans only touch each candidate vector once.
SKETCH_AVX512F_TARGET inline double ComputeCos_AVX512::dist_f32_with_query_norm(const uint8_t *a,
        const uint8_t *b, size_t dim, double query_norm_sq) {
    const float *va = reinterpret_cast<const float *>(a);
    const float *vb = reinterpret_cast<const float *>(b);
    __m512 dot0 = _mm512_setzero_ps();
    __m512 dot1 = _mm512_setzero_ps();
    __m512 dot2 = _mm512_setzero_ps();
    __m512 dot3 = _mm512_setzero_ps();
    __m512 norm_a0 = _mm512_setzero_ps();
    __m512 norm_a1 = _mm512_setzero_ps();
    __m512 norm_a2 = _mm512_setzero_ps();
    __m512 norm_a3 = _mm512_setzero_ps();

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
        norm_a0 = fmadd_ps_512(a0, a0, norm_a0);
        norm_a1 = fmadd_ps_512(a1, a1, norm_a1);
        norm_a2 = fmadd_ps_512(a2, a2, norm_a2);
        norm_a3 = fmadd_ps_512(a3, a3, norm_a3);
    }
    for (; i + 16 <= dim; i += 16) {
        const __m512 a16 = _mm512_loadu_ps(va + i);
        const __m512 b16 = _mm512_loadu_ps(vb + i);
        dot0 = fmadd_ps_512(a16, b16, dot0);
        norm_a0 = fmadd_ps_512(a16, a16, norm_a0);
    }

    const __m512 dot = _mm512_add_ps(_mm512_add_ps(dot0, dot1), _mm512_add_ps(dot2, dot3));
    const __m512 norm_a = _mm512_add_ps(_mm512_add_ps(norm_a0, norm_a1), _mm512_add_ps(norm_a2, norm_a3));
    double dot_sum = hsum_ps_512(dot);
    double norm_a_sum = hsum_ps_512(norm_a);

    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        const double bi = static_cast<double>(vb[i]);
        dot_sum += ai * bi;
        norm_a_sum += ai * ai;
    }

    return finalize_cosine_distance_avx512(dot_sum, norm_a_sum, query_norm_sq);
}

// The query-norm f16 path keeps exactly the same contract as the AVX2 and
// scalar paths: widen, accumulate in f32 lanes, then finalize in scalar.
SKETCH_AVX512F_TARGET inline double ComputeCos_AVX512::dist_f16_with_query_norm(const uint8_t *a,
        const uint8_t *b, size_t dim, double query_norm_sq) {
    const float16 *va = reinterpret_cast<const float16 *>(a);
    const float16 *vb = reinterpret_cast<const float16 *>(b);
    __m512 dot0 = _mm512_setzero_ps();
    __m512 dot1 = _mm512_setzero_ps();
    __m512 dot2 = _mm512_setzero_ps();
    __m512 dot3 = _mm512_setzero_ps();
    __m512 norm_a0 = _mm512_setzero_ps();
    __m512 norm_a1 = _mm512_setzero_ps();
    __m512 norm_a2 = _mm512_setzero_ps();
    __m512 norm_a3 = _mm512_setzero_ps();

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
        norm_a0 = fmadd_ps_512(a0, a0, norm_a0);
        norm_a1 = fmadd_ps_512(a1, a1, norm_a1);
        norm_a2 = fmadd_ps_512(a2, a2, norm_a2);
        norm_a3 = fmadd_ps_512(a3, a3, norm_a3);
    }
    for (; i + 16 <= dim; i += 16) {
        const __m512 a16 = load_f16x16_ps(va + i);
        const __m512 b16 = load_f16x16_ps(vb + i);
        dot0 = fmadd_ps_512(a16, b16, dot0);
        norm_a0 = fmadd_ps_512(a16, a16, norm_a0);
    }

    const __m512 dot = _mm512_add_ps(_mm512_add_ps(dot0, dot1), _mm512_add_ps(dot2, dot3));
    const __m512 norm_a = _mm512_add_ps(_mm512_add_ps(norm_a0, norm_a1), _mm512_add_ps(norm_a2, norm_a3));
    double dot_sum = hsum_ps_512(dot);
    double norm_a_sum = hsum_ps_512(norm_a);

    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        const double bi = static_cast<double>(vb[i]);
        dot_sum += ai * bi;
        norm_a_sum += ai * ai;
    }

    return finalize_cosine_distance_avx512(dot_sum, norm_a_sum, query_norm_sq);
}

// The i16 norm path keeps exact 64-bit accumulation so the backend preserves
// the scalar/AVX2 contract even for vectors containing INT16_MIN.
SKETCH_AVX512F_TARGET inline double ComputeCos_AVX512::squared_norm_i16(const uint8_t *a, size_t dim) {
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

// Dot products use sign extension after the 32-bit product so negative lanes
// remain exact when they are merged into the 64-bit accumulation.
SKETCH_AVX512F_TARGET inline double ComputeCos_AVX512::dot_i16(const uint8_t *a, const uint8_t *b, size_t dim) {
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

SKETCH_AVX512F_TARGET inline double ComputeCos_AVX512::dist_i16(const uint8_t *a, const uint8_t *b, size_t dim) {
    return dist_i16_with_query_norm(a, b, dim, squared_norm_i16(b, dim));
}

// The i16 query-norm path keeps dot and norm accumulation exact in 64-bit
// integers. That is why the backend avoids vpdpwssd despite targeting AVX-512.
SKETCH_AVX512F_TARGET inline double ComputeCos_AVX512::dist_i16_with_query_norm(const uint8_t *a,
        const uint8_t *b, size_t dim, double query_norm_sq) {
    const int16_t *va = reinterpret_cast<const int16_t *>(a);
    const int16_t *vb = reinterpret_cast<const int16_t *>(b);
    const __m512i zero = _mm512_setzero_si512();
    __m512i dot_lo = _mm512_setzero_si512();
    __m512i dot_hi = _mm512_setzero_si512();
    __m512i norm_lo = _mm512_setzero_si512();
    __m512i norm_hi = _mm512_setzero_si512();

    size_t i = 0;
    for (; i + 32 <= dim; i += 32) {
        const __m512i a0 = load_i16x16_i32(va + i);
        const __m512i b0 = load_i16x16_i32(vb + i);
        const __m512i a1 = load_i16x16_i32(va + i + 16);
        const __m512i b1 = load_i16x16_i32(vb + i + 16);
        accumulate_i32_as_i64(_mm512_mullo_epi32(a0, b0), &dot_lo, &dot_hi);
        accumulate_i32_as_i64(_mm512_mullo_epi32(a1, b1), &dot_lo, &dot_hi);
        accumulate_u32_as_i64(_mm512_mullo_epi32(a0, a0), zero, &norm_lo, &norm_hi);
        accumulate_u32_as_i64(_mm512_mullo_epi32(a1, a1), zero, &norm_lo, &norm_hi);
    }
    for (; i + 16 <= dim; i += 16) {
        const __m512i a0 = load_i16x16_i32(va + i);
        const __m512i b0 = load_i16x16_i32(vb + i);
        accumulate_i32_as_i64(_mm512_mullo_epi32(a0, b0), &dot_lo, &dot_hi);
        accumulate_u32_as_i64(_mm512_mullo_epi32(a0, a0), zero, &norm_lo, &norm_hi);
    }

    double dot = hsum_epi64_512(_mm512_add_epi64(dot_lo, dot_hi));
    double norm_a = hsum_epi64_512(_mm512_add_epi64(norm_lo, norm_hi));
    for (; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        const double bi = static_cast<double>(vb[i]);
        dot += ai * bi;
        norm_a += ai * ai;
    }
    return finalize_cosine_distance_avx512(dot, norm_a, query_norm_sq);
}

#if defined(SKETCH_ENABLE_AVX512VNNI) && SKETCH_ENABLE_AVX512VNNI && (defined(__x86_64__) || defined(__i386__))

SKETCH_AVX512VNNI_TARGET inline double ComputeCos_AVX512_VNNI::dist_f32(const uint8_t *a, const uint8_t *b, size_t dim) {
    return ComputeCos_AVX512::dist_f32(a, b, dim);
}

SKETCH_AVX512VNNI_TARGET inline double ComputeCos_AVX512_VNNI::dist_f16(const uint8_t *a, const uint8_t *b, size_t dim) {
    return ComputeCos_AVX512::dist_f16(a, b, dim);
}

SKETCH_AVX512VNNI_TARGET inline double ComputeCos_AVX512_VNNI::dist_f32_with_query_norm(const uint8_t *a,
        const uint8_t *b, size_t dim, double query_norm_sq) {
    return ComputeCos_AVX512::dist_f32_with_query_norm(a, b, dim, query_norm_sq);
}

SKETCH_AVX512VNNI_TARGET inline double ComputeCos_AVX512_VNNI::dist_f16_with_query_norm(const uint8_t *a,
        const uint8_t *b, size_t dim, double query_norm_sq) {
    return ComputeCos_AVX512::dist_f16_with_query_norm(a, b, dim, query_norm_sq);
}

SKETCH_AVX512VNNI_TARGET inline double ComputeCos_AVX512_VNNI::squared_norm_f32(const uint8_t *a, size_t dim) {
    return ComputeCos_AVX512::squared_norm_f32(a, dim);
}

SKETCH_AVX512VNNI_TARGET inline double ComputeCos_AVX512_VNNI::squared_norm_f16(const uint8_t *a, size_t dim) {
    return ComputeCos_AVX512::squared_norm_f16(a, dim);
}

SKETCH_AVX512VNNI_TARGET inline double ComputeCos_AVX512_VNNI::dot_f32(const uint8_t *a, const uint8_t *b, size_t dim) {
    return ComputeCos_AVX512::dot_f32(a, b, dim);
}

SKETCH_AVX512VNNI_TARGET inline double ComputeCos_AVX512_VNNI::dot_f16(const uint8_t *a, const uint8_t *b, size_t dim) {
    return ComputeCos_AVX512::dot_f16(a, b, dim);
}

SKETCH_AVX512VNNI_TARGET inline double ComputeCos_AVX512_VNNI::squared_norm_i16(const uint8_t *a, size_t dim) {
    return ComputeCos_AVX512::squared_norm_i16(a, dim);
}

SKETCH_AVX512VNNI_TARGET inline double ComputeCos_AVX512_VNNI::dot_i16(const uint8_t *a, const uint8_t *b, size_t dim) {
    return ComputeCos_AVX512::dot_i16(a, b, dim);
}

SKETCH_AVX512VNNI_TARGET inline double ComputeCos_AVX512_VNNI::dist_i16(const uint8_t *a, const uint8_t *b, size_t dim) {
    return dist_i16_with_query_norm(a, b, dim, squared_norm_i16(b, dim));
}

SKETCH_AVX512VNNI_TARGET inline double ComputeCos_AVX512_VNNI::dist_i16_with_query_norm(const uint8_t *a,
        const uint8_t *b, size_t dim, double query_norm_sq) {
    return ComputeCos_AVX512::dist_i16_with_query_norm(a, b, dim, query_norm_sq);
}

#endif // SKETCH_ENABLE_AVX512VNNI

#endif // SKETCH_ENABLE_AVX512F || SKETCH_ENABLE_AVX512VNNI

} // namespace sketch2
