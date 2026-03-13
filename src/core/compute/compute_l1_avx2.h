// Implements AVX2-optimized L1-distance kernels.

#pragma once
#include "core/compute/compute_avx2_utils.h"
#include "core/compute/compute.h"
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace sketch2 {

// Computes L1 (Manhattan) distance between two vectors.
// ComputeL1_AVX2 exists to provide AVX2-specialized L1 kernels for x86 scan
// workloads. It exposes typed entry points that match the portable L1 interface.
class ComputeL1_AVX2 {
public:
    static double dist_f32(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_f16(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_i16(const uint8_t *a, const uint8_t *b, size_t dim);

private:
    static double dist_f32_8(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_i16_16(const uint8_t *a, const uint8_t *b, size_t dim);
};

#if defined(__AVX2__)

inline double hsum_epi64_256(__m256i v) {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    const __m128i sum = _mm_add_epi64(lo, hi);
    alignas(16) uint64_t lanes[2];
    _mm_store_si128(reinterpret_cast<__m128i *>(lanes), sum);
    return static_cast<double>(lanes[0] + lanes[1]);
}

inline __m256i accumulate_abs_i16_as_i64(__m256i acc, __m256i a16, __m256i b16, __m256i zero) {
    const __m128i a_lo16 = _mm256_castsi256_si128(a16);
    const __m128i a_hi16 = _mm256_extracti128_si256(a16, 1);
    const __m128i b_lo16 = _mm256_castsi256_si128(b16);
    const __m128i b_hi16 = _mm256_extracti128_si256(b16, 1);

    const __m256i a_lo32 = _mm256_cvtepi16_epi32(a_lo16);
    const __m256i a_hi32 = _mm256_cvtepi16_epi32(a_hi16);
    const __m256i b_lo32 = _mm256_cvtepi16_epi32(b_lo16);
    const __m256i b_hi32 = _mm256_cvtepi16_epi32(b_hi16);

    const __m256i abs_lo32 = _mm256_abs_epi32(_mm256_sub_epi32(a_lo32, b_lo32));
    const __m256i abs_hi32 = _mm256_abs_epi32(_mm256_sub_epi32(a_hi32, b_hi32));

    acc = _mm256_add_epi64(acc, _mm256_unpacklo_epi32(abs_lo32, zero));
    acc = _mm256_add_epi64(acc, _mm256_unpackhi_epi32(abs_lo32, zero));
    acc = _mm256_add_epi64(acc, _mm256_unpacklo_epi32(abs_hi32, zero));
    acc = _mm256_add_epi64(acc, _mm256_unpackhi_epi32(abs_hi32, zero));
    return acc;
}

inline double ComputeL1_AVX2::dist_f32(const uint8_t *a, const uint8_t *b, size_t dim) {
    if (dim % 8 == 0) {
        return dist_f32_8(a, b, dim);
    }

    const float *va = reinterpret_cast<const float *>(a);
    const float *vb = reinterpret_cast<const float *>(b);
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    const bool aligned =
        (((reinterpret_cast<uintptr_t>(va) | reinterpret_cast<uintptr_t>(vb)) & (kAvx2VectorAlignment - 1u)) == 0u);

    size_t i = 0;
    if (aligned) {
        for (; i + 32 <= dim; i += 32) {
            const __m256 a0 = _mm256_load_ps(va + i);
            const __m256 b0 = _mm256_load_ps(vb + i);
            const __m256 a1 = _mm256_load_ps(va + i + 8);
            const __m256 b1 = _mm256_load_ps(vb + i + 8);
            const __m256 a2 = _mm256_load_ps(va + i + 16);
            const __m256 b2 = _mm256_load_ps(vb + i + 16);
            const __m256 a3 = _mm256_load_ps(va + i + 24);
            const __m256 b3 = _mm256_load_ps(vb + i + 24);

            acc0 = _mm256_add_ps(acc0, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a0, b0)));
            acc1 = _mm256_add_ps(acc1, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a1, b1)));
            acc2 = _mm256_add_ps(acc2, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a2, b2)));
            acc3 = _mm256_add_ps(acc3, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a3, b3)));
        }
        for (; i + 8 <= dim; i += 8) {
            const __m256 a8 = _mm256_load_ps(va + i);
            const __m256 b8 = _mm256_load_ps(vb + i);
            acc0 = _mm256_add_ps(acc0, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a8, b8)));
        }
    } else {
        for (; i + 32 <= dim; i += 32) {
            const __m256 a0 = _mm256_loadu_ps(va + i);
            const __m256 b0 = _mm256_loadu_ps(vb + i);
            const __m256 a1 = _mm256_loadu_ps(va + i + 8);
            const __m256 b1 = _mm256_loadu_ps(vb + i + 8);
            const __m256 a2 = _mm256_loadu_ps(va + i + 16);
            const __m256 b2 = _mm256_loadu_ps(vb + i + 16);
            const __m256 a3 = _mm256_loadu_ps(va + i + 24);
            const __m256 b3 = _mm256_loadu_ps(vb + i + 24);

            acc0 = _mm256_add_ps(acc0, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a0, b0)));
            acc1 = _mm256_add_ps(acc1, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a1, b1)));
            acc2 = _mm256_add_ps(acc2, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a2, b2)));
            acc3 = _mm256_add_ps(acc3, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a3, b3)));
        }
        for (; i + 8 <= dim; i += 8) {
            const __m256 a8 = _mm256_loadu_ps(va + i);
            const __m256 b8 = _mm256_loadu_ps(vb + i);
            acc0 = _mm256_add_ps(acc0, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a8, b8)));
        }
    }

    const __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    double sum = hsum_ps_256(acc);

    for (; i < dim; ++i) {
        sum += std::abs(va[i] - vb[i]);
    }
    return sum;
}

inline double ComputeL1_AVX2::dist_f32_8(const uint8_t *a, const uint8_t *b, size_t dim) {
    const float *va = reinterpret_cast<const float *>(a);
    const float *vb = reinterpret_cast<const float *>(b);
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    const bool aligned =
        (((reinterpret_cast<uintptr_t>(va) | reinterpret_cast<uintptr_t>(vb)) & (kAvx2VectorAlignment - 1u)) == 0u);

    size_t i = 0;
    if (aligned) {
        for (; i + 32 <= dim; i += 32) {
            const __m256 a0 = _mm256_load_ps(va + i);
            const __m256 b0 = _mm256_load_ps(vb + i);
            const __m256 a1 = _mm256_load_ps(va + i + 8);
            const __m256 b1 = _mm256_load_ps(vb + i + 8);
            const __m256 a2 = _mm256_load_ps(va + i + 16);
            const __m256 b2 = _mm256_load_ps(vb + i + 16);
            const __m256 a3 = _mm256_load_ps(va + i + 24);
            const __m256 b3 = _mm256_load_ps(vb + i + 24);

            acc0 = _mm256_add_ps(acc0, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a0, b0)));
            acc1 = _mm256_add_ps(acc1, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a1, b1)));
            acc2 = _mm256_add_ps(acc2, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a2, b2)));
            acc3 = _mm256_add_ps(acc3, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a3, b3)));
        }
        for (; i < dim; i += 8) {
            const __m256 a8 = _mm256_load_ps(va + i);
            const __m256 b8 = _mm256_load_ps(vb + i);
            acc0 = _mm256_add_ps(acc0, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a8, b8)));
        }
    } else {
        for (; i + 32 <= dim; i += 32) {
            const __m256 a0 = _mm256_loadu_ps(va + i);
            const __m256 b0 = _mm256_loadu_ps(vb + i);
            const __m256 a1 = _mm256_loadu_ps(va + i + 8);
            const __m256 b1 = _mm256_loadu_ps(vb + i + 8);
            const __m256 a2 = _mm256_loadu_ps(va + i + 16);
            const __m256 b2 = _mm256_loadu_ps(vb + i + 16);
            const __m256 a3 = _mm256_loadu_ps(va + i + 24);
            const __m256 b3 = _mm256_loadu_ps(vb + i + 24);

            acc0 = _mm256_add_ps(acc0, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a0, b0)));
            acc1 = _mm256_add_ps(acc1, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a1, b1)));
            acc2 = _mm256_add_ps(acc2, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a2, b2)));
            acc3 = _mm256_add_ps(acc3, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a3, b3)));
        }
        for (; i < dim; i += 8) {
            const __m256 a8 = _mm256_loadu_ps(va + i);
            const __m256 b8 = _mm256_loadu_ps(vb + i);
            acc0 = _mm256_add_ps(acc0, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a8, b8)));
        }
    }

    const __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    return hsum_ps_256(acc);
}

inline double ComputeL1_AVX2::dist_f16(const uint8_t *a, const uint8_t *b, size_t dim) {
    const float16 *va = reinterpret_cast<const float16 *>(a);
    const float16 *vb = reinterpret_cast<const float16 *>(b);
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    const bool aligned =
        (((reinterpret_cast<uintptr_t>(va) | reinterpret_cast<uintptr_t>(vb)) & (kHalfVectorAlignment - 1u)) == 0u);

    size_t i = 0;
    if (aligned) {
        for (; i + 32 <= dim; i += 32) {
            const __m256 a0 = load_f16x8_ps_aligned(va + i);
            const __m256 b0 = load_f16x8_ps_aligned(vb + i);
            const __m256 a1 = load_f16x8_ps_aligned(va + i + 8);
            const __m256 b1 = load_f16x8_ps_aligned(vb + i + 8);
            const __m256 a2 = load_f16x8_ps_aligned(va + i + 16);
            const __m256 b2 = load_f16x8_ps_aligned(vb + i + 16);
            const __m256 a3 = load_f16x8_ps_aligned(va + i + 24);
            const __m256 b3 = load_f16x8_ps_aligned(vb + i + 24);

            acc0 = _mm256_add_ps(acc0, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a0, b0)));
            acc1 = _mm256_add_ps(acc1, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a1, b1)));
            acc2 = _mm256_add_ps(acc2, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a2, b2)));
            acc3 = _mm256_add_ps(acc3, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a3, b3)));
        }
        for (; i + 8 <= dim; i += 8) {
            const __m256 a8 = load_f16x8_ps_aligned(va + i);
            const __m256 b8 = load_f16x8_ps_aligned(vb + i);
            acc0 = _mm256_add_ps(acc0, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a8, b8)));
        }
    } else {
        for (; i + 32 <= dim; i += 32) {
            const __m256 a0 = load_f16x8_ps(va + i);
            const __m256 b0 = load_f16x8_ps(vb + i);
            const __m256 a1 = load_f16x8_ps(va + i + 8);
            const __m256 b1 = load_f16x8_ps(vb + i + 8);
            const __m256 a2 = load_f16x8_ps(va + i + 16);
            const __m256 b2 = load_f16x8_ps(vb + i + 16);
            const __m256 a3 = load_f16x8_ps(va + i + 24);
            const __m256 b3 = load_f16x8_ps(vb + i + 24);

            acc0 = _mm256_add_ps(acc0, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a0, b0)));
            acc1 = _mm256_add_ps(acc1, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a1, b1)));
            acc2 = _mm256_add_ps(acc2, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a2, b2)));
            acc3 = _mm256_add_ps(acc3, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a3, b3)));
        }
        for (; i + 8 <= dim; i += 8) {
            const __m256 a8 = load_f16x8_ps(va + i);
            const __m256 b8 = load_f16x8_ps(vb + i);
            acc0 = _mm256_add_ps(acc0, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a8, b8)));
        }
    }

    const __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    double sum = hsum_ps_256(acc);
    for (; i < dim; ++i) {
        const double d = static_cast<double>(va[i]) - static_cast<double>(vb[i]);
        sum += std::abs(d);
    }
    return sum;
}

inline double ComputeL1_AVX2::dist_i16(const uint8_t *a, const uint8_t *b, size_t dim) {
    if (dim % 16 == 0) {
        return dist_i16_16(a, b, dim);
    }

    const int16_t *va = reinterpret_cast<const int16_t *>(a);
    const int16_t *vb = reinterpret_cast<const int16_t *>(b);
    const __m256i zero = _mm256_setzero_si256();
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    const bool aligned =
        (((reinterpret_cast<uintptr_t>(va) | reinterpret_cast<uintptr_t>(vb)) & (kAvx2VectorAlignment - 1u)) == 0u);

    size_t i = 0;
    if (aligned) {
        for (; i + 32 <= dim; i += 32) {
            const __m256i a16_0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(va + i));
            const __m256i b16_0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(vb + i));
            const __m256i a16_1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(va + i + 16));
            const __m256i b16_1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(vb + i + 16));

            acc0 = accumulate_abs_i16_as_i64(acc0, a16_0, b16_0, zero);
            acc1 = accumulate_abs_i16_as_i64(acc1, a16_1, b16_1, zero);
        }
        for (; i + 16 <= dim; i += 16) {
            const __m256i a16 = _mm256_load_si256(reinterpret_cast<const __m256i *>(va + i));
            const __m256i b16 = _mm256_load_si256(reinterpret_cast<const __m256i *>(vb + i));
            acc0 = accumulate_abs_i16_as_i64(acc0, a16, b16, zero);
        }
    } else {
        for (; i + 32 <= dim; i += 32) {
            const __m256i a16_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(va + i));
            const __m256i b16_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(vb + i));
            const __m256i a16_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(va + i + 16));
            const __m256i b16_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(vb + i + 16));

            acc0 = accumulate_abs_i16_as_i64(acc0, a16_0, b16_0, zero);
            acc1 = accumulate_abs_i16_as_i64(acc1, a16_1, b16_1, zero);
        }
        for (; i + 16 <= dim; i += 16) {
            const __m256i a16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(va + i));
            const __m256i b16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(vb + i));
            acc0 = accumulate_abs_i16_as_i64(acc0, a16, b16, zero);
        }
    }

    const __m256i acc64 = _mm256_add_epi64(acc0, acc1);
    double sum = hsum_epi64_256(acc64);

    for (; i < dim; ++i) {
        const int diff = static_cast<int>(va[i]) - static_cast<int>(vb[i]);
        sum += std::abs(diff);
    }
    return sum;
}

inline double ComputeL1_AVX2::dist_i16_16(const uint8_t *a, const uint8_t *b, size_t dim) {
    const int16_t *va = reinterpret_cast<const int16_t *>(a);
    const int16_t *vb = reinterpret_cast<const int16_t *>(b);
    const __m256i zero = _mm256_setzero_si256();
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    const bool aligned =
        (((reinterpret_cast<uintptr_t>(va) | reinterpret_cast<uintptr_t>(vb)) & (kAvx2VectorAlignment - 1u)) == 0u);

    size_t i = 0;
    if (aligned) {
        for (; i + 32 <= dim; i += 32) {
            const __m256i a16_0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(va + i));
            const __m256i b16_0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(vb + i));
            const __m256i a16_1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(va + i + 16));
            const __m256i b16_1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(vb + i + 16));

            acc0 = accumulate_abs_i16_as_i64(acc0, a16_0, b16_0, zero);
            acc1 = accumulate_abs_i16_as_i64(acc1, a16_1, b16_1, zero);
        }
        for (; i < dim; i += 16) {
            const __m256i a16 = _mm256_load_si256(reinterpret_cast<const __m256i *>(va + i));
            const __m256i b16 = _mm256_load_si256(reinterpret_cast<const __m256i *>(vb + i));
            acc0 = accumulate_abs_i16_as_i64(acc0, a16, b16, zero);
        }
    } else {
        for (; i + 32 <= dim; i += 32) {
            const __m256i a16_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(va + i));
            const __m256i b16_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(vb + i));
            const __m256i a16_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(va + i + 16));
            const __m256i b16_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(vb + i + 16));

            acc0 = accumulate_abs_i16_as_i64(acc0, a16_0, b16_0, zero);
            acc1 = accumulate_abs_i16_as_i64(acc1, a16_1, b16_1, zero);
        }
        for (; i < dim; i += 16) {
            const __m256i a16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(va + i));
            const __m256i b16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(vb + i));
            acc0 = accumulate_abs_i16_as_i64(acc0, a16, b16, zero);
        }
    }

    return hsum_epi64_256(_mm256_add_epi64(acc0, acc1));
}

#endif

} // namespace sketch2
