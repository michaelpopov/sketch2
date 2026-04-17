// Implements shared dot-product and squared-norm helpers for compute backends.

#pragma once

#include "core/compute/compute.h"
#include "core/utils/arch_detection.h"
#include "core/utils/singleton.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#if SKETCH_HAS_AVX2
#include "compute_dot_norm_avx2.h"
#endif
#if SKETCH_HAS_AVX512
#include "compute_dot_norm_avx512.h"
#endif

#if SKETCH_HAS_NEON
#include "compute_dot_norm_neon.h"
#endif

namespace sketch2 {

// ComputeDotNorm exposes the shared dot-product and squared-norm primitives
// used by both cosine and L2 scan paths.
class ComputeDotNorm {
public:
    using SquaredNormFn = double (*)(const uint8_t*, size_t);
    using DotFn = double (*)(const uint8_t*, const uint8_t*, size_t);

    static SquaredNormFn resolve_squared_norm(DataType type);
    static DotFn resolve_dot(DataType type);

    static double squared_norm_f32(const uint8_t *a, size_t dim);
    static double squared_norm_f16(const uint8_t *a, size_t dim);
    static double squared_norm_i16(const uint8_t *a, size_t dim);
    static double dot_f32(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dot_f16(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dot_i16(const uint8_t *a, const uint8_t *b, size_t dim);
};

inline ComputeDotNorm::SquaredNormFn ComputeDotNorm::resolve_squared_norm(DataType type) {
    switch (get_singleton().compute_unit().kind()) {
#if SKETCH_HAS_AVX512VNNI
        case ComputeBackendKind::avx512_vnni:
            switch (type) {
                case DataType::f32: return &ComputeDotNorm_AVX512_VNNI::squared_norm_f32;
                case DataType::f16: return &ComputeDotNorm_AVX512_VNNI::squared_norm_f16;
                case DataType::i16: return &ComputeDotNorm_AVX512_VNNI::squared_norm_i16;
                default: break;
            }
            break;
#endif
#if SKETCH_HAS_AVX512F
        case ComputeBackendKind::avx512f:
            switch (type) {
                case DataType::f32: return &ComputeDotNorm_AVX512::squared_norm_f32;
                case DataType::f16: return &ComputeDotNorm_AVX512::squared_norm_f16;
                case DataType::i16: return &ComputeDotNorm_AVX512::squared_norm_i16;
                default: break;
            }
            break;
#endif
#if SKETCH_HAS_AVX2
        case ComputeBackendKind::avx2:
            switch (type) {
                case DataType::f32: return &ComputeDotNorm_AVX2::squared_norm_f32;
                case DataType::f16: return &ComputeDotNorm_AVX2::squared_norm_f16;
                case DataType::i16: return &ComputeDotNorm_AVX2::squared_norm_i16;
                default: break;
            }
            break;
#endif
#if SKETCH_HAS_NEON
        case ComputeBackendKind::neon:
            switch (type) {
                case DataType::f32: return &ComputeDotNorm_Neon::squared_norm_f32;
                case DataType::f16: return &ComputeDotNorm_Neon::squared_norm_f16;
                case DataType::i16: return &ComputeDotNorm_Neon::squared_norm_i16;
                default: break;
            }
            break;
#endif
        case ComputeBackendKind::scalar:
        default:
            break;
    }

    switch (type) {
        case DataType::f32: return &squared_norm_f32;
        case DataType::f16: return &squared_norm_f16;
        case DataType::i16: return &squared_norm_i16;
        default:
            assert(false);
            throw std::runtime_error("ComputeDotNorm::resolve_squared_norm: unsupported data type");
    }
}

inline ComputeDotNorm::DotFn ComputeDotNorm::resolve_dot(DataType type) {
    switch (get_singleton().compute_unit().kind()) {
#if SKETCH_HAS_AVX512VNNI
        case ComputeBackendKind::avx512_vnni:
            switch (type) {
                case DataType::f32: return &ComputeDotNorm_AVX512_VNNI::dot_f32;
                case DataType::f16: return &ComputeDotNorm_AVX512_VNNI::dot_f16;
                case DataType::i16: return &ComputeDotNorm_AVX512_VNNI::dot_i16;
                default: break;
            }
            break;
#endif
#if SKETCH_HAS_AVX512F
        case ComputeBackendKind::avx512f:
            switch (type) {
                case DataType::f32: return &ComputeDotNorm_AVX512::dot_f32;
                case DataType::f16: return &ComputeDotNorm_AVX512::dot_f16;
                case DataType::i16: return &ComputeDotNorm_AVX512::dot_i16;
                default: break;
            }
            break;
#endif
#if SKETCH_HAS_AVX2
        case ComputeBackendKind::avx2:
            switch (type) {
                case DataType::f32: return &ComputeDotNorm_AVX2::dot_f32;
                case DataType::f16: return &ComputeDotNorm_AVX2::dot_f16;
                case DataType::i16: return &ComputeDotNorm_AVX2::dot_i16;
                default: break;
            }
            break;
#endif
#if SKETCH_HAS_NEON
        case ComputeBackendKind::neon:
            switch (type) {
                case DataType::f32: return &ComputeDotNorm_Neon::dot_f32;
                case DataType::f16: return &ComputeDotNorm_Neon::dot_f16;
                case DataType::i16: return &ComputeDotNorm_Neon::dot_i16;
                default: break;
            }
            break;
#endif
        case ComputeBackendKind::scalar:
        default:
            break;
    }

    switch (type) {
        case DataType::f32: return &dot_f32;
        case DataType::f16: return &dot_f16;
        case DataType::i16: return &dot_i16;
        default:
            assert(false);
            throw std::runtime_error("ComputeDotNorm::resolve_dot: unsupported data type");
    }
}

// Scalar norm helpers widen to double so scalar and SIMD backends follow the
// same accumulation model and produce comparable results.
inline double ComputeDotNorm::squared_norm_f32(const uint8_t* a, size_t dim) {
    const float* va = reinterpret_cast<const float*>(a);
    double norm = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        norm += ai * ai;
    }
    return norm;
}

inline double ComputeDotNorm::squared_norm_f16(const uint8_t* a, size_t dim) {
    const float16* va = reinterpret_cast<const float16*>(a);
    double norm = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        norm += ai * ai;
    }
    return norm;
}

inline double ComputeDotNorm::squared_norm_i16(const uint8_t* a, size_t dim) {
    const int16_t* va = reinterpret_cast<const int16_t*>(a);
    double norm = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        norm += ai * ai;
    }
    return norm;
}

// Scalar dot helpers mirror the SIMD kernels semantically: load native values,
// widen to double, and accumulate in a backend-independent format.
inline double ComputeDotNorm::dot_f32(const uint8_t* a, const uint8_t* b, size_t dim) {
    const float* va = reinterpret_cast<const float*>(a);
    const float* vb = reinterpret_cast<const float*>(b);
    double dot = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        dot += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
    }
    return dot;
}

inline double ComputeDotNorm::dot_f16(const uint8_t* a, const uint8_t* b, size_t dim) {
    const float16* va = reinterpret_cast<const float16*>(a);
    const float16* vb = reinterpret_cast<const float16*>(b);
    double dot = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        dot += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
    }
    return dot;
}

inline double ComputeDotNorm::dot_i16(const uint8_t* a, const uint8_t* b, size_t dim) {
    const int16_t* va = reinterpret_cast<const int16_t*>(a);
    const int16_t* vb = reinterpret_cast<const int16_t*>(b);
    double dot = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        dot += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
    }
    return dot;
}

} // namespace sketch2
