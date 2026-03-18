// Implements the portable cosine-distance primitives and helpers.

#pragma once
#include "core/compute/compute.h"
#include "core/utils/singleton.h"
#include "core/utils/arch_detection.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#if SKETCH_HAS_AVX2
#include "compute_cos_avx2.h"
#endif
#if SKETCH_HAS_AVX512
#include "compute_cos_avx512.h"
#endif

#if SKETCH_HAS_NEON
#include "compute_cos_neon.h"
#endif

namespace sketch2 {

// ComputeCos exists to collect all cosine-distance functionality behind one
// backend selector. It provides typed scalar implementations plus runtime
// dispatch helpers that pick the best cosine, dot-product, and norm routines.
class ComputeCos : public ICompute {
public:
    using DistFn = double (*)(const uint8_t*, const uint8_t*, size_t);
    using DistWithQueryNormFn = double (*)(const uint8_t*, const uint8_t*, size_t, double);
    using SquaredNormFn = double (*)(const uint8_t*, size_t);
    using DotFn = double (*)(const uint8_t*, const uint8_t*, size_t);

    double dist(const uint8_t *a, const uint8_t *b, DataType type, size_t dim) override;
    // These runtime resolvers intentionally stay next to the typed entrypoints
    // so scanner/template code can bind a concrete backend once and keep the
    // hot loop on that specialized path. They are not "free" helpers: each
    // resolver reads the process-wide ComputeUnit from the singleton, so
    // callers should use them as setup-time dispatch and cache the result when
    // reusing the same type/backend combination.
    static DistFn resolve_dist(DataType type);
    static DistWithQueryNormFn resolve_dist_with_query_norm(DataType type);
    static SquaredNormFn resolve_squared_norm(DataType type);
    static DotFn resolve_dot(DataType type);

    // Typed entrypoints used by scanner template dispatch and scalar fallback.
    static double dist_f32(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_f16(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_i16(const uint8_t *a, const uint8_t *b, size_t dim);
    static double squared_norm_f32(const uint8_t *a, size_t dim);
    static double squared_norm_f16(const uint8_t *a, size_t dim);
    static double squared_norm_i16(const uint8_t *a, size_t dim);
    static double dot_f32(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dot_f16(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dot_i16(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_f32_with_query_norm(const uint8_t *a, const uint8_t *b, size_t dim, double query_norm_sq);
    static double dist_f16_with_query_norm(const uint8_t *a, const uint8_t *b, size_t dim, double query_norm_sq);
    static double dist_i16_with_query_norm(const uint8_t *a, const uint8_t *b, size_t dim, double query_norm_sq);
};

// Normalize the raw cosine ingredients into the public distance contract,
// including the special zero-vector behavior shared by all backends.
inline double finalize_cosine_distance(double dot, double norm_a, double norm_b) {
    if (norm_a == 0.0 && norm_b == 0.0) {
        return 0.0;
    }
    if (norm_a == 0.0 || norm_b == 0.0) {
        return 1.0;
    }
    const double cosine = std::clamp(dot / std::sqrt(norm_a * norm_b), -1.0, 1.0);
    return 1.0 - cosine;
}

// Readers that persist inverse norms can skip the sqrt/divide work and still
// reuse the same zero-vector and clamping semantics as the raw-norm path.
inline double finalize_cosine_distance_from_inverse_norms(double dot, double inv_norm_a, double inv_norm_b) {
    if (inv_norm_a == 0.0 && inv_norm_b == 0.0) {
        return 0.0;
    }
    if (inv_norm_a == 0.0 || inv_norm_b == 0.0) {
        return 1.0;
    }
    const double cosine = std::clamp(dot * inv_norm_a * inv_norm_b, -1.0, 1.0);
    return 1.0 - cosine;
}

inline double ComputeCos::dist(const uint8_t *a, const uint8_t *b, DataType type, size_t dim) {
    DistFn fn = resolve_dist(type);
    return fn(a, b, dim);
}

inline ComputeCos::DistFn ComputeCos::resolve_dist(DataType type) {
    validate_type(type);
    switch (get_singleton().compute_unit().kind()) {
#if SKETCH_HAS_AVX512VNNI
        case ComputeBackendKind::avx512_vnni:
            switch (type) {
                case DataType::f32: return &ComputeCos_AVX512_VNNI::dist_f32;
                case DataType::f16: return &ComputeCos_AVX512_VNNI::dist_f16;
                case DataType::i16: return &ComputeCos_AVX512_VNNI::dist_i16;
                default: break;
            }
            break;
#endif
#if SKETCH_HAS_AVX512F
        case ComputeBackendKind::avx512f:
            switch (type) {
                case DataType::f32: return &ComputeCos_AVX512::dist_f32;
                case DataType::f16: return &ComputeCos_AVX512::dist_f16;
                case DataType::i16: return &ComputeCos_AVX512::dist_i16;
                default: break;
            }
            break;
#endif
#if SKETCH_HAS_AVX2
        case ComputeBackendKind::avx2:
            switch (type) {
                case DataType::f32: return &ComputeCos_AVX2::dist_f32;
                case DataType::f16: return &ComputeCos_AVX2::dist_f16;
                case DataType::i16: return &ComputeCos_AVX2::dist_i16;
                default: break;
            }
            break;
#endif
#if SKETCH_HAS_NEON
        case ComputeBackendKind::neon:
            switch (type) {
                case DataType::f32: return &ComputeCos_Neon::dist_f32;
                case DataType::f16: return &ComputeCos_Neon::dist_f16;
                case DataType::i16: return &ComputeCos_Neon::dist_i16;
                default: break;
            }
            break;
#endif
        case ComputeBackendKind::scalar:
        default:
            break;
    }

    switch (type) {
        case DataType::f32: return &dist_f32;
        case DataType::f16: return &dist_f16;
        case DataType::i16: return &dist_i16;
        default:
            assert(false);
            throw std::runtime_error("ComputeCos::resolve_dist: unsupported data type");
    }
}

inline ComputeCos::DistWithQueryNormFn ComputeCos::resolve_dist_with_query_norm(DataType type) {
    validate_type(type);
    switch (get_singleton().compute_unit().kind()) {
#if SKETCH_HAS_AVX512VNNI
        case ComputeBackendKind::avx512_vnni:
            switch (type) {
                case DataType::f32: return &ComputeCos_AVX512_VNNI::dist_f32_with_query_norm;
                case DataType::f16: return &ComputeCos_AVX512_VNNI::dist_f16_with_query_norm;
                case DataType::i16: return &ComputeCos_AVX512_VNNI::dist_i16_with_query_norm;
                default: break;
            }
            break;
#endif
#if SKETCH_HAS_AVX512F
        case ComputeBackendKind::avx512f:
            switch (type) {
                case DataType::f32: return &ComputeCos_AVX512::dist_f32_with_query_norm;
                case DataType::f16: return &ComputeCos_AVX512::dist_f16_with_query_norm;
                case DataType::i16: return &ComputeCos_AVX512::dist_i16_with_query_norm;
                default: break;
            }
            break;
#endif
#if SKETCH_HAS_AVX2
        case ComputeBackendKind::avx2:
            switch (type) {
                case DataType::f32: return &ComputeCos_AVX2::dist_f32_with_query_norm;
                case DataType::f16: return &ComputeCos_AVX2::dist_f16_with_query_norm;
                case DataType::i16: return &ComputeCos_AVX2::dist_i16_with_query_norm;
                default: break;
            }
            break;
#endif
#if SKETCH_HAS_NEON
        case ComputeBackendKind::neon:
            switch (type) {
                case DataType::f32: return &ComputeCos_Neon::dist_f32_with_query_norm;
                case DataType::f16: return &ComputeCos_Neon::dist_f16_with_query_norm;
                case DataType::i16: return &ComputeCos_Neon::dist_i16_with_query_norm;
                default: break;
            }
            break;
#endif
        case ComputeBackendKind::scalar:
        default:
            break;
    }

    switch (type) {
        case DataType::f32: return &dist_f32_with_query_norm;
        case DataType::f16: return &dist_f16_with_query_norm;
        case DataType::i16: return &dist_i16_with_query_norm;
        default:
            assert(false);
            throw std::runtime_error("ComputeCos::resolve_dist_with_query_norm: unsupported data type");
    }
}

inline ComputeCos::SquaredNormFn ComputeCos::resolve_squared_norm(DataType type) {
    validate_type(type);
    switch (get_singleton().compute_unit().kind()) {
#if SKETCH_HAS_AVX512VNNI
        case ComputeBackendKind::avx512_vnni:
            switch (type) {
                case DataType::f32: return &ComputeCos_AVX512_VNNI::squared_norm_f32;
                case DataType::f16: return &ComputeCos_AVX512_VNNI::squared_norm_f16;
                case DataType::i16: return &ComputeCos_AVX512_VNNI::squared_norm_i16;
                default: break;
            }
            break;
#endif
#if SKETCH_HAS_AVX512F
        case ComputeBackendKind::avx512f:
            switch (type) {
                case DataType::f32: return &ComputeCos_AVX512::squared_norm_f32;
                case DataType::f16: return &ComputeCos_AVX512::squared_norm_f16;
                case DataType::i16: return &ComputeCos_AVX512::squared_norm_i16;
                default: break;
            }
            break;
#endif
#if SKETCH_HAS_AVX2
        case ComputeBackendKind::avx2:
            switch (type) {
                case DataType::f32: return &ComputeCos_AVX2::squared_norm_f32;
                case DataType::f16: return &ComputeCos_AVX2::squared_norm_f16;
                case DataType::i16: return &ComputeCos_AVX2::squared_norm_i16;
                default: break;
            }
            break;
#endif
#if SKETCH_HAS_NEON
        case ComputeBackendKind::neon:
            switch (type) {
                case DataType::f32: return &ComputeCos_Neon::squared_norm_f32;
                case DataType::f16: return &ComputeCos_Neon::squared_norm_f16;
                case DataType::i16: return &ComputeCos_Neon::squared_norm_i16;
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
            throw std::runtime_error("ComputeCos::resolve_squared_norm: unsupported data type");
    }
}

inline ComputeCos::DotFn ComputeCos::resolve_dot(DataType type) {
    validate_type(type);
    switch (get_singleton().compute_unit().kind()) {
#if SKETCH_HAS_AVX512VNNI
        case ComputeBackendKind::avx512_vnni:
            switch (type) {
                case DataType::f32: return &ComputeCos_AVX512_VNNI::dot_f32;
                case DataType::f16: return &ComputeCos_AVX512_VNNI::dot_f16;
                case DataType::i16: return &ComputeCos_AVX512_VNNI::dot_i16;
                default: break;
            }
            break;
#endif
#if SKETCH_HAS_AVX512F
        case ComputeBackendKind::avx512f:
            switch (type) {
                case DataType::f32: return &ComputeCos_AVX512::dot_f32;
                case DataType::f16: return &ComputeCos_AVX512::dot_f16;
                case DataType::i16: return &ComputeCos_AVX512::dot_i16;
                default: break;
            }
            break;
#endif
#if SKETCH_HAS_AVX2
        case ComputeBackendKind::avx2:
            switch (type) {
                case DataType::f32: return &ComputeCos_AVX2::dot_f32;
                case DataType::f16: return &ComputeCos_AVX2::dot_f16;
                case DataType::i16: return &ComputeCos_AVX2::dot_i16;
                default: break;
            }
            break;
#endif
#if SKETCH_HAS_NEON
        case ComputeBackendKind::neon:
            switch (type) {
                case DataType::f32: return &ComputeCos_Neon::dot_f32;
                case DataType::f16: return &ComputeCos_Neon::dot_f16;
                case DataType::i16: return &ComputeCos_Neon::dot_i16;
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
            throw std::runtime_error("ComputeCos::resolve_dot: unsupported data type");
    }
}

inline double ComputeCos::dist_f32(const uint8_t* a, const uint8_t* b, size_t dim) {
    return dist_f32_with_query_norm(a, b, dim, squared_norm_f32(b, dim));
}

inline double ComputeCos::dist_f16(const uint8_t* a, const uint8_t* b, size_t dim) {
    return dist_f16_with_query_norm(a, b, dim, squared_norm_f16(b, dim));
}

inline double ComputeCos::dist_i16(const uint8_t* a, const uint8_t* b, size_t dim) {
    return dist_i16_with_query_norm(a, b, dim, squared_norm_i16(b, dim));
}

// Scalar norm helpers widen to double so scalar and SIMD backends follow the
// same accumulation model and produce comparable results.
inline double ComputeCos::squared_norm_f32(const uint8_t* a, size_t dim) {
    const float* va = reinterpret_cast<const float*>(a);
    double norm = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        norm += ai * ai;
    }
    return norm;
}

inline double ComputeCos::squared_norm_f16(const uint8_t* a, size_t dim) {
    const float16* va = reinterpret_cast<const float16*>(a);
    double norm = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        norm += ai * ai;
    }
    return norm;
}

inline double ComputeCos::squared_norm_i16(const uint8_t* a, size_t dim) {
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
inline double ComputeCos::dot_f32(const uint8_t* a, const uint8_t* b, size_t dim) {
    const float* va = reinterpret_cast<const float*>(a);
    const float* vb = reinterpret_cast<const float*>(b);
    double dot = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        dot += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
    }
    return dot;
}

inline double ComputeCos::dot_f16(const uint8_t* a, const uint8_t* b, size_t dim) {
    const float16* va = reinterpret_cast<const float16*>(a);
    const float16* vb = reinterpret_cast<const float16*>(b);
    double dot = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        dot += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
    }
    return dot;
}

inline double ComputeCos::dot_i16(const uint8_t* a, const uint8_t* b, size_t dim) {
    const int16_t* va = reinterpret_cast<const int16_t*>(a);
    const int16_t* vb = reinterpret_cast<const int16_t*>(b);
    double dot = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        dot += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
    }
    return dot;
}

// Query-norm variants walk the candidate once, accumulating both its self norm
// and its dot product with the query because the query norm was already
// computed once by the scanner before entering the hot scan loop.
inline double ComputeCos::dist_f32_with_query_norm(const uint8_t* a, const uint8_t* b, size_t dim,
        double query_norm_sq) {
    const float* va = reinterpret_cast<const float*>(a);
    const float* vb = reinterpret_cast<const float*>(b);
    double dot = 0.0;
    double norm_a = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        const double bi = static_cast<double>(vb[i]);
        dot += ai * bi;
        norm_a += ai * ai;
    }
    return finalize_cosine_distance(dot, norm_a, query_norm_sq);
}

inline double ComputeCos::dist_f16_with_query_norm(const uint8_t* a, const uint8_t* b, size_t dim,
        double query_norm_sq) {
    const float16* va = reinterpret_cast<const float16*>(a);
    const float16* vb = reinterpret_cast<const float16*>(b);
    double dot = 0.0;
    double norm_a = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        const double bi = static_cast<double>(vb[i]);
        dot += ai * bi;
        norm_a += ai * ai;
    }
    return finalize_cosine_distance(dot, norm_a, query_norm_sq);
}

inline double ComputeCos::dist_i16_with_query_norm(const uint8_t* a, const uint8_t* b, size_t dim,
        double query_norm_sq) {
    const int16_t* va = reinterpret_cast<const int16_t*>(a);
    const int16_t* vb = reinterpret_cast<const int16_t*>(b);
    double dot = 0.0;
    double norm_a = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        const double bi = static_cast<double>(vb[i]);
        dot += ai * bi;
        norm_a += ai * ai;
    }
    return finalize_cosine_distance(dot, norm_a, query_norm_sq);
}

} // namespace sketch2
