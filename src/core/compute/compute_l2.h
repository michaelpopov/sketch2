// Implements the portable L2-distance primitives.

#pragma once
#include "core/compute/compute.h"
#include "core/utils/singleton.h"
#include <cassert>
#include <cmath>
#include <cstdint>

#if defined(__x86_64__) || defined(__i386__)
#if defined(SKETCH_ENABLE_AVX2) && SKETCH_ENABLE_AVX2
#include "compute_l2_avx2.h"
#endif
#if (defined(SKETCH_ENABLE_AVX512F) && SKETCH_ENABLE_AVX512F) || \
    (defined(SKETCH_ENABLE_AVX512VNNI) && SKETCH_ENABLE_AVX512VNNI)
#include "compute_l2_avx512.h"
#endif
#endif

#if defined(__aarch64__)
#include "compute_l2_neon.h"
#endif

namespace sketch2 {

// Computes squared L2 distance between two vectors.
// ComputeL2 exists to group the portable squared-L2 implementation and the
// typed dispatch helpers used by the scanner. It is the scalar fallback for
// platforms without a vectorized L2 backend.
class ComputeL2 : public ICompute {
public:
    using DistFn = double (*)(const uint8_t*, const uint8_t*, size_t);

    double dist(const uint8_t *a, const uint8_t *b, DataType type, size_t dim) override;
    // Runtime backend selection intentionally stays here with the typed
    // entrypoints so scanner/template code can resolve a concrete kernel once
    // and then stay on that path. This is not a "free" helper: it reads the
    // process-wide ComputeUnit from the singleton, so callers should treat it
    // as setup-time dispatch and cache the result if they plan to reuse it.
    static DistFn resolve_dist(DataType type);

    // Typed entrypoints used by scanner template dispatch and scalar fallback.
    static double dist_f32(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_f16(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_i16(const uint8_t *a, const uint8_t *b, size_t dim);
};

inline double ComputeL2::dist(const uint8_t *a, const uint8_t *b, DataType type, size_t dim) {
    DistFn fn = resolve_dist(type);
    return fn(a, b, dim);
}

inline ComputeL2::DistFn ComputeL2::resolve_dist(DataType type) {
    validate_type(type);
    switch (get_singleton().compute_unit().kind()) {
#if defined(SKETCH_ENABLE_AVX512VNNI) && SKETCH_ENABLE_AVX512VNNI && (defined(__x86_64__) || defined(__i386__))
        case ComputeBackendKind::avx512_vnni:
            switch (type) {
                case DataType::f32: return &ComputeL2_AVX512_VNNI::dist_f32;
                case DataType::f16: return &ComputeL2_AVX512_VNNI::dist_f16;
                case DataType::i16: return &ComputeL2_AVX512_VNNI::dist_i16;
                default: break;
            }
            break;
#endif
#if defined(SKETCH_ENABLE_AVX512F) && SKETCH_ENABLE_AVX512F && (defined(__x86_64__) || defined(__i386__))
        case ComputeBackendKind::avx512f:
            switch (type) {
                case DataType::f32: return &ComputeL2_AVX512::dist_f32;
                case DataType::f16: return &ComputeL2_AVX512::dist_f16;
                case DataType::i16: return &ComputeL2_AVX512::dist_i16;
                default: break;
            }
            break;
#endif
#if defined(SKETCH_ENABLE_AVX2) && SKETCH_ENABLE_AVX2 && (defined(__x86_64__) || defined(__i386__))
        case ComputeBackendKind::avx2:
            switch (type) {
                case DataType::f32: return &ComputeL2_AVX2::dist_f32;
                case DataType::f16: return &ComputeL2_AVX2::dist_f16;
                case DataType::i16: return &ComputeL2_AVX2::dist_i16;
                default: break;
            }
            break;
#endif
#if defined(__aarch64__)
        case ComputeBackendKind::neon:
            switch (type) {
                case DataType::f32: return &ComputeL2_Neon::dist_f32;
                case DataType::f16: return &ComputeL2_Neon::dist_f16;
                case DataType::i16: return &ComputeL2_Neon::dist_i16;
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
            throw std::runtime_error("ComputeL2::resolve_dist: unsupported data type");
    }
}

inline double ComputeL2::dist_f32(const uint8_t* a, const uint8_t* b, size_t dim) {
    const float* va = reinterpret_cast<const float*>(a);
    const float* vb = reinterpret_cast<const float*>(b);
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double d = static_cast<double>(va[i]) - static_cast<double>(vb[i]);
        sum += d * d;
    }
    return sum;
}

inline double ComputeL2::dist_f16(const uint8_t* a, const uint8_t* b, size_t dim) {
    const float16* va = reinterpret_cast<const float16*>(a);
    const float16* vb = reinterpret_cast<const float16*>(b);
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double d = static_cast<double>(va[i]) - static_cast<double>(vb[i]);
        sum += d * d;
    }
    return sum;
}

inline double ComputeL2::dist_i16(const uint8_t* a, const uint8_t* b, size_t dim) {
    const int16_t* va = reinterpret_cast<const int16_t*>(a);
    const int16_t* vb = reinterpret_cast<const int16_t*>(b);
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const int64_t d = static_cast<int64_t>(va[i]) - static_cast<int64_t>(vb[i]);
        sum += static_cast<double>(d * d);
    }
    return sum;
}

} // namespace sketch2
