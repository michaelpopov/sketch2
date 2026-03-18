// Implements the portable L1-distance primitives.

#pragma once
#include "core/compute/compute.h"
#include "core/utils/singleton.h"
#include "core/utils/arch_detection.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#if SKETCH_HAS_AVX2
#include "compute_l1_avx2.h"
#endif
#if SKETCH_HAS_AVX512
#include "compute_l1_avx512.h"
#endif

#if SKETCH_HAS_NEON
#include "compute_l1_neon.h"
#endif

namespace sketch2 {

// Computes L1 (Manhattan) distance between two vectors.
// ComputeL1 exists to group the portable L1-distance implementation and the
// typed dispatch helpers used by the scanner. It serves as the scalar fallback
// when no architecture-specific implementation is selected.
class ComputeL1 : public ICompute {
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

inline double ComputeL1::dist(const uint8_t *a, const uint8_t *b, DataType type, size_t dim) {
    DistFn fn = resolve_dist(type);
    return fn(a, b, dim);
}

inline ComputeL1::DistFn ComputeL1::resolve_dist(DataType type) {
    switch (get_singleton().compute_unit().kind()) {
#if SKETCH_HAS_AVX512VNNI
        case ComputeBackendKind::avx512_vnni:
            switch (type) {
                case DataType::f32: return &ComputeL1_AVX512_VNNI::dist_f32;
                case DataType::f16: return &ComputeL1_AVX512_VNNI::dist_f16;
                case DataType::i16: return &ComputeL1_AVX512_VNNI::dist_i16;
                default: break;
            }
            break;
#endif
#if SKETCH_HAS_AVX512F
        case ComputeBackendKind::avx512f:
            switch (type) {
                case DataType::f32: return &ComputeL1_AVX512::dist_f32;
                case DataType::f16: return &ComputeL1_AVX512::dist_f16;
                case DataType::i16: return &ComputeL1_AVX512::dist_i16;
                default: break;
            }
            break;
#endif
#if SKETCH_HAS_AVX2
        case ComputeBackendKind::avx2:
            switch (type) {
                case DataType::f32: return &ComputeL1_AVX2::dist_f32;
                case DataType::f16: return &ComputeL1_AVX2::dist_f16;
                case DataType::i16: return &ComputeL1_AVX2::dist_i16;
                default: break;
            }
            break;
#endif
#if SKETCH_HAS_NEON
        case ComputeBackendKind::neon:
            switch (type) {
                case DataType::f32: return &ComputeL1_Neon::dist_f32;
                case DataType::f16: return &ComputeL1_Neon::dist_f16;
                case DataType::i16: return &ComputeL1_Neon::dist_i16;
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
            throw std::runtime_error("ComputeL1::resolve_dist: unsupported data type");
    }
}

inline double ComputeL1::dist_f32(const uint8_t* a, const uint8_t* b, size_t dim) {
    const float* va = reinterpret_cast<const float*>(a);
    const float* vb = reinterpret_cast<const float*>(b);
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        sum += std::abs(va[i] - vb[i]);
    }
    return sum;
}

inline double ComputeL1::dist_f16(const uint8_t* a, const uint8_t* b, size_t dim) {
    const float16* va = reinterpret_cast<const float16*>(a);
    const float16* vb = reinterpret_cast<const float16*>(b);
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        sum += std::abs(static_cast<double>(va[i]) - static_cast<double>(vb[i]));
    }
    return sum;
}

inline double ComputeL1::dist_i16(const uint8_t* a, const uint8_t* b, size_t dim) {
    const int16_t* va = reinterpret_cast<const int16_t*>(a);
    const int16_t* vb = reinterpret_cast<const int16_t*>(b);
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const int diff = static_cast<int>(va[i]) - static_cast<int>(vb[i]);
        sum += std::abs(diff);
    }
    return sum;
}

} // namespace sketch2
