// Implements the portable L2-distance primitives.

#pragma once
#include "core/compute/compute.h"
#include <cassert>
#include <cmath>
#include <cstdint>

#if defined(__AVX2__)
#include "compute_l2_avx2.h"
#elif defined(__aarch64__)
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
    switch (type) {
#if defined(__AVX2__)
    case DataType::f32: return &ComputeL2_AVX2::dist_f32;
    case DataType::f16: return &ComputeL2_AVX2::dist_f16;
    case DataType::i16: return &ComputeL2_AVX2::dist_i16;
#elif defined(__aarch64__)
    case DataType::f32: return &ComputeL2_Neon::dist_f32;
    case DataType::f16: return &ComputeL2_Neon::dist_f16;
    case DataType::i16: return &ComputeL2_Neon::dist_i16;
#else
    case DataType::f32: return &dist_f32;
    case DataType::f16: return &dist_f16;
    case DataType::i16: return &dist_i16;
#endif
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
