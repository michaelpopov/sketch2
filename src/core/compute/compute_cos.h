#pragma once
#include "core/compute/compute.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#if defined(__AVX2__)
#include "compute_cos_avx2.h"
#elif defined(__aarch64__)
#include "compute_cos_neon.h"
#endif

namespace sketch2 {

// Computes cosine distance between two vectors.
class ComputeCos : public ICompute {
public:
    using DistFn = double (*)(const uint8_t*, const uint8_t*, size_t);

    double dist(const uint8_t *a, const uint8_t *b, DataType type, size_t dim) override;
    static DistFn resolve_dist(DataType type);

private:
    static double dist_f32(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_f16(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_i16(const uint8_t *a, const uint8_t *b, size_t dim);
};

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

inline double ComputeCos::dist(const uint8_t *a, const uint8_t *b, DataType type, size_t dim) {
    DistFn fn = resolve_dist(type);
    return fn(a, b, dim);
}

inline ComputeCos::DistFn ComputeCos::resolve_dist(DataType type) {
    validate_type(type);
    switch (type) {
#if defined(__AVX2__)
    case DataType::f32: return &ComputeCos_AVX2::dist_f32;
    case DataType::f16: return &ComputeCos_AVX2::dist_f16;
    case DataType::i16: return &ComputeCos_AVX2::dist_i16;
#elif defined(__aarch64__)
    case DataType::f32: return &ComputeCos_Neon::dist_f32;
    case DataType::f16: return &ComputeCos_Neon::dist_f16;
    case DataType::i16: return &ComputeCos_Neon::dist_i16;
#else
    case DataType::f32: return &dist_f32;
    case DataType::f16: return &dist_f16;
    case DataType::i16: return &dist_i16;
#endif
    default:
        assert(false);
        throw std::runtime_error("ComputeCos::resolve_dist: unsupported data type");
    }
}

inline double ComputeCos::dist_f32(const uint8_t* a, const uint8_t* b, size_t dim) {
    const float* va = reinterpret_cast<const float*>(a);
    const float* vb = reinterpret_cast<const float*>(b);
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        const double bi = static_cast<double>(vb[i]);
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    return finalize_cosine_distance(dot, norm_a, norm_b);
}

inline double ComputeCos::dist_f16(const uint8_t* a, const uint8_t* b, size_t dim) {
    const float16* va = reinterpret_cast<const float16*>(a);
    const float16* vb = reinterpret_cast<const float16*>(b);
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        const double bi = static_cast<double>(vb[i]);
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    return finalize_cosine_distance(dot, norm_a, norm_b);
}

inline double ComputeCos::dist_i16(const uint8_t* a, const uint8_t* b, size_t dim) {
    const int16_t* va = reinterpret_cast<const int16_t*>(a);
    const int16_t* vb = reinterpret_cast<const int16_t*>(b);
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double ai = static_cast<double>(va[i]);
        const double bi = static_cast<double>(vb[i]);
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    return finalize_cosine_distance(dot, norm_a, norm_b);
}

} // namespace sketch2
