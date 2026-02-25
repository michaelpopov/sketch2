#pragma once
#include "core/compute/compute.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#if defined(__AVX2__)
#include "compute_l1_avx2.h"
#elif defined(__aarch64__)
#include "compute_l1_neon.h"
#endif

namespace sketch2 {

// Computes L1 (Manhattan) distance between two vectors.
class ComputeL1 : public ICompute {
public:
    double dist(const uint8_t *a, const uint8_t *b, DataType type, size_t dim) override;

private:
    static double dist_f32(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_f16(const uint8_t *a, const uint8_t *b, size_t dim);
    static double dist_i16(const uint8_t *a, const uint8_t *b, size_t dim);
};

inline double ComputeL1::dist(const uint8_t *a, const uint8_t *b, DataType type, size_t dim) {
    switch (type) {
#if defined(__AVX2__)
    case DataType::f32: return ComputeL1_AVX2::dist_f32(a, b, dim);
    case DataType::f16: return ComputeL1_AVX2::dist_f16(a, b, dim);
    case DataType::i16: return ComputeL1_AVX2::dist_i16(a, b, dim);
#elif defined(__aarch64__)
    case DataType::f32: return ComputeL1_Neon::dist_f32(a, b, dim);
    case DataType::f16: return ComputeL1_Neon::dist_f16(a, b, dim);
    case DataType::i16: return ComputeL1_Neon::dist_i16(a, b, dim);
#else
    case DataType::f32: return dist_f32(a, b, dim);
    case DataType::f16: return dist_f16(a, b, dim);
    case DataType::i16: return dist_i16(a, b, dim);
#endif
    default:            return 0.0;
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
