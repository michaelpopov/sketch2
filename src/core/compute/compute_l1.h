#pragma once
#include "core/compute/compute.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace sketch2 {

// Computes L1 (Manhattan) distance between two vectors.
class ComputeL1 : public ICompute {
public:
    double dist(const uint8_t* a, const uint8_t* b, DataType type, size_t dim) override;

private:
    static double   dist_f32(const uint8_t* a, const uint8_t* b, size_t dim);
    static double   dist_f16(const uint8_t* a, const uint8_t* b, size_t dim);
    static double   dist_i16(const uint8_t* a, const uint8_t* b, size_t dim);
};

inline double ComputeL1::dist(const uint8_t* a, const uint8_t* b, DataType type, size_t dim) {
    switch (type) {
        case DataType::f32: return dist_f32(a, b, dim);
        case DataType::f16: return dist_f16(a, b, dim);
        case DataType::i16: return dist_i16(a, b, dim);
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
        sum += std::abs(va[i] - vb[i]);
    }
    return sum;
}

inline double ComputeL1::dist_i16(const uint8_t* a, const uint8_t* b, size_t dim) {
    const int16_t* va = reinterpret_cast<const int16_t*>(a);
    const int16_t* vb = reinterpret_cast<const int16_t*>(b);
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        sum += std::abs(va[i] - vb[i]);
    }
    return sum;
}

} // namespace sketch2
