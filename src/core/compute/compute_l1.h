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
    static double   dist_i32(const uint8_t* a, const uint8_t* b, size_t dim);
    static float    f16_to_float(uint16_t h);
};

inline float ComputeL1::f16_to_float(uint16_t /*h*/) {
    throw std::runtime_error("f16_to_float not implemented");
}

inline double ComputeL1::dist(const uint8_t* a, const uint8_t* b, DataType type, size_t dim) {
    switch (type) {
        case DataType::f32: return dist_f32(a, b, dim);
        case DataType::f16: return dist_f16(a, b, dim);
        case DataType::i32: return dist_i32(a, b, dim);
        default:            return 0.0;
    }
}

inline double ComputeL1::dist_f32(const uint8_t* a, const uint8_t* b, size_t dim) {
    const float* va = reinterpret_cast<const float*>(a);
    const float* vb = reinterpret_cast<const float*>(b);
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        sum += std::abs(static_cast<double>(va[i]) - static_cast<double>(vb[i]));
    }
    return sum;
}

inline double ComputeL1::dist_f16(const uint8_t* a, const uint8_t* b, size_t dim) {
    const uint16_t* va = reinterpret_cast<const uint16_t*>(a);
    const uint16_t* vb = reinterpret_cast<const uint16_t*>(b);
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        double fa = static_cast<double>(f16_to_float(va[i]));
        double fb = static_cast<double>(f16_to_float(vb[i]));
        sum += std::abs(fa - fb);
    }
    return sum;
}

inline double ComputeL1::dist_i32(const uint8_t* a, const uint8_t* b, size_t dim) {
    const int32_t* va = reinterpret_cast<const int32_t*>(a);
    const int32_t* vb = reinterpret_cast<const int32_t*>(b);
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        sum += std::abs(static_cast<double>(va[i]) - static_cast<double>(vb[i]));
    }
    return sum;
}

} // namespace sketch2
