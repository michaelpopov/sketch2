// Defines scalar norm helpers shared by calc and storage code.

#pragma once

#include "utils/shared_types.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace sketch2 {

template <typename T>
inline double sum_squares(const uint8_t* data, size_t dim) {
    const auto* values = reinterpret_cast<const T*>(data);
    double norm_sq = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double value = static_cast<double>(values[i]);
        norm_sq += value * value;
    }
    return norm_sq;
}

inline double compute_norm_sq(const uint8_t* data, DataType type, size_t dim, const char* context) {
    switch (type) {
        case DataType::f32:
            return sum_squares<float>(data, dim);
        case DataType::f16:
            return sum_squares<float16>(data, dim);
        case DataType::i16:
            return sum_squares<int16_t>(data, dim);
        default:
            throw std::runtime_error(std::string(context) + ": unsupported data type");
    }
}

inline double compute_cosine_inverse_norm(const uint8_t* data, DataType type, size_t dim) {
    const double norm_sq = compute_norm_sq(data, type, dim, "compute_cosine_inverse_norm");

    if (norm_sq == 0.0) {
        return 0.0;
    }
    return 1.0 / std::sqrt(norm_sq);
}

inline double compute_squared_norm(const uint8_t* data, DataType type, size_t dim) {
    return compute_norm_sq(data, type, dim, "compute_squared_norm");
}

} // namespace sketch2
