// Defines scalar norm helpers shared by calc and storage code.

#pragma once

#include "utils/shared_types.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace sketch2 {

inline float compute_cosine_inverse_norm(const uint8_t* data, DataType type, size_t dim) {
    double norm_sq = 0.0;
    switch (type) {
        case DataType::f32: {
            const auto* values = reinterpret_cast<const float*>(data);
            for (size_t i = 0; i < dim; ++i) {
                const double value = static_cast<double>(values[i]);
                norm_sq += value * value;
            }
            break;
        }
        case DataType::f16: {
            const auto* values = reinterpret_cast<const float16*>(data);
            for (size_t i = 0; i < dim; ++i) {
                const double value = static_cast<double>(values[i]);
                norm_sq += value * value;
            }
            break;
        }
        case DataType::i16: {
            const auto* values = reinterpret_cast<const int16_t*>(data);
            for (size_t i = 0; i < dim; ++i) {
                const double value = static_cast<double>(values[i]);
                norm_sq += value * value;
            }
            break;
        }
        default:
            throw std::runtime_error("compute_cosine_inverse_norm: unsupported data type");
    }

    if (norm_sq == 0.0) {
        return 0.0f;
    }
    return static_cast<float>(1.0 / std::sqrt(norm_sq));
}

inline float compute_squared_norm(const uint8_t* data, DataType type, size_t dim) {
    double norm_sq = 0.0;
    switch (type) {
        case DataType::f32: {
            const auto* values = reinterpret_cast<const float*>(data);
            for (size_t i = 0; i < dim; ++i) {
                const double value = static_cast<double>(values[i]);
                norm_sq += value * value;
            }
            break;
        }
        case DataType::f16: {
            const auto* values = reinterpret_cast<const float16*>(data);
            for (size_t i = 0; i < dim; ++i) {
                const double value = static_cast<double>(values[i]);
                norm_sq += value * value;
            }
            break;
        }
        case DataType::i16: {
            const auto* values = reinterpret_cast<const int16_t*>(data);
            for (size_t i = 0; i < dim; ++i) {
                const double value = static_cast<double>(values[i]);
                norm_sq += value * value;
            }
            break;
        }
        default:
            throw std::runtime_error("compute_squared_norm: unsupported data type");
    }
    return static_cast<float>(norm_sq);
}

} // namespace sketch2
