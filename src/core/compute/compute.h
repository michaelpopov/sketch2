// Defines common ranked-result types and the base compute interface.

#pragma once
#include "utils/shared_types.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace sketch2 {

struct DistItem {
    uint64_t id;
    double   score;
};

inline bool smaller_score_is_better(DistFunc func) {
    switch (func) {
        case DistFunc::DOT:
            return false;
        case DistFunc::L2:
        case DistFunc::COS:
            return true;
        default:
            assert(false && "smaller_score_is_better: unsupported distance function");
            // Fall back to ascending ordering to keep comparators noexcept-like.
            return true;
    }
}

inline bool dist_item_is_better(DistFunc func, const DistItem& a, const DistItem& b) {
    if (a.score != b.score) {
        return smaller_score_is_better(func) ? (a.score < b.score) : (a.score > b.score);
    }
    return a.id < b.id;
}

struct DistItemCompare {
    explicit DistItemCompare(DistFunc func_) : func(func_) {}

    DistFunc func;

    bool operator()(const DistItem& a, const DistItem& b) const {
        return dist_item_is_better(func, a, b);
    }
};

// ICompute exists as the common interface for distance calculators so higher
// layers can talk to scalar and architecture-specific kernels through one API.
// It provides the single virtual distance entry point shared by compute backends.
class ICompute {
public:
    virtual ~ICompute() = default;
    // Returns the distance between two vectors of the given type and dimension.
    virtual double dist(const uint8_t*, const uint8_t*, DataType, size_t /*dim*/) = 0;
};

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
