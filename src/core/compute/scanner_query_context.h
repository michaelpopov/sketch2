// Shared scanner query contexts and metric finalizers.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace sketch2 {

inline double query_inverse_norm(double query_norm_sq) {
    if (query_norm_sq == 0.0) {
        return 0.0;
    }
    return 1.0 / std::sqrt(query_norm_sq);
}

inline double l2_dist_from_dot_and_norms(
        double dot, double norm_a_sq, double norm_b_sq) {
    return std::max(0.0, norm_a_sq + norm_b_sq - (2.0 * dot));
}

struct QueryDistContext {
    const uint8_t* vec = nullptr;
    size_t dim = 0;
};

struct QueryDotContext {
    const uint8_t* vec = nullptr;
    size_t dim = 0;
};

struct QueryCosContext {
    const uint8_t* vec = nullptr;
    size_t dim = 0;
    double norm_sq = 0.0;
    double inv_norm = 0.0;
};

struct QueryL2Context {
    const uint8_t* vec = nullptr;
    size_t dim = 0;
    double norm_sq = 0.0;
};

} // namespace sketch2
