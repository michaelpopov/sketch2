// Per-query context structs passed to scan kernels.

#pragma once

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
