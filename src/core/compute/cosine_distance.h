// Shared cosine-distance finalizers used across compute backends.

#pragma once

#include <algorithm>
#include <cmath>

namespace sketch2 {

// Normalize the raw cosine ingredients into the public distance contract,
// including the special zero-vector behavior shared by all backends.
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

// Readers that persist inverse norms can skip the sqrt/divide work and still
// reuse the same zero-vector and clamping semantics as the raw-norm path.
inline double finalize_cosine_distance_from_inverse_norms(double dot, double inv_norm_a, double inv_norm_b) {
    if (inv_norm_a == 0.0 && inv_norm_b == 0.0) {
        return 0.0;
    }
    if (inv_norm_a == 0.0 || inv_norm_b == 0.0) {
        return 1.0;
    }
    const double cosine = std::clamp(dot * inv_norm_a * inv_norm_b, -1.0, 1.0);
    return 1.0 - cosine;
}

} // namespace sketch2
