// Distance-metric finalizers: combine intermediate values (dot product,
// stored norms) into the public distance contract for each metric.

#pragma once

#include <algorithm>
#include <cmath>

namespace sketch2 {

// Squared L2: ||a-b||² = ||a||² + ||b||² - 2·a·b. Clamped at zero so floating
// rounding cannot produce a negative distance.
inline double l2_dist_from_dot_and_norms(double dot, double norm_a_sq, double norm_b_sq) {
    return std::max(0.0, norm_a_sq + norm_b_sq - (2.0 * dot));
}

// Cosine distance from raw norms. Includes the zero-vector contract shared by
// all Highway kernels: both zero → distance 0; one zero → distance 1.
inline double cos_dist_from_norms(double dot, double norm_a, double norm_b) {
    if (norm_a == 0.0 && norm_b == 0.0) {
        return 0.0;
    }
    if (norm_a == 0.0 || norm_b == 0.0) {
        return 1.0;
    }
    const double cosine = std::clamp(dot / std::sqrt(norm_a * norm_b), -1.0, 1.0);
    return 1.0 - cosine;
}

// Cosine distance from inverse norms. Readers that persist inverse norms can
// skip the sqrt/divide work and still reuse the same zero-vector semantics.
inline double cos_dist_from_inv_norms(double dot, double inv_norm_a, double inv_norm_b) {
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
