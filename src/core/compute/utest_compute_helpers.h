// Shared helper utilities for compute kernel unit tests.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/compute/compute_value_helpers.h"
#include "core/utils/shared_types.h"
#include "core/utils/utest_float8_helpers.h"

namespace sketch2 {
namespace test {

template <typename T>
struct TestBuffer {
    std::vector<uint8_t> storage;
    T* ptr = nullptr;
};

template <typename T>
TestBuffer<T> make_buffer(size_t dim, size_t misalign_bytes) {
    TestBuffer<T> out;
    const size_t data_bytes = dim * sizeof(T);
    out.storage.resize(data_bytes + 64 + misalign_bytes);

    // Keep the actual address calculation in pointer form so GCC can retain
    // the backing allocation's object-size information.
    void* aligned = out.storage.data();
    size_t available = out.storage.size();
    auto* const aligned_bytes = static_cast<uint8_t*>(
        std::align(32U, data_bytes + misalign_bytes, aligned, available));
    assert(aligned_bytes != nullptr);
    assert(misalign_bytes % alignof(T) == 0);
    out.ptr = reinterpret_cast<T*>(aligned_bytes + misalign_bytes);
    return out;
}

template <typename T>
double reference_l2(const T* a, const T* b, size_t dim) {
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum += d * d;
    }
    return sum;
}

template <>
inline double reference_l2<int16_t>(const int16_t* a, const int16_t* b, size_t dim) {
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const int64_t d = static_cast<int64_t>(a[i]) - static_cast<int64_t>(b[i]);
        sum += static_cast<double>(d * d);
    }
    return sum;
}

template <typename T>
double reference_dot(const T* a, const T* b, size_t dim) {
    double dot = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return dot;
}

template <typename T>
double reference_squared_norm(const T* a, size_t dim) {
    double norm = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double ai = static_cast<double>(a[i]);
        norm += ai * ai;
    }
    return norm;
}

template <typename T>
double reference_cosine_distance(const T* a, const T* b, size_t dim) {
    const double dot = reference_dot(a, b, dim);
    const double norm_a = reference_squared_norm(a, dim);
    const double norm_b = reference_squared_norm(b, dim);

    if (norm_a == 0.0 && norm_b == 0.0) {
        return 0.0;
    }
    if (norm_a == 0.0 || norm_b == 0.0) {
        return 1.0;
    }

    const double cosine = std::clamp(dot / std::sqrt(norm_a * norm_b), -1.0, 1.0);
    return 1.0 - cosine;
}

// The f8 inputs are already exact grid values. These bounds model only the
// f32 fused multiply-add and lane-reduction order used by Highway, including
// cancellation in DOT via the sum of absolute products.
inline double f8_f32_accumulation_tolerance(double absolute_sum, size_t dim) {
    constexpr double kF32Epsilon = 0x1p-23;
    constexpr double kReductionSafetyFactor = 8.0;
    const double operations = static_cast<double>(dim + 4);
    return std::max(1e-6,
        kReductionSafetyFactor * operations * kF32Epsilon * std::max(1.0, absolute_sum));
}

inline double f8_cosine_tolerance(const uint8_t* a, const uint8_t* b, size_t dim) {
    const double dot_abs_sum = reference_f8_dot_abs_sum(a, b, dim);
    const double norm_a = reference_f8_squared_norm(a, dim);
    const double norm_b = reference_f8_squared_norm(b, dim);
    const double denominator = std::sqrt(norm_a * norm_b);
    const double dot_error = f8_f32_accumulation_tolerance(dot_abs_sum, dim);
    const double norm_relative_error = 0.5 * (
        f8_f32_accumulation_tolerance(norm_a, dim) / norm_a +
        f8_f32_accumulation_tolerance(norm_b, dim) / norm_b);
    return std::max(1e-6, 4.0 * (
        dot_error / denominator + dot_abs_sum / denominator * norm_relative_error));
}

} // namespace test
} // namespace sketch2
