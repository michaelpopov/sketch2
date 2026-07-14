// Shared helper utilities for compute kernel unit tests.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "core/compute/compute_value_helpers.h"
#include "core/utils/checked_arithmetic.h"
#include "core/utils/shared_types.h"
#include "core/utils/utest_float8_helpers.h"

namespace sketch2 {
namespace test {

template <typename T>
struct TestBuffer {
    std::vector<T> storage;
    T* ptr = nullptr;
};

template <typename T>
TestBuffer<T> make_buffer(size_t dim, size_t misalign_bytes) {
    constexpr size_t kAlignment = 32;
    static_assert(kAlignment % sizeof(T) == 0);

    if (misalign_bytes % sizeof(T) != 0) {
        throw std::invalid_argument("test buffer misalignment must preserve element alignment");
    }

    const size_t misalign_elements = misalign_bytes / sizeof(T);
    size_t storage_elements = 0;
    if (add_overflows(dim, kAlignment / sizeof(T), &storage_elements) ||
            add_overflows(storage_elements, misalign_elements, &storage_elements)) {
        throw std::length_error("test buffer size overflow");
    }

    TestBuffer<T> out;
    out.storage.resize(storage_elements);

    // Keep the pointer derived from the typed allocation so the compiler can
    // retain both object lifetime and bounds information through optimization.
    const uintptr_t address = reinterpret_cast<uintptr_t>(out.storage.data());
    const size_t align_bytes = (kAlignment - address % kAlignment) % kAlignment;
    if (align_bytes % sizeof(T) != 0) {
        throw std::logic_error("test buffer element size is incompatible with alignment");
    }
    out.ptr = out.storage.data() + align_bytes / sizeof(T) + misalign_elements;
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
