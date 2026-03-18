// Shared helper utilities for compute unit tests.

#pragma once
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "core/utils/arch_detection.h"
#include "core/compute/compute.h"

namespace sketch2 {
namespace test {

template <typename T>
struct TestBuffer {
    std::vector<uint8_t> storage;
    T *ptr = nullptr;
};

template <typename T>
TestBuffer<T> make_buffer(size_t dim, size_t misalign_bytes) {
    TestBuffer<T> out;
    out.storage.resize(dim * sizeof(T) + 64 + misalign_bytes);
    uintptr_t p = reinterpret_cast<uintptr_t>(out.storage.data());
    p = (p + 31u) & ~uintptr_t(31u);
    p += misalign_bytes;
    out.ptr = reinterpret_cast<T *>(p);
    return out;
}

// --- Reference Implementations ---

template <typename T>
double reference_l1(const T *a, const T *b, size_t dim) {
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        sum += std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    }
    return sum;
}

template <typename T>
double reference_l2(const T *a, const T *b, size_t dim) {
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum += d * d;
    }
    return sum;
}

template <>
inline double reference_l2<int16_t>(const int16_t *a, const int16_t *b, size_t dim) {
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const int64_t d = static_cast<int64_t>(a[i]) - static_cast<int64_t>(b[i]);
        sum += static_cast<double>(d * d);
    }
    return sum;
}

template <typename T>
double reference_dot(const T *a, const T *b, size_t dim) {
    double dot = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return dot;
}

template <typename T>
double reference_squared_norm(const T *a, size_t dim) {
    double norm = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double ai = static_cast<double>(a[i]);
        norm += ai * ai;
    }
    return norm;
}

template <typename T>
double reference_cosine_distance(const T *a, const T *b, size_t dim) {
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

// --- Data Generation ---

inline void fill_f32(float *a, float *b, size_t dim, uint32_t seed) {
    for (size_t i = 0; i < dim; ++i) {
        const int32_t ai = static_cast<int32_t>((i * 17 + seed * 13) % 401) - 200;
        const int32_t bi = static_cast<int32_t>((i * 29 + seed * 7) % 401) - 200;
        a[i] = static_cast<float>(ai) * 0.125f + static_cast<float>((i + seed) % 5) * 0.03125f;
        b[i] = static_cast<float>(bi) * 0.125f - static_cast<float>((i + seed) % 3) * 0.0625f;
    }
}

inline void fill_i16(int16_t *a, int16_t *b, size_t dim, uint32_t seed) {
    for (size_t i = 0; i < dim; ++i) {
        const int32_t ai = static_cast<int32_t>((i * 977 + seed * 131) % 65536) - 32768;
        const int32_t bi = static_cast<int32_t>((i * 733 + seed * 191) % 65536) - 32768;
        a[i] = static_cast<int16_t>(ai);
        b[i] = static_cast<int16_t>(bi);
    }
}

#if defined(__FLT16_MANT_DIG__) || (defined(SKETCH_ARCH_ARM64) && !defined(_MSC_VER))
inline void fill_f16(float16 *a, float16 *b, size_t dim, uint32_t seed) {
    for (size_t i = 0; i < dim; ++i) {
        const int32_t ai = static_cast<int32_t>((i * 17 + seed * 13) % 401) - 200;
        const int32_t bi = static_cast<int32_t>((i * 29 + seed * 7) % 401) - 200;
        a[i] = static_cast<float16>(static_cast<float>(ai) * 0.125f + static_cast<float>((i + seed) % 5) * 0.03125f);
        b[i] = static_cast<float16>(static_cast<float>(bi) * 0.125f - static_cast<float>((i + seed) % 3) * 0.0625f);
    }
}
#endif

} // namespace test
} // namespace sketch2
