// Shared raw-F8 helpers for unit-test data construction and reference math.
// These decode byte buffers through float8 and accumulate in double, keeping
// test oracles independent from the production scan/kernel finalizers.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "core/utils/float8.h"

namespace sketch2::test {

inline std::string f8_payload_text(const std::vector<uint8_t>& bits) {
    std::ostringstream out;
    out << std::setprecision(9);
    for (size_t i = 0; i < bits.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << static_cast<float>(float8::from_bits(bits[i]));
    }
    return out.str();
}

inline std::vector<uint8_t> f8_ordinal_bytes(uint64_t ordinal, size_t dim) {
    std::vector<float8> values(dim);
    if (!float8_codebook::fill_ordinal_vector(ordinal, values.data(), values.size())) {
        return {};
    }

    std::vector<uint8_t> bits;
    bits.reserve(values.size());
    for (const float8 value : values) {
        bits.push_back(value.to_bits());
    }
    return bits;
}

inline double decode_f8(uint8_t bits) {
    return static_cast<double>(static_cast<float>(float8::from_bits(bits)));
}

inline double reference_f8_dot(const uint8_t* a, const uint8_t* b, size_t dim) {
    double dot = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        dot += decode_f8(a[i]) * decode_f8(b[i]);
    }
    return dot;
}

inline double reference_f8_dot(
        const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    assert(a.size() == b.size());
    return reference_f8_dot(a.data(), b.data(), std::min(a.size(), b.size()));
}

inline double reference_f8_dot_abs_sum(const uint8_t* a, const uint8_t* b, size_t dim) {
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        sum += std::abs(decode_f8(a[i]) * decode_f8(b[i]));
    }
    return sum;
}

inline double reference_f8_l2(const uint8_t* a, const uint8_t* b, size_t dim) {
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double d = decode_f8(a[i]) - decode_f8(b[i]);
        sum += d * d;
    }
    return sum;
}

inline double reference_f8_squared_norm(const uint8_t* data, size_t dim) {
    double norm = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double value = decode_f8(data[i]);
        norm += value * value;
    }
    return norm;
}

inline double reference_f8_squared_norm(const std::vector<uint8_t>& bits) {
    return reference_f8_squared_norm(bits.data(), bits.size());
}

inline double reference_f8_cosine_distance(const uint8_t* a, const uint8_t* b, size_t dim) {
    const double dot = reference_f8_dot(a, b, dim);
    const double norm_a = reference_f8_squared_norm(a, dim);
    const double norm_b = reference_f8_squared_norm(b, dim);

    if (norm_a == 0.0 && norm_b == 0.0) {
        return 0.0;
    }
    if (norm_a == 0.0 || norm_b == 0.0) {
        return 1.0;
    }

    const double cosine = std::clamp(dot / std::sqrt(norm_a * norm_b), -1.0, 1.0);
    return 1.0 - cosine;
}

} // namespace sketch2::test
