// Shared deterministic vector values for compute tests and benchmarks.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "core/utils/float8.h"
#include "core/utils/shared_types.h"

namespace sketch2 {

inline uint16_t float_to_f16_bits(float value) {
    const float16 h = static_cast<float16>(value);
    uint16_t bits = 0;
    static_assert(sizeof(bits) == sizeof(h));
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

inline void fill_f32(float* a, float* b, size_t dim, uint32_t seed) {
    for (size_t i = 0; i < dim; ++i) {
        const int32_t ai = static_cast<int32_t>((i * 17 + seed * 13) % 401) - 200;
        const int32_t bi = static_cast<int32_t>((i * 29 + seed * 7) % 401) - 200;
        a[i] = static_cast<float>(ai) * 0.125f + static_cast<float>((i + seed) % 5) * 0.03125f;
        b[i] = static_cast<float>(bi) * 0.125f - static_cast<float>((i + seed) % 3) * 0.0625f;
    }
}

inline void fill_i16(int16_t* a, int16_t* b, size_t dim, uint32_t seed) {
    for (size_t i = 0; i < dim; ++i) {
        const int32_t ai = static_cast<int32_t>((i * 977 + seed * 131) % 65536) - 32768;
        const int32_t bi = static_cast<int32_t>((i * 733 + seed * 191) % 65536) - 32768;
        a[i] = static_cast<int16_t>(ai);
        b[i] = static_cast<int16_t>(bi);
    }
}

inline void fill_f16(float16* a, float16* b, size_t dim, uint32_t seed) {
    for (size_t i = 0; i < dim; ++i) {
        const int32_t ai = static_cast<int32_t>((i * 17 + seed * 13) % 401) - 200;
        const int32_t bi = static_cast<int32_t>((i * 29 + seed * 7) % 401) - 200;
        a[i] = static_cast<float16>(
            static_cast<float>(ai) * 0.125f + static_cast<float>((i + seed) % 5) * 0.03125f);
        b[i] = static_cast<float16>(
            static_cast<float>(bi) * 0.125f - static_cast<float>((i + seed) % 3) * 0.0625f);
    }
}

// F8-4's scalar/kernel tests use the same bounded E5M2 values as storage
// generators.  Keep the mixing here intentionally simple; the canonical
// value set and selection operation live in float8_codebook.
inline void fill_f8(float8* a, float8* b, size_t dim, uint32_t seed) {
    const uint64_t seed_digit = static_cast<uint64_t>(seed) % float8_codebook::kSize;
    for (size_t i = 0; i < dim; ++i) {
        const uint64_t index_digit = static_cast<uint64_t>(i % float8_codebook::kSize);
        const size_t a_index = static_cast<size_t>(
            (index_digit * 17U + seed_digit * 13U) % float8_codebook::kSize);
        const size_t b_index = static_cast<size_t>(
            (index_digit * 29U + seed_digit * 7U) % float8_codebook::kSize);
        a[i] = float8_codebook::value_at(a_index);
        b[i] = float8_codebook::value_at(b_index);
    }
}

} // namespace sketch2
