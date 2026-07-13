// E5M2 float8 conversion and value-type support.
//
// E5M2 is the numerically high byte of an IEEE binary16 value.  Raw storage
// remains byte-oriented; this type is for values that are actually constructed
// as float8 objects.

#pragma once

#include "utils/shared_types.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace sketch2 {

inline constexpr float kFloat8Max = 57344.0f;

static_assert(sizeof(float16) == 2);
static_assert(__FLT16_MANT_DIG__ == 11);
static_assert(__FLT16_MAX_EXP__ == 16);

namespace float8_detail {

inline uint32_t float_bits(float value) noexcept {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline uint16_t float16_bits(float16 value) noexcept {
    uint16_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline float16 float16_from_bits(uint16_t bits) noexcept {
    float16 value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// Rounds an unsigned integer right by shift bits using round-to-nearest-even.
inline uint32_t round_shift_right_rne(uint32_t value, unsigned shift) noexcept {
    if (shift == 0) {
        return value;
    }
    if (shift >= 32) {
        return 0;
    }

    const uint64_t divisor = uint64_t{1} << shift;
    const uint32_t retained = value >> shift;
    const uint64_t remainder = static_cast<uint64_t>(value) & (divisor - 1);
    const uint64_t halfway = divisor >> 1;
    return retained + static_cast<uint32_t>(
        remainder > halfway || (remainder == halfway && (retained & 1U) != 0));
}

// This is a bit-level implementation rather than a native cast so its RNE
// result does not depend on the process floating-point rounding mode.
inline uint16_t f32_bits_to_f16_rne_bits(uint32_t bits) noexcept {
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000U);
    const uint32_t exponent = (bits >> 23) & 0xffU;
    const uint32_t fraction = bits & 0x7fffffU;

    if (exponent == 0xffU) {
        if (fraction == 0) {
            return static_cast<uint16_t>(sign | 0x7c00U);
        }

        // Numeric ingest rejects NaN before this point.  Keep unchecked
        // conversion deterministic and preserve a quiet f16 NaN instead of
        // letting a truncated payload turn into infinity.
        uint16_t payload = static_cast<uint16_t>(fraction >> 13);
        payload = static_cast<uint16_t>(payload | 0x0200U);
        return static_cast<uint16_t>(sign | 0x7c00U | payload);
    }

    // Every f32 subnormal is strictly below half the smallest f16 subnormal.
    // Returning the sign bit also preserves negative zero.
    if (exponent == 0) {
        return sign;
    }

    const int unbiased_exponent = static_cast<int>(exponent) - 127;
    if (unbiased_exponent >= 16) {
        return static_cast<uint16_t>(sign | 0x7c00U);
    }

    if (unbiased_exponent >= -14) {
        uint32_t f16_exponent = static_cast<uint32_t>(unbiased_exponent + 15);
        uint32_t f16_fraction = round_shift_right_rne(fraction, 13);

        // A rounded fraction can carry into the implicit bit and then the
        // exponent.  At exponent 31 that correctly becomes infinity.
        if (f16_fraction == 0x0400U) {
            f16_fraction = 0;
            ++f16_exponent;
            if (f16_exponent >= 0x1fU) {
                return static_cast<uint16_t>(sign | 0x7c00U);
            }
        }

        return static_cast<uint16_t>(sign | (f16_exponent << 10) | f16_fraction);
    }

    // In f16-subnormal units (2^-24), an f32 normal with exponent e has to
    // drop -(e + 1) bits from its explicit 24-bit significand.
    const uint32_t significand = 0x800000U | fraction;
    const unsigned shift = static_cast<unsigned>(-(unbiased_exponent + 1));
    const uint32_t f16_fraction = round_shift_right_rne(significand, shift);
    if (f16_fraction >= 0x0400U) {
        // Rounding a subnormal up to 1024 yields the smallest f16 normal.
        return static_cast<uint16_t>(sign | 0x0400U);
    }
    return static_cast<uint16_t>(sign | f16_fraction);
}

} // namespace float8_detail

// Normative f32 -> f16 stage of E5M2 encoding.  The explicit bit logic pins
// IEEE RNE behavior regardless of any ambient non-default rounding mode.
inline float16 f32_to_f16_rne(float value) noexcept {
    return float8_detail::float16_from_bits(
        float8_detail::f32_bits_to_f16_rne_bits(float8_detail::float_bits(value)));
}

// Normative f16 -> f8 stage.  h is the numeric f16 bit pattern, not a memory
// byte, so this is independent of host endianness.
inline uint8_t f16_bits_to_float8_rne(uint16_t h) noexcept {
    const uint32_t r = (static_cast<uint32_t>(h) + 0x7fU + ((h >> 8) & 1U)) >> 8;
    return static_cast<uint8_t>(r);
}

inline uint8_t encode_float8_unchecked_bits(float value) noexcept {
    const float16 f16 = f32_to_f16_rne(value);
    return f16_bits_to_float8_rne(float8_detail::float16_bits(f16));
}

class float8 {
public:
    // This remains trivial.  Value initialization (float8{}) zero-initializes
    // bits_, while default initialization intentionally leaves it unspecified.
    float8() = default;

    explicit float8(float value) noexcept : bits_(encode_float8_unchecked_bits(value)) {}

    static constexpr float8 from_bits(uint8_t bits) noexcept {
        return float8(bits, from_bits_tag {});
    }

    constexpr uint8_t to_bits() const noexcept {
        return bits_;
    }

    // The sole implicit conversion operator: decode through the exact f16-bit
    // alias.  NaN payloads are intentionally not promised by this interface.
    operator float() const noexcept {
        const float16 f16 = float8_detail::float16_from_bits(
            static_cast<uint16_t>(bits_) << 8);
        return static_cast<float>(f16);
    }

private:
    struct from_bits_tag {};

    constexpr float8(uint8_t bits, from_bits_tag) noexcept : bits_(bits) {}

    uint8_t bits_;
};

static_assert(sizeof(float8) == 1);
static_assert(alignof(float8) == 1);
static_assert(std::is_trivially_copyable_v<float8>);
static_assert(std::is_standard_layout_v<float8>);
static_assert(std::is_trivially_default_constructible_v<float8>);

// The canonical bounded synthetic-data codebook.  Entries are constructed from
// E5M2 fields, not decimal floats: negative values are emitted from largest
// magnitude to smallest so the complete sequence is numerically sorted, then
// the positive values are emitted from smallest to largest.  Its layout is:
//
//   sign | exponent fields 19..11 | mantissas 3..0   (-28 .. -0.0625)
//   sign | exponent fields 11..19 | mantissas 0..3   (+0.0625 .. +28)
//
// Keeping this one byte-level definition here lets storage generators and
// later compute tests share an auditable, grid-exact source of f8 values.
namespace float8_codebook {

inline constexpr size_t kSize = 72;

namespace detail {

constexpr std::array<uint8_t, kSize> make_bits() noexcept {
    std::array<uint8_t, kSize> bits {};
    size_t index = 0;

    for (int exponent = 19; exponent >= 11; --exponent) {
        for (int mantissa = 3; mantissa >= 0; --mantissa) {
            bits[index++] = static_cast<uint8_t>(
                0x80U | (static_cast<uint8_t>(exponent) << 2) |
                static_cast<uint8_t>(mantissa));
        }
    }
    for (int exponent = 11; exponent <= 19; ++exponent) {
        for (int mantissa = 0; mantissa <= 3; ++mantissa) {
            bits[index++] = static_cast<uint8_t>(
                (static_cast<uint8_t>(exponent) << 2) |
                static_cast<uint8_t>(mantissa));
        }
    }

    return bits;
}

} // namespace detail

inline constexpr std::array<uint8_t, kSize> kBits = detail::make_bits();

static_assert(kBits[0] == 0xcfU);
static_assert(kBits[35] == 0xacU);
static_assert(kBits[36] == 0x2cU);
static_assert(kBits[kSize - 1] == 0x4fU);

inline constexpr const std::array<uint8_t, kSize>& bits() noexcept {
    return kBits;
}

inline constexpr uint8_t bit_at(size_t index) noexcept {
    return kBits[index];
}

inline constexpr float8 value_at(size_t index) noexcept {
    return float8::from_bits(bit_at(index));
}

// Returns the exact 72^dim capacity only while it fits in uint64_t.  Callers
// that only need validation should use range_fits(), which is overflow-safe
// even when the mathematical capacity exceeds uint64_t.
inline bool capacity(size_t dim, uint64_t* output) noexcept {
    if (output == nullptr) {
        return false;
    }

    uint64_t result = 1;
    for (size_t d = 0; d < dim; ++d) {
        if (result > std::numeric_limits<uint64_t>::max() / kSize) {
            *output = 0;
            return false;
        }
        result *= kSize;
    }

    *output = result;
    return true;
}

// Tests whether ordinal can be represented by dim base-72 digits without
// multiplying a potentially overflowing 72^dim value.
inline bool ordinal_fits(size_t dim, uint64_t ordinal) noexcept {
    while (dim != 0 && ordinal != 0) {
        ordinal /= kSize;
        --dim;
    }
    return ordinal == 0;
}

// item_count is range-relative: the generated ordinals are [0, item_count).
// Empty ranges fit by definition.  This never wraps/cycles when 72^dim is
// larger than uint64_t.
inline bool range_fits(size_t dim, uint64_t item_count) noexcept {
    return item_count == 0 || ordinal_fits(dim, item_count - 1);
}

// Maps an ordinal to little-endian base-72 digits: dimension zero contains
// the least-significant digit.  Failure leaves the output untouched.
inline bool fill_ordinal_vector(uint64_t ordinal, float8* output, size_t dim) noexcept {
    if ((dim != 0 && output == nullptr) || !ordinal_fits(dim, ordinal)) {
        return false;
    }

    uint64_t remaining = ordinal;
    for (size_t d = 0; d < dim; ++d) {
        output[d] = value_at(static_cast<size_t>(remaining % kSize));
        remaining /= kSize;
    }
    return true;
}

// Deterministic selection used by seeded perf/kernel inputs.  State mixing is
// owned by the caller; this helper keeps the final codebook selection shared.
inline constexpr size_t seeded_index(uint64_t state) noexcept {
    return static_cast<size_t>(state % kSize);
}

inline constexpr float8 seeded_value(uint64_t state) noexcept {
    return value_at(seeded_index(state));
}

// InputVector<float8> treats max_val as an inclusive numeric upper bound on a
// prefix of this sorted codebook.  Values below -28 (and NaN) clamp to index
// zero; values above +28 clamp to the final index.
inline size_t upper_bound_index(float max_val) noexcept {
    const float first = static_cast<float>(value_at(0));
    const float last = static_cast<float>(value_at(kSize - 1));
    if (std::isnan(max_val) || max_val <= first) {
        return 0;
    }
    if (max_val >= last) {
        return kSize - 1;
    }

    size_t index = 0;
    for (size_t candidate = 1; candidate < kSize; ++candidate) {
        if (static_cast<float>(value_at(candidate)) > max_val) {
            break;
        }
        index = candidate;
    }
    return index;
}

} // namespace float8_codebook

// Checked numeric ingest policy.  from_bits and the explicit constructor are
// deliberately unchecked so trusted raw bytes can represent every E5M2 class.
inline bool try_encode_float8(float value, float8& output) noexcept {
    if (!std::isfinite(value)) {
        return false;
    }

    const float8 encoded(value);
    if ((encoded.to_bits() & 0x7fU) >= 0x7cU) {
        return false;
    }

    output = encoded;
    return true;
}

} // namespace sketch2
