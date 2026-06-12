#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace sketch2 {

inline bool is_power_of_two_alignment(size_t alignment) {
    return alignment != 0 && (alignment & (alignment - 1u)) == 0;
}

inline bool is_aligned(const void* ptr, size_t alignment) {
    const bool valid_alignment = is_power_of_two_alignment(alignment);
    assert(valid_alignment);
    if (!valid_alignment) {
        return false;
    }
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

inline bool align_up(size_t value, size_t alignment, size_t* out) {
    const bool valid_alignment = is_power_of_two_alignment(alignment);
    assert(valid_alignment);
    if (!valid_alignment) {
        return false;
    }
    const size_t mask = alignment - 1u;
    if (value > std::numeric_limits<size_t>::max() - mask) {
        return false;
    }
    *out = (value + mask) & ~mask;
    return true;
}

inline bool add_overflows(size_t lhs, size_t rhs, size_t* out) {
    if (lhs > std::numeric_limits<size_t>::max() - rhs) {
        return true;
    }
    *out = lhs + rhs;
    return false;
}

inline bool multiply_overflows(size_t lhs, size_t rhs, size_t* out) {
    if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs) {
        return true;
    }
    *out = lhs * rhs;
    return false;
}

} // namespace sketch2
