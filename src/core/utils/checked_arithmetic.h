#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace sketch2 {

inline bool is_aligned(const void* ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

inline bool align_up(size_t value, size_t alignment, size_t* out) {
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
