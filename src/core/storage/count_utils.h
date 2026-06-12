// Shared checked conversions for on-disk uint32 count fields.

#pragma once

#include "core/utils/shared_types.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace sketch2 {

inline Ret checked_size_to_uint32(size_t value, uint32_t* out, const char* error_message) {
    if (out == nullptr) {
        return Ret("checked_size_to_uint32: missing output");
    }
    if (value > std::numeric_limits<uint32_t>::max()) {
        return Ret(error_message);
    }
    *out = static_cast<uint32_t>(value);
    return Ret(0);
}

} // namespace sketch2
