// Declares the shared bitset-based allowlist filter used by scanner APIs.

#pragma once

#include <cstdint>

namespace sketch2 {

struct BitsetFilter {
    const uint8_t* data;
    const uint64_t size;
};

} // namespace sketch2
