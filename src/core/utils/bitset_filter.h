// Declares the shared bitset-based allowlist filter used by scanner APIs.

#pragma once

#include <cstdint>

namespace sketch2 {

class ChunkedBits;

struct BitsetFilter {
    const uint64_t base_id;
    const uint8_t* data;
    const uint64_t size;
    const ChunkedBits* chunked_bits = nullptr;
};

} // namespace sketch2
