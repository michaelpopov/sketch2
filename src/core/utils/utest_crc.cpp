// Unit tests for checksum helpers.

#include "crc.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>

namespace sketch2 {

TEST(crc, crc32_update_uses_crc32c_known_answer) {
    const char* payload = "123456789";
    const uint32_t crc = crc32_update(
        0, reinterpret_cast<const uint8_t*>(payload), std::strlen(payload));
    EXPECT_EQ(0xE3069283u, crc);
}

TEST(crc, crc32_update_matches_chunked_updates) {
    std::array<uint8_t, 4099> payload {};
    for (size_t i = 0; i < payload.size(); ++i) {
        payload[i] = static_cast<uint8_t>((i * 131u + i / 7u) & 0xFFu);
    }

    const uint32_t whole = crc32_update(0, payload.data(), payload.size());
    uint32_t chunked = 0;
    size_t offset = 0;
    for (size_t chunk : {1u, 3u, 17u, 256u, 1021u, 4096u}) {
        const size_t n = std::min(chunk, payload.size() - offset);
        chunked = crc32_update(chunked, payload.data() + offset, n);
        offset += n;
        if (offset == payload.size()) {
            break;
        }
    }
    if (offset < payload.size()) {
        chunked = crc32_update(chunked, payload.data() + offset, payload.size() - offset);
    }

    EXPECT_EQ(whole, chunked);
}

} // namespace sketch2
