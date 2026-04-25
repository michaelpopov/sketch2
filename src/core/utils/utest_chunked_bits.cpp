// Unit tests for chunked sparse allowlists.

#include "chunked_bits.h"

#include <gtest/gtest.h>

namespace sketch2 {

TEST(chunked_bits, rejects_too_many_chunks) {
    ChunkedBits bits;

    for (size_t i = 0; i < kChunkedBitsMaxChunks; ++i) {
        ASSERT_EQ(0, bits.add(static_cast<uint64_t>(i) << kChunkBits).code());
    }

    const Ret ret = bits.add(static_cast<uint64_t>(kChunkedBitsMaxChunks) << kChunkBits);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("ChunkedBits::add: too many chunks", ret.message());
}

} // namespace sketch2
