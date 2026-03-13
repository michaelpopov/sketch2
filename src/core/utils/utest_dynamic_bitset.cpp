// Unit tests for the dynamic bitset utility.

#include "dynamic_bitset.h"

#include <gtest/gtest.h>

namespace sketch2 {

TEST(dynamic_bitset, resize_initializes_bits_to_false) {
    DynamicBitset bitset;

    bitset.resize(130);

    EXPECT_EQ(bitset.size(), 130U);
    for (size_t i = 0; i < bitset.size(); ++i) {
        EXPECT_FALSE(bitset.get(i));
    }
}

TEST(dynamic_bitset, set_and_clear_bits) {
    DynamicBitset bitset;
    bitset.resize(130);

    bitset.set(0);
    bitset.set(64);
    bitset.set(129);

    EXPECT_TRUE(bitset.get(0));
    EXPECT_TRUE(bitset.get(64));
    EXPECT_TRUE(bitset.get(129));
    EXPECT_FALSE(bitset.get(1));

    bitset.set(64, false);

    EXPECT_FALSE(bitset.get(64));
}

TEST(dynamic_bitset, resize_preserves_existing_bits_and_clears_new_range) {
    DynamicBitset bitset;
    bitset.resize(10);
    bitset.set(3);
    bitset.set(9);

    bitset.resize(70);

    EXPECT_TRUE(bitset.get(3));
    EXPECT_TRUE(bitset.get(9));
    EXPECT_FALSE(bitset.get(10));
    EXPECT_FALSE(bitset.get(69));
}

TEST(dynamic_bitset, resize_down_discards_truncated_bits) {
    DynamicBitset bitset;
    bitset.resize(70);
    bitset.set(5);
    bitset.set(69);

    bitset.resize(10);
    bitset.resize(70);

    EXPECT_TRUE(bitset.get(5));
    EXPECT_FALSE(bitset.get(69));
}

TEST(dynamic_bitset, out_of_range_access_throws) {
    DynamicBitset bitset;
    bitset.resize(5);

    EXPECT_THROW(bitset.get(5), std::out_of_range);
    EXPECT_THROW(bitset.set(5), std::out_of_range);
}

} // namespace sketch2
