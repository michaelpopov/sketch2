// Unit tests for checked arithmetic helpers.

#include "utils/checked_arithmetic.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace sketch2 {

TEST(checked_arithmetic, align_up_accepts_power_of_two_alignment) {
    size_t out = 0;

    ASSERT_TRUE(align_up(33, 32, &out));

    EXPECT_EQ(64u, out);
}

TEST(checked_arithmetic, align_up_reports_overflow) {
    size_t out = 123;

    EXPECT_FALSE(align_up(std::numeric_limits<size_t>::max(), 32, &out));
    EXPECT_EQ(123u, out);
}

TEST(checked_arithmetic, is_aligned_accepts_power_of_two_alignment) {
    alignas(32) uint8_t bytes[64] = {};

    EXPECT_TRUE(is_aligned(bytes, 32));
    EXPECT_FALSE(is_aligned(bytes + 1, 32));
}

#ifdef NDEBUG

TEST(checked_arithmetic, align_upRejectsInvalidAlignmentInRelease) {
    size_t out = 123;

    EXPECT_FALSE(align_up(33, 0, &out));
    EXPECT_EQ(123u, out);
    EXPECT_FALSE(align_up(33, 24, &out));
    EXPECT_EQ(123u, out);
}

TEST(checked_arithmetic, is_alignedRejectsInvalidAlignmentInRelease) {
    alignas(32) uint8_t bytes[64] = {};

    EXPECT_FALSE(is_aligned(bytes, 0));
    EXPECT_FALSE(is_aligned(bytes, 24));
}

#else

TEST(checked_arithmetic, align_upAssertsOnInvalidAlignment) {
    size_t out = 0;

    EXPECT_DEATH((void)align_up(33, 0, &out), "valid_alignment");
    EXPECT_DEATH((void)align_up(33, 24, &out), "valid_alignment");
}

TEST(checked_arithmetic, is_alignedAssertsOnInvalidAlignment) {
    alignas(32) uint8_t bytes[64] = {};

    EXPECT_DEATH((void)is_aligned(bytes, 0), "valid_alignment");
    EXPECT_DEATH((void)is_aligned(bytes, 24), "valid_alignment");
}

#endif

} // namespace sketch2
