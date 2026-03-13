// Test runner entry point for the compute module.

#include <gtest/gtest.h>

TEST(ComputeDummy, Basic) { EXPECT_TRUE(true); }

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
