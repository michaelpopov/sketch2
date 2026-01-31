#include <gtest/gtest.h>

TEST(StorageDummy, Basic) { EXPECT_TRUE(true); }

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
