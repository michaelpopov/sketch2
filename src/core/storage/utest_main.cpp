// Test runner entry point for the storage module.

#include <gtest/gtest.h>
#include "utils/shared_types.h"
#include "core/storage/data_file.h"

using namespace sketch2;

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(STORAGE, HeaderSizeCheck) {
    EXPECT_EQ(2, sizeof(float16));
    EXPECT_EQ(0, sizeof(sketch2::BaseFileHeader) % 8);
    EXPECT_EQ(0, sizeof(sketch2::DataFileHeader) % 8);
    EXPECT_EQ(0, sizeof(sketch2::WalFileHeader) % 8);
    EXPECT_EQ(0, sizeof(sketch2::WalRecordHeader) % 8);
}
