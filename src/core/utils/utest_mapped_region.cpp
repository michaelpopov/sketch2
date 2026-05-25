// Unit tests for MappedRegion.

#include "utils/mapped_region.h"

#include <gtest/gtest.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>

#include "utest_tmp_dir.h"

using namespace sketch2;

namespace {

class MappedRegionTest : public ::testing::Test {
protected:
    void SetUp() override {
        path_ = tmp_dir() + "/sketch2_utest_mapped_region_" + std::to_string(getpid()) + ".bin";
    }

    void TearDown() override {
        std::remove(path_.c_str());
    }

    int create_file(size_t size) {
        const int fd = open(path_.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
        EXPECT_GE(fd, 0);
        if (fd < 0) {
            return fd;
        }
        const int truncate_ret = ftruncate(fd, static_cast<off_t>(size));
        EXPECT_EQ(0, truncate_ret);
        if (truncate_ret != 0) {
            close(fd);
            return -1;
        }

        std::string bytes(size, '\0');
        for (size_t i = 0; i < size; ++i) {
            bytes[i] = static_cast<char>(i % 251);
        }
        const ssize_t written = pwrite(fd, bytes.data(), size, 0);
        EXPECT_EQ(static_cast<ssize_t>(size), written);
        if (written != static_cast<ssize_t>(size)) {
            close(fd);
            return -1;
        }

        return fd;
    }

    std::string path_;
};

} // namespace

TEST_F(MappedRegionTest, DefaultConstructedRegionIsEmpty) {
    MappedRegion region;
    EXPECT_EQ(nullptr, region.data());
    EXPECT_EQ(0u, region.size());
    EXPECT_TRUE(region.empty());
}

TEST_F(MappedRegionTest, InitMapsRequestedRegion) {
    const int fd = create_file(4096);
    ASSERT_GE(fd, 0);

    MappedRegion region;
    ASSERT_EQ(0, region.init(fd, 0, 4096).code());
    ASSERT_NE(nullptr, region.data());
    EXPECT_EQ(4096u, region.size());
    EXPECT_FALSE(region.empty());
    EXPECT_EQ(0u, region.data()[0]);
    EXPECT_EQ(1u, region.data()[1]);
    EXPECT_EQ(2u, region.data()[2]);
    EXPECT_EQ(3u, region.data()[3]);

    close(fd);
}

TEST_F(MappedRegionTest, ResetClearsOwnedRegion) {
    const int fd = create_file(4096);
    ASSERT_GE(fd, 0);

    MappedRegion region;
    ASSERT_EQ(0, region.init(fd, 0, 4096).code());
    close(fd);

    region.reset();
    EXPECT_EQ(nullptr, region.data());
    EXPECT_EQ(0u, region.size());
    EXPECT_TRUE(region.empty());
}

TEST_F(MappedRegionTest, MoveTransfersOwnership) {
    const int fd = create_file(4096);
    ASSERT_GE(fd, 0);

    MappedRegion src;
    ASSERT_EQ(0, src.init(fd, 0, 4096).code());
    close(fd);
    const uint8_t* original = src.data();

    MappedRegion dst(std::move(src));
    EXPECT_EQ(nullptr, src.data());
    EXPECT_EQ(0u, src.size());
    EXPECT_EQ(original, dst.data());
    EXPECT_EQ(4096u, dst.size());
}

TEST_F(MappedRegionTest, MoveAssignmentReplacesOwnedRegion) {
    const int fd0 = create_file(4096);
    ASSERT_GE(fd0, 0);

    MappedRegion src;
    ASSERT_EQ(0, src.init(fd0, 0, 4096).code());
    close(fd0);
    const uint8_t* original = src.data();

    const int fd1 = create_file(8192);
    ASSERT_GE(fd1, 0);
    MappedRegion dst;
    ASSERT_EQ(0, dst.init(fd1, 0, 8192).code());
    close(fd1);

    dst = std::move(src);
    EXPECT_EQ(nullptr, src.data());
    EXPECT_EQ(0u, src.size());
    EXPECT_EQ(original, dst.data());
    EXPECT_EQ(4096u, dst.size());
}

TEST_F(MappedRegionTest, WritableMappingAllowsModification) {
    const int fd = create_file(4096);
    ASSERT_GE(fd, 0);

    MappedRegion region;
    ASSERT_EQ(0, region.init(fd, 0, 4096, MappedRegionAccess::Writable).code());
    uint8_t* writable = region.mutable_data();
    ASSERT_NE(nullptr, writable);
    writable[0] = 42;
    writable[1] = 77;
    ASSERT_EQ(0, region.sync("sync failed").code());
    close(fd);

    const int read_fd = open(path_.c_str(), O_RDONLY);
    ASSERT_GE(read_fd, 0);
    uint8_t bytes[2] = {};
    ASSERT_EQ(2, pread(read_fd, bytes, sizeof(bytes), 0));
    close(read_fd);

    EXPECT_EQ(42u, bytes[0]);
    EXPECT_EQ(77u, bytes[1]);
}
