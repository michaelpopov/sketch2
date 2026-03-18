// Unit tests for UpdateNotifier cross-process cache invalidation.

#include "utils/update_notifier.h"

#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <filesystem>
#include <string>
#include <unistd.h>

#include <gtest/gtest.h>
#include "utest_tmp_dir.h"

using namespace sketch2;

namespace fs = std::filesystem;

class UpdateNotifierTest : public ::testing::Test {
protected:
    std::string file_path_;

    void SetUp() override {
        file_path_ = tmp_dir() + "/sketch2_utest_notifier_" + std::to_string(getpid()) + ".lock";
        std::remove(file_path_.c_str());
    }

    void TearDown() override {
        std::remove(file_path_.c_str());
    }

    uint64_t read_counter_from_file() {
        int fd = open(file_path_.c_str(), O_RDONLY);
        if (fd < 0) return UINT64_MAX;
        uint64_t value = 0;
        ssize_t n = pread(fd, &value, sizeof(value), 0);
        close(fd);
        return (n == static_cast<ssize_t>(sizeof(value))) ? value : UINT64_MAX;
    }
};

TEST_F(UpdateNotifierTest, InitUpdaterCreatesFileWithZeroCounter) {
    UpdateNotifier notifier;
    Ret ret = notifier.init_updater(file_path_);
    ASSERT_EQ(0, ret.code()) << ret.message();
    EXPECT_TRUE(fs::exists(file_path_));
    EXPECT_EQ(0u, read_counter_from_file());
}

TEST_F(UpdateNotifierTest, UpdateIncrementsCounter) {
    UpdateNotifier notifier;
    ASSERT_EQ(0, notifier.init_updater(file_path_).code());

    ASSERT_EQ(0, notifier.update().code());
    EXPECT_EQ(1u, read_counter_from_file());

    ASSERT_EQ(0, notifier.update().code());
    EXPECT_EQ(2u, read_counter_from_file());
}

TEST_F(UpdateNotifierTest, UpdateFailsWithoutInit) {
    UpdateNotifier notifier;
    Ret ret = notifier.update();
    EXPECT_NE(0, ret.code());
}

TEST_F(UpdateNotifierTest, CheckerReturnsTrueOnFirstCall) {
    // Create the file with a known counter.
    {
        UpdateNotifier updater;
        ASSERT_EQ(0, updater.init_updater(file_path_).code());
    }

    UpdateNotifier checker;
    ASSERT_EQ(0, checker.init_checker(file_path_).code());
    EXPECT_TRUE(checker.check_updated());
}

TEST_F(UpdateNotifierTest, CheckerReturnsFalseWhenUnchanged) {
    {
        UpdateNotifier updater;
        ASSERT_EQ(0, updater.init_updater(file_path_).code());
    }

    UpdateNotifier checker;
    ASSERT_EQ(0, checker.init_checker(file_path_).code());
    EXPECT_TRUE(checker.check_updated());   // first call
    EXPECT_FALSE(checker.check_updated());  // no change
    EXPECT_FALSE(checker.check_updated());  // still no change
}

TEST_F(UpdateNotifierTest, CheckerDetectsUpdate) {
    UpdateNotifier updater;
    ASSERT_EQ(0, updater.init_updater(file_path_).code());

    UpdateNotifier checker;
    ASSERT_EQ(0, checker.init_checker(file_path_).code());
    EXPECT_TRUE(checker.check_updated());   // first call
    EXPECT_FALSE(checker.check_updated());  // no change yet

    ASSERT_EQ(0, updater.update().code());
    EXPECT_TRUE(checker.check_updated());   // detects the update
    EXPECT_FALSE(checker.check_updated());  // no further change
}

TEST_F(UpdateNotifierTest, CheckerDetectsMultipleUpdates) {
    UpdateNotifier updater;
    ASSERT_EQ(0, updater.init_updater(file_path_).code());

    UpdateNotifier checker;
    ASSERT_EQ(0, checker.init_checker(file_path_).code());
    EXPECT_TRUE(checker.check_updated());

    for (int i = 0; i < 5; ++i) {
        ASSERT_EQ(0, updater.update().code());
        EXPECT_TRUE(checker.check_updated());
        EXPECT_FALSE(checker.check_updated());
    }
}

TEST_F(UpdateNotifierTest, CheckerReturnsTrueWhenFileMissing) {
    UpdateNotifier checker;
    ASSERT_EQ(0, checker.init_checker(file_path_).code());
    // File does not exist — conservative true.
    EXPECT_TRUE(checker.check_updated());
    // File still does not exist — fd was not opened, so still true.
    EXPECT_TRUE(checker.check_updated());
}

TEST_F(UpdateNotifierTest, UpdaterReadsExistingCounter) {
    // First updater writes counter = 3.
    {
        UpdateNotifier updater;
        ASSERT_EQ(0, updater.init_updater(file_path_).code());
        ASSERT_EQ(0, updater.update().code()); // 1
        ASSERT_EQ(0, updater.update().code()); // 2
        ASSERT_EQ(0, updater.update().code()); // 3
    }

    EXPECT_EQ(3u, read_counter_from_file());

    // Second updater picks up where the first left off.
    {
        UpdateNotifier updater;
        ASSERT_EQ(0, updater.init_updater(file_path_).code());
        ASSERT_EQ(0, updater.update().code()); // 4
    }

    EXPECT_EQ(4u, read_counter_from_file());
}

TEST_F(UpdateNotifierTest, InitUpdaterFailsOnBadPath) {
    const std::string bad_path = tmp_dir() + "/no_such_dir_" + std::to_string(getpid()) + "/notifier.lock";
    UpdateNotifier notifier;
    Ret ret = notifier.init_updater(bad_path);
    EXPECT_NE(0, ret.code());
}
