// Unit tests for filesystem locking helpers.

#include "utils/file_lock.h"
#include "utils/file_path_lock.h"

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>
#include "utest_tmp_dir.h"

using namespace sketch2;

namespace fs = std::filesystem;

class FileLockGuardTest : public ::testing::Test {
protected:
    std::string lock_path_;

    void SetUp() override {
        lock_path_ = tmp_dir() + "/sketch2_utest_lock_" + std::to_string(getpid()) + ".lock";
        std::remove(lock_path_.c_str());
    }

    void TearDown() override {
        std::remove(lock_path_.c_str());
    }
};

TEST_F(FileLockGuardTest, LockCreatesFile) {
    FileLockGuard guard;
    const Ret ret = guard.lock(lock_path_);
    ASSERT_EQ(0, ret.code()) << ret.message();
    EXPECT_TRUE(fs::exists(lock_path_));
}

TEST_F(FileLockGuardTest, LockIsExclusiveAcrossProcesses) {
    int pipe_fds[2];
    ASSERT_EQ(0, pipe(pipe_fds));
    ASSERT_GE(fcntl(pipe_fds[0], F_SETFL, O_NONBLOCK), 0);

    pid_t child_pid = -1;
    {
        FileLockGuard guard;
        const Ret ret = guard.lock(lock_path_);
        ASSERT_EQ(0, ret.code()) << ret.message();

        child_pid = fork();
        ASSERT_GE(child_pid, 0);
        if (child_pid == 0) {
            close(pipe_fds[0]);
            FileLockGuard child_guard;
            const Ret child_ret = child_guard.lock(lock_path_);
            const char status = (child_ret.code() == 0) ? '1' : '0';
            const auto sz = write(pipe_fds[1], &status, 1);
            (void)sz;
            close(pipe_fds[1]);
            _exit(child_ret.code() == 0 ? 0 : 1);
        }

        close(pipe_fds[1]);

        char status = '\0';
        errno = 0;
        const ssize_t n = read(pipe_fds[0], &status, 1);
        EXPECT_EQ(-1, n);
        EXPECT_TRUE(errno == EAGAIN || errno == EWOULDBLOCK);
    }

    char status = '\0';
    ssize_t n = -1;
    for (int i = 0; i < 100; ++i) {
        n = read(pipe_fds[0], &status, 1);
        if (n == 1) {
            break;
        }
        usleep(10 * 1000);
    }
    close(pipe_fds[0]);

    ASSERT_EQ(1, n);
    EXPECT_EQ('1', status);

    int child_status = 0;
    ASSERT_EQ(child_pid, waitpid(child_pid, &child_status, 0));
    EXPECT_TRUE(WIFEXITED(child_status));
    EXPECT_EQ(0, WEXITSTATUS(child_status));
}

TEST_F(FileLockGuardTest, LockFailsWhenParentDirectoryDoesNotExist) {
    const std::string nonexistent_path = tmp_dir() + "/sketch2_no_such_dir_" + std::to_string(getpid()) + "/lock.lock";
    FileLockGuard guard;
    const Ret ret = guard.lock(nonexistent_path);
    EXPECT_NE(0, ret.code());
}

TEST_F(FileLockGuardTest, DestructorReleasesLockAllowingReacquisition) {
    {
        FileLockGuard guard;
        ASSERT_EQ(0, guard.lock(lock_path_).code());
    }  // guard destroyed here; lock must be released

    FileLockGuard guard2;
    const Ret ret = guard2.lock(lock_path_);
    EXPECT_EQ(0, ret.code()) << ret.message();
}

TEST(FilePathLockTest, RejectsNonexistentFile) {
    FilePathLock lock;
    const std::string missing_path = tmp_dir() + "/sketch2_utest_missing_file_" + std::to_string(getpid());
    EXPECT_FALSE(lock.check_file_path(missing_path));
}

TEST(FilePathLockTest, AllowsOnlyFirstCheckPerPath) {
    const std::string file_path = tmp_dir() + "/sketch2_utest_file_" + std::to_string(getpid());
    std::ofstream file(file_path);
    ASSERT_TRUE(file.is_open());
    file << "test";
    file.close();

    FilePathLock lock;
    EXPECT_TRUE(lock.check_file_path(file_path));
    EXPECT_FALSE(lock.check_file_path(file_path));

    std::remove(file_path.c_str());
}

TEST(FilePathLockTest, ReleaseAllowsReuse) {
    const std::string file_path = tmp_dir() + "/sketch2_utest_file_release_" + std::to_string(getpid());
    std::ofstream file(file_path);
    ASSERT_TRUE(file.is_open());
    file << "data";
    file.close();

    FilePathLock lock;
    ASSERT_TRUE(lock.check_file_path(file_path));
    EXPECT_TRUE(lock.release_file_path(file_path));
    EXPECT_FALSE(lock.release_file_path(file_path));
    EXPECT_TRUE(lock.check_file_path(file_path));

    std::remove(file_path.c_str());
}

TEST(FilePathLockTest, ReleaseAfterFileRemoved) {
    const std::string file_path = tmp_dir() + "/sketch2_utest_file_release_missing_" + std::to_string(getpid());
    std::ofstream file(file_path);
    ASSERT_TRUE(file.is_open());
    file << "data";
    file.close();

    FilePathLock lock;
    ASSERT_TRUE(lock.check_file_path(file_path));
    EXPECT_TRUE(fs::exists(file_path));
    std::remove(file_path.c_str());
    EXPECT_FALSE(fs::exists(file_path));
    EXPECT_TRUE(lock.release_file_path(file_path));
    EXPECT_FALSE(lock.check_file_path(file_path));

    std::ofstream recreated(file_path);
    ASSERT_TRUE(recreated.is_open());
    recreated << "data";
    recreated.close();
    EXPECT_TRUE(lock.check_file_path(file_path));

    std::remove(file_path.c_str());
}

TEST(FilePathLockTest, CanonicalizesDuplicatePaths) {
    const fs::path base_dir = fs::path(tmp_dir()) / ("sketch2_utest_dir_" + std::to_string(getpid()));
    const fs::path target_file = base_dir / "data.bin";
    const fs::path link_dir = fs::path(tmp_dir()) / ("sketch2_utest_dir_link_" + std::to_string(getpid()));

    fs::create_directories(base_dir);
    std::ofstream file(target_file);
    ASSERT_TRUE(file.is_open());
    file << "data";
    file.close();

    if (fs::exists(link_dir)) {
        fs::remove(link_dir);
    }
    fs::create_directory_symlink(base_dir, link_dir);

    FilePathLock lock;
    EXPECT_TRUE(lock.check_file_path(target_file.string()));
    EXPECT_FALSE(lock.check_file_path((link_dir / "data.bin").string()));

    fs::remove(link_dir);
    fs::remove(target_file);
    fs::remove(base_dir);
}
