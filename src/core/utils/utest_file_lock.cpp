// Unit tests for filesystem locking helpers.

#include "utils/file_lock.h"

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

using namespace sketch2;

namespace fs = std::filesystem;

class FileLockGuardTest : public ::testing::Test {
protected:
    std::string lock_path_;

    void SetUp() override {
        lock_path_ = "/tmp/sketch2_utest_lock_" + std::to_string(getpid()) + ".lock";
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
            (void)write(pipe_fds[1], &status, 1);
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
