// Declares filesystem lock helpers used to coordinate dataset ownership.

#pragma once

#include "utils/shared_types.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/file.h>
#include <unistd.h>

namespace sketch2 {

// FileLockGuard exists to give dataset code a simple RAII wrapper around
// process-level file locks. It opens a lock file, acquires an exclusive flock,
// and guarantees unlock/close on destruction so ownership paths stay exception-safe.
class FileLockGuard {
public:
    FileLockGuard() = default;
    FileLockGuard(const FileLockGuard&) = delete;
    FileLockGuard& operator=(const FileLockGuard&) = delete;
    FileLockGuard(FileLockGuard&&) = delete;
    FileLockGuard& operator=(FileLockGuard&&) = delete;

    ~FileLockGuard() {
        if (fd_ >= 0) {
            (void)flock(fd_, LOCK_UN);
            (void)close(fd_);
        }
    }

    Ret lock(const std::string& path) {
        fd_ = open(path.c_str(), O_RDWR | O_CREAT, 0666);
        if (fd_ < 0) {
            return Ret("Dataset: failed to open lock file " + path + ": " + std::strerror(errno));
        }

        while (flock(fd_, LOCK_EX) != 0) {
            if (errno == EINTR) {
                continue;
            }
            (void)close(fd_);
            fd_ = -1;
            return Ret("Dataset: failed to lock file " + path + ": " + std::strerror(errno));
        }

        return Ret(0);
    }

    // Returns true and holds the lock if it was immediately available,
    // false if another process already holds it (non-blocking).
    bool try_lock(const std::string& path) {
        fd_ = open(path.c_str(), O_RDWR | O_CREAT, 0666);
        if (fd_ < 0) {
            return false;
        }

        if (flock(fd_, LOCK_EX | LOCK_NB) != 0) {
            (void)close(fd_);
            fd_ = -1;
            return false;
        }

        return true;
    }

private:
    int fd_ = -1;
};

} // namespace sketch2
