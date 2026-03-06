#pragma once

#include "utils/shared_types.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/file.h>
#include <unistd.h>

namespace sketch2 {

class FileLockGuard {
public:
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

private:
    int fd_ = -1;
};

} // namespace sketch2
