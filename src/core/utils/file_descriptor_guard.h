// RAII wrapper that closes an owned file descriptor on destruction.

#pragma once

#include <unistd.h>

namespace sketch2 {

class FileDescriptorGuard {
public:
    FileDescriptorGuard() = default;
    explicit FileDescriptorGuard(int fd) noexcept
        : fd_(fd) {}

    FileDescriptorGuard(const FileDescriptorGuard&) = delete;
    FileDescriptorGuard& operator=(const FileDescriptorGuard&) = delete;

    FileDescriptorGuard(FileDescriptorGuard&& other) noexcept
        : fd_(other.fd_) {
        other.fd_ = -1;
    }

    FileDescriptorGuard& operator=(FileDescriptorGuard&& other) noexcept {
        if (this != &other) {
            close_fd();
            fd_ = other.fd_;
            other.fd_ = -1;
        }
        return *this;
    }

    ~FileDescriptorGuard() {
        close_fd();
    }

    void reset(int fd) noexcept {
        close_fd();
        fd_ = fd;
    }

    void close_fd() noexcept {
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
    }

    int release_fd() noexcept {
        const int fd = fd_;
        fd_ = -1;
        return fd;
    }

    int fd() const noexcept {
        return fd_;
    }

private:
    int fd_ = -1;
};

} // namespace sketch2
