// Implements UpdateNotifier: a file-backed uint64 counter for cross-process
// cache invalidation.

#include "utils/update_notifier.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

namespace sketch2 {

UpdateNotifier::~UpdateNotifier() {
    if (fd_ >= 0) {
        (void)close(fd_);
    }
}

Ret UpdateNotifier::init_updater(const std::string& path) {
    path_ = path;
    is_updater_ = true;

    fd_ = open(path.c_str(), O_RDWR | O_CREAT, 0666);
    if (fd_ < 0) {
        return Ret("UpdateNotifier: failed to open file " + path + ": " + std::strerror(errno));
    }

    // Try to read an existing counter.  A short or failed read means the file
    // is new or empty — counter stays at its default (0) and we write it out so
    // the file always contains a valid 8-byte value.
    uint64_t value = 0;
    const ssize_t n = pread(fd_, &value, sizeof(value), 0);
    if (n == static_cast<ssize_t>(sizeof(value))) {
        counter_ = value;
    } else {
        counter_ = 0;
        if (pwrite(fd_, &counter_, sizeof(counter_), 0) != static_cast<ssize_t>(sizeof(counter_))) {
            return Ret("UpdateNotifier: failed to write initial counter: " + std::string(std::strerror(errno)));
        }
        if (fdatasync(fd_) != 0) {
            return Ret("UpdateNotifier: fdatasync failed: " + std::string(std::strerror(errno)));
        }
    }

    return Ret(0);
}

Ret UpdateNotifier::update() {
    if (!is_updater_ || fd_ < 0) {
        return Ret("UpdateNotifier: update() called without init_updater()");
    }

    ++counter_;

    if (pwrite(fd_, &counter_, sizeof(counter_), 0) != static_cast<ssize_t>(sizeof(counter_))) {
        return Ret("UpdateNotifier: failed to write counter: " + std::string(std::strerror(errno)));
    }

    if (fdatasync(fd_) != 0) {
        return Ret("UpdateNotifier: fdatasync failed: " + std::string(std::strerror(errno)));
    }

    return Ret(0);
}

Ret UpdateNotifier::init_checker(const std::string& path) {
    path_ = path;
    is_updater_ = false;
    return Ret(0);
}

bool UpdateNotifier::check_updated() {
    // First call (or file not opened yet): open and read.
    if (fd_ < 0) {
        fd_ = open(path_.c_str(), O_RDONLY);
        if (fd_ < 0) {
            return true; // file missing or error — conservative
        }

        uint64_t value = 0;
        if (pread(fd_, &value, sizeof(value), 0) != static_cast<ssize_t>(sizeof(value))) {
            (void)close(fd_);
            fd_ = -1;
            return true; // short read — conservative
        }

        counter_ = value;
        return true; // first observation is always treated as "updated"
    }

    // Subsequent calls: re-read and compare.
    uint64_t value = 0;
    if (pread(fd_, &value, sizeof(value), 0) != static_cast<ssize_t>(sizeof(value))) {
        return true; // read error — conservative
    }

    if (value == counter_) {
        return false;
    }

    counter_ = value;
    return true;
}

} // namespace sketch2
