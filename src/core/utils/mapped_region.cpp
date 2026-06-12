// Implements MappedRegion.

#include "utils/mapped_region.h"

#include <cerrno>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace sketch2 {

MappedRegion::MappedRegion(MappedRegion&& other) noexcept
    : data_(other.data_), size_(other.size_), access_(other.access_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.access_ = MappedRegionAccess::ReadOnly;
}

MappedRegion& MappedRegion::operator=(MappedRegion&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    reset();
    data_ = other.data_;
    size_ = other.size_;
    access_ = other.access_;
    other.data_ = nullptr;
    other.size_ = 0;
    other.access_ = MappedRegionAccess::ReadOnly;
    return *this;
}

MappedRegion::~MappedRegion() {
    reset();
}

Ret MappedRegion::init(
        int fd,
        size_t offset,
        size_t size,
        MappedRegionAccess access,
        const std::string& context) {
    const auto error = [&context](const std::string& message) {
        if (context.empty()) {
            return Ret("MappedRegion::init: " + message);
        }
        return Ret(context + ": " + message);
    };

    if (data_ != nullptr) {
        return error("region is initialized already");
    }
    if (fd < 0) {
        return error("invalid file descriptor");
    }
    if (size == 0) {
        return error("size must be greater than zero");
    }
    const long page_size_long = sysconf(_SC_PAGESIZE);
    if (page_size_long <= 0) {
        return error("failed to determine system page size");
    }
    const size_t page_size = static_cast<size_t>(page_size_long);
    if ((offset % page_size) != 0) {
        return error("offset " + std::to_string(offset)
            + " is not aligned to system page size " + std::to_string(page_size));
    }

    struct stat st {};
    if (fstat(fd, &st) != 0) {
        return error("failed to stat file: " + std::string(std::strerror(errno)));
    }
    if (st.st_size < 0) {
        return error("invalid file size");
    }
    const size_t file_size = static_cast<size_t>(st.st_size);
    if (offset > file_size || size > file_size - offset) {
        return error("mapping range exceeds file size");
    }

    int prot = PROT_READ;
    int flags = MAP_PRIVATE;
    switch (access) {
        case MappedRegionAccess::ReadOnly:
            prot = PROT_READ;
            flags = MAP_PRIVATE;
            break;
        case MappedRegionAccess::Writable:
            prot = PROT_READ | PROT_WRITE;
            flags = MAP_SHARED;
            break;
    }

    void* region = mmap(nullptr, size, prot, flags, fd, static_cast<off_t>(offset));
    if (region == MAP_FAILED) {
        return error("failed to mmap region: " + std::string(std::strerror(errno)));
    }

    data_ = static_cast<uint8_t*>(region);
    size_ = size;
    access_ = access;
    return Ret(0);
}

Ret MappedRegion::sync(const std::string& error_message) const {
    if (data_ == nullptr || size_ == 0) {
        return Ret(0);
    }
    if (msync(data_, size_, MS_SYNC) != 0) {
        return Ret(error_message);
    }
    return Ret(0);
}

void MappedRegion::reset() {
    if (data_ != nullptr) {
        munmap(data_, size_);
        data_ = nullptr;
        size_ = 0;
        access_ = MappedRegionAccess::ReadOnly;
    }
}

} // namespace sketch2
