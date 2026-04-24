// Implements MappedRegion.

#include "utils/mapped_region.h"

#include <sys/mman.h>

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
        bool is_seq,
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
        return error("failed to mmap region");
    }
    if (is_seq && madvise(region, size, MADV_SEQUENTIAL) != 0) {
        munmap(region, size);
        return error("failed to madvise region");
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
