// Implements MappedRegion.

#include "utils/mapped_region.h"

#include <sys/mman.h>

namespace sketch2 {

MappedRegion::MappedRegion(MappedRegion&& other) noexcept
    : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

MappedRegion& MappedRegion::operator=(MappedRegion&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    reset();
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
    return *this;
}

MappedRegion::~MappedRegion() {
    reset();
}

Ret MappedRegion::init(int fd, size_t offset, size_t size, MappedRegionAccess access) {
    if (data_ != nullptr) {
        return Ret("MappedRegion::init: region is initialized already");
    }
    if (fd < 0) {
        return Ret("MappedRegion::init: invalid file descriptor");
    }
    if (size == 0) {
        return Ret("MappedRegion::init: size must be greater than zero");
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
        return Ret("MappedRegion::init: failed to mmap region");
    }
    if (madvise(region, size, MADV_SEQUENTIAL) != 0) {
        munmap(region, size);
        return Ret("MappedRegion::init: failed to madvise region");
    }

    data_ = static_cast<const uint8_t*>(region);
    size_ = size;
    return Ret(0);
}

void MappedRegion::reset() {
    if (data_ != nullptr) {
        munmap(const_cast<uint8_t*>(data_), size_);
        data_ = nullptr;
        size_ = 0;
    }
}

} // namespace sketch2
