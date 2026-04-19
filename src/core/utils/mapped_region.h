// Declares a small RAII wrapper for a mapped memory region.

#pragma once

#include "utils/shared_types.h"

#include <cstddef>
#include <cstdint>

namespace sketch2 {

enum class MappedRegionAccess {
    ReadOnly,
    Writable,
};

class MappedRegion {
public:
    MappedRegion() = default;

    MappedRegion(const MappedRegion&) = delete;
    MappedRegion& operator=(const MappedRegion&) = delete;

    MappedRegion(MappedRegion&& other) noexcept;
    MappedRegion& operator=(MappedRegion&& other) noexcept;

    ~MappedRegion();

    Ret init(int fd, size_t offset, size_t size, bool is_seq = false,
        MappedRegionAccess access = MappedRegionAccess::ReadOnly);

    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }
    bool empty() const { return data_ == nullptr || size_ == 0; }

    void reset();

private:
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
};

} // namespace sketch2
