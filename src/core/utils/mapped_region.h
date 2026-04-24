// Declares a small RAII wrapper for a mapped memory region.

#pragma once

#include "utils/shared_types.h"

#include <cstddef>
#include <cstdint>
#include <string>

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
        MappedRegionAccess access = MappedRegionAccess::ReadOnly,
        const std::string& context = "");

    const uint8_t* data() const { return data_; }
    uint8_t* mutable_data() {
        return access_ == MappedRegionAccess::Writable ? data_ : nullptr;
    }
    size_t size() const { return size_; }
    bool empty() const { return data_ == nullptr || size_ == 0; }

    Ret sync(const std::string& error_message) const;
    void reset();

private:
    uint8_t* data_ = nullptr;
    size_t size_ = 0;
    MappedRegionAccess access_ = MappedRegionAccess::ReadOnly;
};

} // namespace sketch2
