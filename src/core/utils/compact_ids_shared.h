#pragma once
#include <cstdint>
#include <vector>

namespace sketch2 {

enum class CompactIdsExtEncoding : uint8_t {
    Offsets32 = 1,
    Bitset = 2,
    Misses32 = 3,
};

struct CompactIdsHeader {
    uint8_t encoding = 0;
    uint8_t reserved0 = 0;
    uint16_t reserved1 = 0;
    uint32_t count = 0;
    uint32_t miss_count = 0;
    uint32_t payload_size = 0;
    uint64_t base = 0;
};

class CompactIdsAccumulator {
public:
    void init(uint64_t base, size_t count) { base_ = base; offsets_.reserve(count); }
    void add(uint64_t id) { offsets_.push_back(static_cast<uint32_t>(id - base_));}
    CompactIdsExtEncoding encoding();
    size_t size() const { return offsets_.size(); }
    uint64_t operator[](size_t index) const { return base_ + offsets_[index]; }
private:
    uint64_t base_ = 0;
    std::vector<uint32_t> offsets_;
};

        
} // namespace sketch2
