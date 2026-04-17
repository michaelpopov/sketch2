#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/utils/shared_types.h"

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
    // Format-specific metadata: max_offset for offsets/bitset, miss_count for misses.
    uint32_t aux_data = 0;
    uint32_t payload_size = 0;
    uint64_t base = 0;
};

struct CompactIdsMappedPrefix {
    CompactIdsHeader header{};
    size_t payload_size = 0;
    size_t consumed = 0;
};

Ret parse_compact_ids_mapped_prefix(
    const uint8_t* data,
    size_t size,
    CompactIdsExtEncoding expected_encoding,
    const char* context,
    size_t* bytes_consumed,
    CompactIdsMappedPrefix* prefix);

Ret validate_compact_ids_empty_header(
    const CompactIdsHeader& header,
    const char* error_message);

Ret write_compact_ids_header(
    FILE* f,
    CompactIdsExtEncoding encoding,
    uint32_t count,
    uint32_t aux_data,
    uint32_t payload_size,
    uint64_t base,
    const std::string& base_message);

CompactIdsExtEncoding choose_compact_ids_encoding(size_t count, uint64_t max_offset);

class CompactIdsAccumulator {
public:
    void init(uint64_t base, size_t count) { base_ = base; offsets_.reserve(count); is_sorted_ = false; }
    void add(uint64_t id) {
        is_sorted_ = false;
        if (id < base_) {
            throw std::invalid_argument("Invalid argument in CompactIdsAccumulator::add(uint64_t id)");
        }
        const uint64_t offset = id - base_;
        if (offset > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            throw std::out_of_range("CompactIdsAccumulator::add: id offset exceeds uint32_t range");
        }
        offsets_.push_back(static_cast<uint32_t>(offset));
    }
    void complete_adding();
    CompactIdsExtEncoding encoding() const;
    size_t size() const { return offsets_.size(); }
    uint64_t operator[](size_t index) const { return base_ + offsets_[index]; }
    bool is_sorted() const { return is_sorted_; }
private:
    uint64_t base_ = 0;
    std::vector<uint32_t> offsets_;
    bool is_sorted_ = false;
};

        
} // namespace sketch2
