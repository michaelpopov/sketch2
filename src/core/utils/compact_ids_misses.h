// Declares a compact sorted-id container backed by a miss-list.

#pragma once

#include "utils/compact_ids_offsets.h"
#include "utils/shared_types.h"

#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace sketch2 {

class CompactIdsAccumulator;

class CompactIdsMisses {
public:
    static constexpr size_t npos = std::numeric_limits<size_t>::max();

    class Iterator {
        friend class CompactIdsMisses;
    public:
        void next();
        bool eof() const;
        uint64_t id() const;
        size_t index() const;

    private:
        friend class CompactIdsOffsets;
        Iterator(const CompactIdsMisses* ids)
            : ids_(ids), index_(0), current_(ids ? ids->base_ : 0), miss_idx_(0) {}

        const CompactIdsMisses* ids_ = nullptr;
        size_t index_ = 0;
        uint64_t current_ = 0;
        size_t miss_idx_ = 0;
    };

    Ret init(const CompactIdsAccumulator& accumulator);
    Ret init(const std::vector<uint64_t>& ids);
    void clear();

    size_t count() const;
    bool empty() const;
    uint64_t base() const;
    uint64_t min_id() const;
    uint64_t max_id() const;
    uint32_t offset(size_t index) const;
    uint32_t max_offset() const;
    size_t offsets_storage_size_bytes() const;
    size_t serialized_size_bytes() const;

    uint64_t id(size_t index) const;
    uint64_t id_unchecked(size_t index) const;
    size_t lower_bound_index(uint64_t id) const;
    size_t index_of(uint64_t id) const;
    bool contains(uint64_t id) const;

    Ret write(FILE* f, const std::string& error_message) const;
    Ret map(const uint8_t* data, size_t size, size_t* bytes_consumed);

    Iterator begin() const;

private:
    uint64_t base_ = 0;
    uint32_t count_ = 0;
    const uint32_t* misses_ = nullptr;
    uint32_t miss_count_ = 0;
    std::vector<uint32_t> owned_misses_;
};

} // namespace sketch2
