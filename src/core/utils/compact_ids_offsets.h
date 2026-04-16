// Declares a compact sorted-id container backed by 32-bit offsets from a base id.

#pragma once

#include "utils/shared_types.h"

#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace sketch2 {

class CompactIdsAccumulator;

// CompactIdsOffsets stores a sorted set of uint64 ids as:
// - one uint64 base id
// - one uint32 offset per id relative to that base
//
// This keeps indexed access and binary search simple while reducing memory
// compared with storing every id as uint64_t.
class CompactIdsOffsets {
public:
    static constexpr size_t npos = std::numeric_limits<size_t>::max();

    class Iterator {
    public:
        // Advances to the next item. Calling next() at EOF is a no-op.
        void next();
        bool eof() const;
        uint64_t id() const;
        size_t index() const;

    private:
        friend class CompactIdsOffsets;
        Iterator(const CompactIdsOffsets* ids, size_t index) : ids_(ids), index_(index) {}

        const CompactIdsOffsets* ids_ = nullptr;
        size_t index_ = 0;
    };

    Ret init(const CompactIdsAccumulator& accumulator);
    Ret init(const std::vector<uint64_t>& ids);

    Ret map(const uint8_t* data, size_t size, size_t* bytes_consumed);
    void clear();

    size_t count() const { return count_; }
    bool empty() const { return count_ == 0; }
    uint64_t base() const { return base_; }
    uint64_t min_id() const;
    uint64_t max_id() const;
    uint32_t offset(size_t index) const;
    uint32_t max_offset() const;
    size_t offsets_storage_size_bytes() const { return static_cast<size_t>(count_) * sizeof(uint32_t); }
    size_t serialized_size_bytes() const;

    uint64_t id(size_t index) const;
    // Fast path for tight loops that have already validated index bounds.
    uint64_t id_unchecked(size_t index) const { return base_ + offsets_[index]; }
    size_t lower_bound_index(uint64_t id) const;
    size_t index_of(uint64_t id) const;
    bool contains(uint64_t id) const;
    Ret write(FILE* f, const std::string& error_message) const;

    Iterator begin() const { return Iterator(this, 0); }

private:
    uint64_t base_ = 0;
    uint32_t count_ = 0;
    const uint32_t* offsets_ = nullptr;
    std::vector<uint32_t> owned_offsets_;
};

} // namespace sketch2
