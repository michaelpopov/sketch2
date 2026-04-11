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

enum class CompactIdsEncoding : uint8_t {
    Offsets32 = 1,
    Bitset = 2,
};

// CompactIds stores a sorted set of uint64 ids as:
// - one uint64 base id
// - one uint32 offset per id relative to that base
//
// This keeps indexed access and binary search simple while reducing memory
// compared with storing every id as uint64_t.
class CompactIds {
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
        friend class CompactIds;
        Iterator(const CompactIds* ids, size_t index) : ids_(ids), index_(index) {}

        const CompactIds* ids_ = nullptr;
        size_t index_ = 0;
    };

    Ret init(const uint64_t* ids, size_t count);
    Ret init(const std::vector<uint64_t>& ids);
    Ret init(uint64_t base, const uint32_t* offsets, size_t count);
    Ret init(uint64_t base, const std::vector<uint32_t>& offsets);
    Ret init(uint64_t base, std::vector<uint32_t>&& offsets);
    Ret read(FILE* f);
    void clear();

    size_t count() const { return offsets_.size(); }
    bool empty() const { return offsets_.empty(); }
    uint64_t base() const { return base_; }
    uint64_t min_id() const;
    uint64_t max_id() const;
    uint32_t offset(size_t index) const;
    uint32_t max_offset() const;
    size_t offsets_storage_size_bytes() const { return offsets_.size() * sizeof(uint32_t); }
    size_t bitset_storage_size_bytes() const;
    CompactIdsEncoding preferred_encoding() const;
    size_t serialized_size_bytes() const;

    uint64_t id(size_t index) const;
    size_t lower_bound_index(uint64_t id) const;
    size_t index_of(uint64_t id) const;
    bool contains(uint64_t id) const;
    const uint32_t* offset_data() const { return offsets_.data(); }
    Ret write_offsets(FILE* f, const std::string& error_message) const;
    Ret write(FILE* f, const std::string& error_message) const;

    Iterator begin() const { return Iterator(this, 0); }

private:
    uint64_t base_ = 0;
    std::vector<uint32_t> offsets_;
};

} // namespace sketch2
