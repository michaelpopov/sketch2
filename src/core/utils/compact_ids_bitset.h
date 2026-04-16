// Declares a compact sorted-id container backed by a bitset.

#pragma once

#include "utils/compact_ids_misses.h"

#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace sketch2 {

class CompactIdsAccumulator;

class CompactIdsBitset {
public:
    static constexpr size_t npos = std::numeric_limits<size_t>::max();

    class Iterator {
        friend class CompactIdsBitset;
    public:
        void next();
        bool eof() const;
        uint64_t id() const;
        size_t index() const;

    private:
        Iterator(const CompactIdsBitset* ids);

        const CompactIdsBitset* ids_ = nullptr;
        size_t index_ = 0;
        uint32_t bit_pos_ = 0;
    };

    Ret init(const CompactIdsAccumulator& accumulator);
    Ret init(const std::vector<uint64_t>& ids);
    void clear();

    size_t count() const;
    bool empty() const;
    size_t serialized_size_bytes() const;

    uint64_t id(size_t index) const;
    uint64_t id_unchecked(size_t index) const;
    size_t lower_bound_index(uint64_t id) const;

    Ret write(FILE* f, const std::string& error_message) const;
    Ret map(const uint8_t* data, size_t size, size_t* bytes_consumed);

    Iterator begin() const;

private:
    Ret init_(const CompactIdsAccumulator& accumulator);
    Ret init_(const std::vector<uint64_t>& ids);
    Ret map_(const uint8_t* data, size_t size, size_t* bytes_consumed);
    uint32_t select_bit(size_t target) const;
    size_t rank_bit(uint32_t offset) const;
    uint32_t next_set_bit(uint32_t offset) const;

    uint64_t base_ = 0;
    uint32_t count_ = 0;
    uint32_t span_ = 0;
    const uint8_t* bitset_ = nullptr;
    uint32_t bitset_size_ = 0;
    std::vector<uint8_t> owned_bitset_;
};

} // namespace sketch2
