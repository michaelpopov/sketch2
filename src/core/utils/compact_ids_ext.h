// Declares the auto-selecting compact sorted-id container.

#pragma once

#include "utils/compact_ids_bitset.h"
#include "utils/compact_ids_misses.h"
#include "utils/compact_ids_offsets.h"

#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <variant>
#include <vector>

namespace sketch2 {

class CompactIdsAccumulator;

class CompactIdsExt {
public:
    static constexpr size_t npos = std::numeric_limits<size_t>::max();

    class Iterator {
        friend class CompactIdsExt;
    public:
        void next();
        bool eof() const;
        uint64_t id() const;
        size_t index() const;

    private:
        explicit Iterator(const CompactIdsExt* ids)
            : ids_(ids) {}

        const CompactIdsExt* ids_ = nullptr;
        size_t index_ = 0;
    };

    Ret init(CompactIdsAccumulator& accumulator);
    Ret init(std::vector<uint64_t>& ids);
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
    enum class StorageKind : uint8_t {
        Offsets,
        Misses,
        Bitset,
    };
    using Storage = std::variant<CompactIdsOffsets, CompactIdsMisses, CompactIdsBitset>;

    template <typename T>
    T& storage_as();

    template <typename T>
    const T& storage_as() const;

    void set_storage_kind(StorageKind kind);

    Storage storage_{CompactIdsOffsets{}};
    StorageKind storage_kind_ = StorageKind::Offsets;

    static StorageKind detect_storage_kind(std::vector<uint64_t>& ids);
};

} // namespace sketch2
