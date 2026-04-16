// Implements CompactIdsExt.

#include "compact_ids_ext.h"
#include "compact_ids_shared.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace sketch2 {

CompactIdsExtEncoding CompactIdsAccumulator::encoding() {
    if (offsets_.empty()) {
        return CompactIdsExtEncoding::Offsets32;
    }

    if (!std::is_sorted(offsets_.begin(), offsets_.end())) {
        std::sort(offsets_.begin(), offsets_.end());
    }

    const size_t offsets_bytes = offsets_.size() * sizeof(uint32_t);
    const size_t bitset_bytes = (static_cast<size_t>(offsets_.back()) + 8u) / 8u;
    const size_t misses_count =
        static_cast<size_t>(offsets_.back()) + 1u - offsets_.size();
    const size_t misses_bytes = misses_count * sizeof(uint32_t);

    if (offsets_bytes <= misses_bytes && offsets_bytes <= bitset_bytes) {
        return CompactIdsExtEncoding::Offsets32;
    }
    if (bitset_bytes < misses_bytes) {
        return CompactIdsExtEncoding::Bitset;
    }
    return CompactIdsExtEncoding::Misses32;
}

void CompactIdsExt::Iterator::next() {
    if (eof()) {
        return;
    }
    ++index_;
}

bool CompactIdsExt::Iterator::eof() const {
    return ids_ == nullptr || index_ >= ids_->count();
}

uint64_t CompactIdsExt::Iterator::id() const {
    if (eof()) {
        throw std::out_of_range("CompactIdsExt::Iterator::id: index out of range");
    }
    return ids_->id(index_);
}

size_t CompactIdsExt::Iterator::index() const {
    if (eof()) {
        throw std::out_of_range("CompactIdsExt::Iterator::index: index out of range");
    }
    return index_;
}

CompactIdsExt::StorageKind CompactIdsExt::detect_storage_kind(std::vector<uint64_t>& ids) {
    if (ids.empty()) {
        return StorageKind::Offsets;
    }

    if (!std::is_sorted(ids.begin(), ids.end())) {
        std::sort(ids.begin(), ids.end());
    }

    const size_t offsets_bytes = ids.size() * sizeof(uint32_t);
    const size_t bitset_bytes = (static_cast<size_t>(ids.back() - ids.front()) + 8u) / 8u;
    const size_t misses_count =
        static_cast<size_t>(ids.back() - ids.front()) + 1u - ids.size();
    const size_t misses_bytes = misses_count * sizeof(uint32_t);

    if (offsets_bytes <= misses_bytes && offsets_bytes <= bitset_bytes) {
        return StorageKind::Offsets;
    }
    if (bitset_bytes < misses_bytes) {
        return StorageKind::Bitset;
    }
    return StorageKind::Misses;
}

Ret CompactIdsExt::init(CompactIdsAccumulator& accumulator) {
    switch (accumulator.encoding()) {
        case CompactIdsExtEncoding::Offsets32:
            storage_kind_ = StorageKind::Offsets;
            return offsets_.init(accumulator);
        case CompactIdsExtEncoding::Bitset:
            storage_kind_ = StorageKind::Bitset;
            return bitset_.init(accumulator);
        case CompactIdsExtEncoding::Misses32:
            storage_kind_ = StorageKind::Misses;
            return misses_.init(accumulator);
    }
    return Ret("CompactIdsExt::init: unknown encoding");
}

Ret CompactIdsExt::init(std::vector<uint64_t>& ids) {
    storage_kind_ = detect_storage_kind(ids);
    switch (storage_kind_) {
        case StorageKind::Offsets: return offsets_.init(ids);
        case StorageKind::Bitset: return bitset_.init(ids);
        case StorageKind::Misses: return misses_.init(ids);
    }
    return Ret(0);
}

void CompactIdsExt::clear() {
    switch (storage_kind_) {
        case StorageKind::Offsets: offsets_.clear(); break;
        case StorageKind::Bitset: bitset_.clear(); break;
        case StorageKind::Misses: misses_.clear(); break;
    }
    storage_kind_ = StorageKind::Offsets;
}

size_t CompactIdsExt::count() const {
    switch (storage_kind_) {
        case StorageKind::Offsets: return offsets_.count();
        case StorageKind::Misses: return misses_.count();
        case StorageKind::Bitset: return bitset_.count();
    }
    return 0;
}

bool CompactIdsExt::empty() const {
    switch (storage_kind_) {
        case StorageKind::Offsets: return offsets_.empty();
        case StorageKind::Misses: return misses_.empty();
        case StorageKind::Bitset: return bitset_.empty();
    }
    return true;
}

size_t CompactIdsExt::serialized_size_bytes() const {
    switch (storage_kind_) {
        case StorageKind::Offsets: return offsets_.serialized_size_bytes();
        case StorageKind::Misses: return misses_.serialized_size_bytes();
        case StorageKind::Bitset: return bitset_.serialized_size_bytes();
    }
    return 0;
}

uint64_t CompactIdsExt::id(size_t index) const {
    switch (storage_kind_) {
        case StorageKind::Offsets: return offsets_.id(index);
        case StorageKind::Misses: return misses_.id(index);
        case StorageKind::Bitset: return bitset_.id(index);
    }
    return 0;
}

uint64_t CompactIdsExt::id_unchecked(size_t index) const {
    switch (storage_kind_) {
        case StorageKind::Offsets: return offsets_.id_unchecked(index);
        case StorageKind::Misses: return misses_.id_unchecked(index);
        case StorageKind::Bitset: return bitset_.id_unchecked(index);
    }
    return 0;
}

size_t CompactIdsExt::lower_bound_index(uint64_t id) const {
    switch (storage_kind_) {
        case StorageKind::Offsets: return offsets_.lower_bound_index(id);
        case StorageKind::Misses: return misses_.lower_bound_index(id);
        case StorageKind::Bitset: return bitset_.lower_bound_index(id);
    }
    return 0;
}

Ret CompactIdsExt::write(FILE* f, const std::string& error_message) const {
    switch (storage_kind_) {
        case StorageKind::Offsets: return offsets_.write(f, error_message);
        case StorageKind::Misses: return misses_.write(f, error_message);
        case StorageKind::Bitset: return bitset_.write(f, error_message);
    }
    return Ret(0);
}

Ret CompactIdsExt::map(const uint8_t* data, size_t size, size_t* bytes_consumed) {
    if (bytes_consumed != nullptr) {
        *bytes_consumed = 0;
    }
    if (data == nullptr) {
        return Ret("CompactIdsExt::map: data pointer is null");
    }
    if (size < sizeof(CompactIdsHeader)) {
        return Ret("CompactIdsExt::map: buffer too small to contain header");
    }

    CompactIdsHeader hdr{};
    std::memcpy(&hdr, data, sizeof(CompactIdsHeader));

    switch (hdr.encoding) {
        case static_cast<uint8_t>(CompactIdsExtEncoding::Offsets32):
            CHECK(offsets_.map(data, size, bytes_consumed));
            misses_.clear();
            bitset_.clear();
            storage_kind_ = StorageKind::Offsets;
            return Ret(0);
        case static_cast<uint8_t>(CompactIdsExtEncoding::Bitset):
            offsets_.clear();
            CHECK(bitset_.map(data, size, bytes_consumed));
            misses_.clear();
            storage_kind_ = StorageKind::Bitset;
            return Ret(0);
        case static_cast<uint8_t>(CompactIdsExtEncoding::Misses32):
            offsets_.clear();
            CHECK(misses_.map(data, size, bytes_consumed));
            bitset_.clear();
            storage_kind_ = StorageKind::Misses;
            return Ret(0);
        default:
            return Ret("CompactIdsExt::map: unknown encoding");
    }
    return Ret(0);
}

CompactIdsExt::Iterator CompactIdsExt::begin() const {
    return Iterator(this);
}

} // namespace sketch2
