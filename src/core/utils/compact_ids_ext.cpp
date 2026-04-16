// Implements CompactIdsExt.

#include "compact_ids_ext.h"
#include "compact_ids_shared.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>

namespace sketch2 {

template <typename T>
T& CompactIdsExt::storage_as() {
    assert(std::holds_alternative<T>(storage_));
    return std::get<T>(storage_);
}

template <typename T>
const T& CompactIdsExt::storage_as() const {
    assert(std::holds_alternative<T>(storage_));
    return std::get<T>(storage_);
}

void CompactIdsExt::set_storage_kind(StorageKind kind) {
    storage_kind_ = kind;
    switch (storage_kind_) {
        case StorageKind::Offsets:
            if (!std::holds_alternative<CompactIdsOffsets>(storage_)) {
                storage_.emplace<CompactIdsOffsets>();
            }
            break;
        case StorageKind::Misses:
            if (!std::holds_alternative<CompactIdsMisses>(storage_)) {
                storage_.emplace<CompactIdsMisses>();
            }
            break;
        case StorageKind::Bitset:
            if (!std::holds_alternative<CompactIdsBitset>(storage_)) {
                storage_.emplace<CompactIdsBitset>();
            }
            break;
    }
}

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
            set_storage_kind(StorageKind::Offsets);
            return storage_as<CompactIdsOffsets>().init(accumulator);
        case CompactIdsExtEncoding::Bitset:
            set_storage_kind(StorageKind::Bitset);
            return storage_as<CompactIdsBitset>().init(accumulator);
        case CompactIdsExtEncoding::Misses32:
            set_storage_kind(StorageKind::Misses);
            return storage_as<CompactIdsMisses>().init(accumulator);
    }
    return Ret("CompactIdsExt::init: unknown encoding");
}

Ret CompactIdsExt::init(std::vector<uint64_t>& ids) {
    set_storage_kind(detect_storage_kind(ids));
    switch (storage_kind_) {
        case StorageKind::Offsets: return storage_as<CompactIdsOffsets>().init(ids);
        case StorageKind::Bitset: return storage_as<CompactIdsBitset>().init(ids);
        case StorageKind::Misses: return storage_as<CompactIdsMisses>().init(ids);
    }
    return Ret(0);
}

void CompactIdsExt::clear() {
    switch (storage_kind_) {
        case StorageKind::Offsets: storage_as<CompactIdsOffsets>().clear(); break;
        case StorageKind::Bitset: storage_as<CompactIdsBitset>().clear(); break;
        case StorageKind::Misses: storage_as<CompactIdsMisses>().clear(); break;
    }
    set_storage_kind(StorageKind::Offsets);
}

size_t CompactIdsExt::count() const {
    switch (storage_kind_) {
        case StorageKind::Offsets: return storage_as<CompactIdsOffsets>().count();
        case StorageKind::Misses: return storage_as<CompactIdsMisses>().count();
        case StorageKind::Bitset: return storage_as<CompactIdsBitset>().count();
    }
    return 0;
}

bool CompactIdsExt::empty() const {
    switch (storage_kind_) {
        case StorageKind::Offsets: return storage_as<CompactIdsOffsets>().empty();
        case StorageKind::Misses: return storage_as<CompactIdsMisses>().empty();
        case StorageKind::Bitset: return storage_as<CompactIdsBitset>().empty();
    }
    return true;
}

size_t CompactIdsExt::serialized_size_bytes() const {
    switch (storage_kind_) {
        case StorageKind::Offsets: return storage_as<CompactIdsOffsets>().serialized_size_bytes();
        case StorageKind::Misses: return storage_as<CompactIdsMisses>().serialized_size_bytes();
        case StorageKind::Bitset: return storage_as<CompactIdsBitset>().serialized_size_bytes();
    }
    return 0;
}

uint64_t CompactIdsExt::id(size_t index) const {
    switch (storage_kind_) {
        case StorageKind::Offsets: return storage_as<CompactIdsOffsets>().id(index);
        case StorageKind::Misses: return storage_as<CompactIdsMisses>().id(index);
        case StorageKind::Bitset: return storage_as<CompactIdsBitset>().id(index);
    }
    return 0;
}

uint64_t CompactIdsExt::id_unchecked(size_t index) const {
    switch (storage_kind_) {
        case StorageKind::Offsets: return storage_as<CompactIdsOffsets>().id_unchecked(index);
        case StorageKind::Misses: return storage_as<CompactIdsMisses>().id_unchecked(index);
        case StorageKind::Bitset: return storage_as<CompactIdsBitset>().id_unchecked(index);
    }
    return 0;
}

size_t CompactIdsExt::lower_bound_index(uint64_t id) const {
    switch (storage_kind_) {
        case StorageKind::Offsets: return storage_as<CompactIdsOffsets>().lower_bound_index(id);
        case StorageKind::Misses: return storage_as<CompactIdsMisses>().lower_bound_index(id);
        case StorageKind::Bitset: return storage_as<CompactIdsBitset>().lower_bound_index(id);
    }
    return 0;
}

Ret CompactIdsExt::write(FILE* f, const std::string& error_message) const {
    switch (storage_kind_) {
        case StorageKind::Offsets: return storage_as<CompactIdsOffsets>().write(f, error_message);
        case StorageKind::Misses: return storage_as<CompactIdsMisses>().write(f, error_message);
        case StorageKind::Bitset: return storage_as<CompactIdsBitset>().write(f, error_message);
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
            set_storage_kind(StorageKind::Offsets);
            CHECK(storage_as<CompactIdsOffsets>().map(data, size, bytes_consumed));
            return Ret(0);
        case static_cast<uint8_t>(CompactIdsExtEncoding::Bitset):
            set_storage_kind(StorageKind::Bitset);
            CHECK(storage_as<CompactIdsBitset>().map(data, size, bytes_consumed));
            return Ret(0);
        case static_cast<uint8_t>(CompactIdsExtEncoding::Misses32):
            set_storage_kind(StorageKind::Misses);
            CHECK(storage_as<CompactIdsMisses>().map(data, size, bytes_consumed));
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
