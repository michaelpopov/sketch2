// Implements CompactIds.

#include "compact_ids.h"
#include "compact_ids_shared.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>

namespace sketch2 {

template <typename T>
T& CompactIds::storage_as() {
    assert(std::holds_alternative<T>(storage_));
    return std::get<T>(storage_);
}

template <typename T>
const T& CompactIds::storage_as() const {
    assert(std::holds_alternative<T>(storage_));
    return std::get<T>(storage_);
}

void CompactIds::set_storage_kind(StorageKind kind) {
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

void CompactIds::Iterator::next() {
    if (eof()) {
        return;
    }
    ++index_;
}

bool CompactIds::Iterator::eof() const {
    return ids_ == nullptr || index_ >= ids_->count();
}

uint64_t CompactIds::Iterator::id() const {
    if (eof()) {
        throw std::out_of_range("CompactIds::Iterator::id: index out of range");
    }
    return ids_->id(index_);
}

size_t CompactIds::Iterator::index() const {
    if (eof()) {
        throw std::out_of_range("CompactIds::Iterator::index: index out of range");
    }
    return index_;
}

CompactIds::StorageKind CompactIds::detect_storage_kind(const uint64_t* ids, size_t size) {
    const CompactIdsExtEncoding encoding =
        size == 0 ? CompactIdsExtEncoding::Offsets32
                  : choose_compact_ids_encoding(size, ids[size - 1] - ids[0]);
    switch (encoding) {
        case CompactIdsExtEncoding::Offsets32: return StorageKind::Offsets;
        case CompactIdsExtEncoding::Bitset: return StorageKind::Bitset;
        case CompactIdsExtEncoding::Misses32: return StorageKind::Misses;
    }
    return StorageKind::Offsets;
}

Ret CompactIds::init(const CompactIdsAccumulator& accumulator) {
    if (!accumulator.is_sorted()) {
        return Ret("CompactIds::init: accumulator not sorted.");
    }

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
    return Ret("CompactIds::init: unknown encoding");
}

Ret CompactIds::init(const std::vector<uint64_t>& ids) {
    return init(ids.data(), ids.size());
}

Ret CompactIds::init(const uint64_t* ids, size_t size) {
    if (size == 0) {
        set_storage_kind(StorageKind::Offsets);
        return storage_as<CompactIdsOffsets>().init(ids, size);
    }
    if (ids == nullptr) {
        return Ret("CompactIds::init: ids pointer is null");
    }

    if (!std::is_sorted(ids, ids + size)) {
        return Ret("CompactIds::init: ids not sorted");
    }

    set_storage_kind(detect_storage_kind(ids, size));
    switch (storage_kind_) {
        case StorageKind::Offsets: return storage_as<CompactIdsOffsets>().init(ids, size);
        case StorageKind::Bitset: return storage_as<CompactIdsBitset>().init(ids, size);
        case StorageKind::Misses: return storage_as<CompactIdsMisses>().init(ids, size);
    }
    return Ret(0);
}

void CompactIds::clear() {
    switch (storage_kind_) {
        case StorageKind::Offsets: storage_as<CompactIdsOffsets>().clear(); break;
        case StorageKind::Bitset: storage_as<CompactIdsBitset>().clear(); break;
        case StorageKind::Misses: storage_as<CompactIdsMisses>().clear(); break;
    }
    set_storage_kind(StorageKind::Offsets);
}

size_t CompactIds::count() const {
    switch (storage_kind_) {
        case StorageKind::Offsets: return storage_as<CompactIdsOffsets>().count();
        case StorageKind::Misses: return storage_as<CompactIdsMisses>().count();
        case StorageKind::Bitset: return storage_as<CompactIdsBitset>().count();
    }
    return 0;
}

bool CompactIds::empty() const {
    switch (storage_kind_) {
        case StorageKind::Offsets: return storage_as<CompactIdsOffsets>().empty();
        case StorageKind::Misses: return storage_as<CompactIdsMisses>().empty();
        case StorageKind::Bitset: return storage_as<CompactIdsBitset>().empty();
    }
    return true;
}

size_t CompactIds::serialized_size_bytes() const {
    switch (storage_kind_) {
        case StorageKind::Offsets: return storage_as<CompactIdsOffsets>().serialized_size_bytes();
        case StorageKind::Misses: return storage_as<CompactIdsMisses>().serialized_size_bytes();
        case StorageKind::Bitset: return storage_as<CompactIdsBitset>().serialized_size_bytes();
    }
    return 0;
}

uint64_t CompactIds::id(size_t index) const {
    switch (storage_kind_) {
        case StorageKind::Offsets: return storage_as<CompactIdsOffsets>().id(index);
        case StorageKind::Misses: return storage_as<CompactIdsMisses>().id(index);
        case StorageKind::Bitset: return storage_as<CompactIdsBitset>().id(index);
    }
    return 0;
}

uint64_t CompactIds::id_unchecked(size_t index) const {
    switch (storage_kind_) {
        case StorageKind::Offsets: return storage_as<CompactIdsOffsets>().id_unchecked(index);
        case StorageKind::Misses: return storage_as<CompactIdsMisses>().id_unchecked(index);
        case StorageKind::Bitset: return storage_as<CompactIdsBitset>().id_unchecked(index);
    }
    return 0;
}

size_t CompactIds::lower_bound_index(uint64_t id) const {
    switch (storage_kind_) {
        case StorageKind::Offsets: return storage_as<CompactIdsOffsets>().lower_bound_index(id);
        case StorageKind::Misses: return storage_as<CompactIdsMisses>().lower_bound_index(id);
        case StorageKind::Bitset: return storage_as<CompactIdsBitset>().lower_bound_index(id);
    }
    return 0;
}

Ret CompactIds::write(FILE* f, const std::string& error_message) const {
    switch (storage_kind_) {
        case StorageKind::Offsets: return storage_as<CompactIdsOffsets>().write(f, error_message);
        case StorageKind::Misses: return storage_as<CompactIdsMisses>().write(f, error_message);
        case StorageKind::Bitset: return storage_as<CompactIdsBitset>().write(f, error_message);
    }
    return Ret(0);
}

Ret CompactIds::map(const uint8_t* data, size_t size, size_t* bytes_consumed) {
    if (bytes_consumed != nullptr) {
        *bytes_consumed = 0;
    }
    if (data == nullptr) {
        return Ret("CompactIds::map: data pointer is null");
    }
    if (size < sizeof(CompactIdsHeader)) {
        return Ret("CompactIds::map: buffer too small to contain header");
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
            return Ret("CompactIds::map: unknown encoding");
    }
    return Ret(0);
}

CompactIds::Iterator CompactIds::begin() const {
    return Iterator(this);
}

} // namespace sketch2
