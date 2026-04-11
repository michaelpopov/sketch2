// Implements CompactIds, a sorted-id container backed by 32-bit offsets.

#include "compact_ids.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace sketch2 {

namespace {

struct SerializedHeader {
    uint8_t encoding = 0;
    uint8_t reserved0 = 0;
    uint16_t reserved1 = 0;
    uint32_t count = 0;
    uint32_t max_offset = 0;
    uint32_t payload_size = 0;
    uint64_t base = 0;
};

static_assert(sizeof(SerializedHeader) == 24, "SerializedHeader must stay compact");

Ret validate_offsets(uint64_t base, const uint32_t* offsets, size_t count);
size_t bitset_storage_size_for_offsets(const std::vector<uint32_t>& offsets);
CompactIdsEncoding preferred_encoding_for_offsets(const std::vector<uint32_t>& offsets);
size_t serialized_size_for_offsets(const std::vector<uint32_t>& offsets);
Ret write_serialized_compact_ids(
    FILE* f,
    uint64_t base,
    const std::vector<uint32_t>& offsets,
    const std::string& error_message);

Ret validate_ids_and_fill_offsets(const uint64_t* ids, size_t count, uint64_t* base,
        std::vector<uint32_t>* offsets) {
    if (count == 0) {
        *base = 0;
        offsets->clear();
        return Ret(0);
    }

    if (ids == nullptr) {
        return Ret("CompactIds::init: ids pointer is null");
    }

    const uint64_t local_base = ids[0];
    offsets->clear();
    offsets->reserve(count);

    uint64_t prev = 0;
    for (size_t i = 0; i < count; ++i) {
        const uint64_t current = ids[i];
        if (i > 0 && prev >= current) {
            return Ret("CompactIds::init: ids must be strictly increasing");
        }

        const uint64_t offset = current - local_base;
        if (offset > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            return Ret("CompactIds::init: id offset exceeds uint32_t range");
        }

        offsets->push_back(static_cast<uint32_t>(offset));
        prev = current;
    }

    *base = local_base;
    return Ret(0);
}

Ret validate_offsets(uint64_t base, const uint32_t* offsets, size_t count) {
    uint32_t prev = 0;
    for (size_t i = 0; i < count; ++i) {
        const uint32_t current = offsets[i];
        if (i > 0 && prev >= current) {
            return Ret("CompactIds::init: offsets must be strictly increasing");
        }
        if (base > std::numeric_limits<uint64_t>::max() - static_cast<uint64_t>(current)) {
            return Ret("CompactIds::init: base plus offset overflows uint64_t");
        }
        prev = current;
    }

    return Ret(0);
}

Ret validate_offsets(uint64_t base, const std::vector<uint32_t>& offsets) {
    return validate_offsets(base, offsets.data(), offsets.size());
}

size_t bitset_payload_size(uint32_t max_offset) {
    return (static_cast<size_t>(max_offset) + 8u) / 8u;
}

bool is_canonical_bitset_encoding(uint32_t count, size_t payload_size) {
    return payload_size < static_cast<size_t>(count) * sizeof(uint32_t);
}

size_t bitset_storage_size_for_offsets(const std::vector<uint32_t>& offsets) {
    if (offsets.empty()) {
        return 0;
    }
    return bitset_payload_size(offsets.back());
}

CompactIdsEncoding preferred_encoding_for_offsets(const std::vector<uint32_t>& offsets) {
    if (offsets.empty()) {
        return CompactIdsEncoding::Offsets32;
    }
    return bitset_storage_size_for_offsets(offsets) < offsets.size() * sizeof(uint32_t)
        ? CompactIdsEncoding::Bitset
        : CompactIdsEncoding::Offsets32;
}

size_t serialized_size_for_offsets(const std::vector<uint32_t>& offsets) {
    const size_t header_size = sizeof(SerializedHeader);
    if (offsets.empty()) {
        return header_size;
    }
    if (preferred_encoding_for_offsets(offsets) == CompactIdsEncoding::Bitset) {
        return header_size + bitset_storage_size_for_offsets(offsets);
    }
    return header_size + offsets.size() * sizeof(uint32_t);
}

Ret decode_bitset_payload(const uint8_t* payload, size_t payload_size, uint32_t count, uint32_t max_offset,
        std::vector<uint32_t>* offsets) {
    offsets->clear();
    offsets->reserve(count);

    const size_t expected_payload_size = bitset_payload_size(max_offset);
    if (payload_size != expected_payload_size) {
        return Ret("CompactIds::read: malformed bitset payload size");
    }
    if (!is_canonical_bitset_encoding(count, payload_size)) {
        return Ret("CompactIds::read: non-canonical bitset encoding");
    }
    if (payload_size > 0) {
        // Bits above max_offset in the last byte are padding and must stay zero.
        const uint32_t used_bits_in_last_byte = (max_offset + 1u) & 7u;
        if (used_bits_in_last_byte != 0u) {
            const uint8_t padding_mask = static_cast<uint8_t>(~((1u << used_bits_in_last_byte) - 1u));
            if ((payload[payload_size - 1] & padding_mask) != 0u) {
                return Ret("CompactIds::read: malformed bitset payload tail bits");
            }
        }
    }

    // Enumerate only the set bits instead of probing every offset in the span.
    size_t byte_index = 0;
    for (; byte_index + sizeof(uint64_t) <= payload_size; byte_index += sizeof(uint64_t)) {
        uint64_t word = 0;
        std::memcpy(&word, payload + byte_index, sizeof(word));
        while (word != 0) {
            const uint32_t bit_index = static_cast<uint32_t>(__builtin_ctzll(word));
            offsets->push_back(static_cast<uint32_t>(byte_index * 8u + bit_index));
            if (offsets->size() > count) {
                return Ret("CompactIds::read: bitset count does not match header");
            }
            word &= (word - 1u);
        }
    }

    for (; byte_index < payload_size; ++byte_index) {
        uint8_t byte = payload[byte_index];
        while (byte != 0) {
            const uint32_t bit_index = static_cast<uint32_t>(__builtin_ctz(static_cast<unsigned int>(byte)));
            offsets->push_back(static_cast<uint32_t>(byte_index * 8u + bit_index));
            if (offsets->size() > count) {
                return Ret("CompactIds::read: bitset count does not match header");
            }
            byte = static_cast<uint8_t>(byte & static_cast<uint8_t>(byte - 1u));
        }
    }

    if (offsets->size() != count) {
        return Ret("CompactIds::read: bitset count does not match header");
    }

    return Ret(0);
}

Ret write_serialized_compact_ids(
        FILE* f,
        uint64_t base,
        const std::vector<uint32_t>& offsets,
        const std::string& error_message) {
    const std::string base_message = error_message.empty() ? "CompactIds::write failed" : error_message;
    if (f == nullptr) {
        return Ret(base_message + ": file handle is null");
    }

    if (offsets.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        return Ret(base_message + ": id count exceeds uint32_t range");
    }

    const CompactIdsEncoding encoding = preferred_encoding_for_offsets(offsets);
    SerializedHeader hdr{};
    hdr.encoding = static_cast<uint8_t>(encoding);
    hdr.count = static_cast<uint32_t>(offsets.size());
    hdr.max_offset = offsets.empty() ? 0 : offsets.back();
    hdr.payload_size = 0;
    hdr.base = offsets.empty() ? 0 : base;

    if (encoding == CompactIdsEncoding::Bitset) {
        const size_t bitset_size = bitset_storage_size_for_offsets(offsets);
        if (bitset_size > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            return Ret(base_message + ": bitset payload exceeds uint32_t range");
        }
        hdr.payload_size = static_cast<uint32_t>(bitset_size);
    } else {
        const size_t offsets_size = offsets.size() * sizeof(uint32_t);
        if (offsets_size > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            return Ret(base_message + ": offsets payload exceeds uint32_t range");
        }
        hdr.payload_size = static_cast<uint32_t>(offsets_size);
    }

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        return Ret(base_message);
    }

    if (offsets.empty()) {
        return Ret(0);
    }

    if (encoding == CompactIdsEncoding::Offsets32) {
        if (fwrite(offsets.data(), sizeof(uint32_t), offsets.size(), f) != offsets.size()) {
            return Ret(base_message);
        }
        return Ret(0);
    }

    std::vector<uint8_t> bitset(bitset_storage_size_for_offsets(offsets), 0);
    for (uint32_t value : offsets) {
        const size_t byte_index = static_cast<size_t>(value) >> 3;
        const uint8_t mask = static_cast<uint8_t>(1u << (value & 7u));
        bitset[byte_index] |= mask;
    }
    if (fwrite(bitset.data(), 1, bitset.size(), f) != bitset.size()) {
        return Ret(base_message);
    }
    return Ret(0);
}

} // namespace

void CompactIdsBuilder::clear() {
    base_ = 0;
    offsets_.clear();
}

void CompactIdsBuilder::reserve(size_t count) {
    offsets_.reserve(count);
}

Ret CompactIdsBuilder::append(uint64_t id) {
    try {
        if (offsets_.empty()) {
            base_ = id;
            offsets_.push_back(0);
            return Ret(0);
        }

        const uint64_t previous_id = base_ + offsets_.back();
        if (id <= previous_id) {
            return Ret("CompactIds::init: ids must be strictly increasing");
        }

        const uint64_t offset = id - base_;
        if (offset > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            return Ret("CompactIds::init: id offset exceeds uint32_t range");
        }

        offsets_.push_back(static_cast<uint32_t>(offset));
        return Ret(0);
    } catch (const std::exception& ex) {
        return Ret(std::string("CompactIdsBuilder::append: ") + ex.what());
    }
}

uint64_t CompactIdsBuilder::min_id() const {
    if (empty()) {
        throw std::out_of_range("CompactIdsBuilder::min_id: container is empty");
    }
    return base_ + offsets_.front();
}

uint64_t CompactIdsBuilder::max_id() const {
    if (empty()) {
        throw std::out_of_range("CompactIdsBuilder::max_id: container is empty");
    }
    return base_ + offsets_.back();
}

size_t CompactIdsBuilder::bitset_storage_size_bytes() const {
    return bitset_storage_size_for_offsets(offsets_);
}

CompactIdsEncoding CompactIdsBuilder::preferred_encoding() const {
    return preferred_encoding_for_offsets(offsets_);
}

size_t CompactIdsBuilder::serialized_size_bytes() const {
    return serialized_size_for_offsets(offsets_);
}

Ret CompactIdsBuilder::write(FILE* f, const std::string& error_message) const {
    return write_serialized_compact_ids(f, base_, offsets_, error_message);
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

Ret CompactIds::init(const std::vector<uint64_t>& ids) {
    try {
        uint64_t new_base = 0;
        std::vector<uint32_t> new_offsets;
        CHECK(validate_ids_and_fill_offsets(ids.data(), ids.size(), &new_base, &new_offsets));
        base_ = new_base;
        offsets_ = std::move(new_offsets);
        return Ret(0);
    } catch (const std::exception& ex) {
        return Ret(std::string("CompactIds::init: ") + ex.what());
    }
}

Ret CompactIds::init(uint64_t base, std::vector<uint32_t>&& offsets) {
    try {
        if (offsets.empty()) {
            clear();
            return Ret(0);
        }
        CHECK(validate_offsets(base, offsets));
        base_ = base;
        offsets_ = std::move(offsets);
        return Ret(0);
    } catch (const std::exception& ex) {
        return Ret(std::string("CompactIds::init: ") + ex.what());
    }
}

Ret CompactIds::read(const uint8_t* data, size_t size, size_t* bytes_consumed) {
    try {
        if (bytes_consumed != nullptr) {
            *bytes_consumed = 0;
        }
        if (data == nullptr) {
            return Ret("CompactIds::read: data pointer is null");
        }
        if (size < sizeof(SerializedHeader)) {
            return Ret("CompactIds::read: buffer too small to contain header");
        }

        SerializedHeader hdr{};
        std::memcpy(&hdr, data, sizeof(SerializedHeader));
        const size_t payload_size = static_cast<size_t>(hdr.payload_size);
        if (payload_size > size - sizeof(SerializedHeader)) {
            return Ret("CompactIds::read: truncated payload");
        }
        const size_t consumed = sizeof(SerializedHeader) + payload_size;
        const uint8_t* payload = data + sizeof(SerializedHeader);

        const CompactIdsEncoding encoding = static_cast<CompactIdsEncoding>(hdr.encoding);
        if (encoding != CompactIdsEncoding::Offsets32 && encoding != CompactIdsEncoding::Bitset) {
            return Ret("CompactIds::read: unknown encoding");
        }

        if (hdr.count == 0) {
            if (hdr.max_offset != 0 || hdr.payload_size != 0 || hdr.base != 0) {
                return Ret("CompactIds::read: malformed empty payload header");
            }
            clear();
        } else if (encoding == CompactIdsEncoding::Offsets32) {
            const size_t expected_payload_size = static_cast<size_t>(hdr.count) * sizeof(uint32_t);
            if (hdr.payload_size != expected_payload_size) {
                return Ret("CompactIds::read: malformed offsets payload size");
            }
            std::vector<uint32_t> offsets(hdr.count);
            std::memcpy(offsets.data(), payload, payload_size);
            CHECK(init(hdr.base, std::move(offsets)));
        } else {
            const size_t expected_payload_size = bitset_payload_size(hdr.max_offset);
            if (hdr.payload_size != expected_payload_size) {
                return Ret("CompactIds::read: malformed bitset payload size");
            }
            std::vector<uint32_t> offsets;
            CHECK(decode_bitset_payload(payload, payload_size, hdr.count, hdr.max_offset, &offsets));
            CHECK(init(hdr.base, std::move(offsets)));
        }

        if (bytes_consumed != nullptr) {
            *bytes_consumed = consumed;
        }
        return Ret(0);
    } catch (const std::exception& ex) {
        return Ret(std::string("CompactIds::read: ") + ex.what());
    }
}

void CompactIds::clear() {
    base_ = 0;
    offsets_.clear();
}

uint64_t CompactIds::min_id() const {
    if (empty()) {
        throw std::out_of_range("CompactIds::min_id: container is empty");
    }
    return base_ + offsets_.front();
}

uint64_t CompactIds::max_id() const {
    if (empty()) {
        throw std::out_of_range("CompactIds::max_id: container is empty");
    }
    return base_ + offsets_.back();
}

uint32_t CompactIds::offset(size_t index) const {
    if (index >= count()) {
        throw std::out_of_range("CompactIds::offset: index out of range");
    }
    return offsets_[index];
}

uint32_t CompactIds::max_offset() const {
    if (empty()) {
        throw std::out_of_range("CompactIds::max_offset: container is empty");
    }
    return offsets_.back();
}

size_t CompactIds::bitset_storage_size_bytes() const {
    return bitset_storage_size_for_offsets(offsets_);
}

CompactIdsEncoding CompactIds::preferred_encoding() const {
    return preferred_encoding_for_offsets(offsets_);
}

size_t CompactIds::serialized_size_bytes() const {
    return serialized_size_for_offsets(offsets_);
}

uint64_t CompactIds::id(size_t index) const {
    if (index >= count()) {
        throw std::out_of_range("CompactIds::id: index out of range");
    }
    return base_ + offsets_[index];
}

size_t CompactIds::lower_bound_index(uint64_t value) const {
    if (empty()) {
        return 0;
    }

    if (value <= base_) {
        return 0;
    }

    const uint64_t relative = value - base_;
    if (relative > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return count();
    }

    const auto it = std::lower_bound(
        offsets_.begin(), offsets_.end(), static_cast<uint32_t>(relative));
    return static_cast<size_t>(it - offsets_.begin());
}

size_t CompactIds::index_of(uint64_t value) const {
    const size_t index = lower_bound_index(value);
    if (index >= count()) {
        return npos;
    }
    return id(index) == value ? index : npos;
}

bool CompactIds::contains(uint64_t value) const {
    return index_of(value) != npos;
}

Ret CompactIds::write(FILE* f, const std::string& error_message) const {
    return write_serialized_compact_ids(f, base_, offsets_, error_message);
}

} // namespace sketch2
