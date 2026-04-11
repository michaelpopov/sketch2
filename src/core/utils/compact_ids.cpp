// Implements CompactIds, a sorted-id container backed by 32-bit offsets.

#include "compact_ids.h"

#include <algorithm>
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

Ret validate_offsets_and_set_base(uint64_t base, const uint32_t* offsets, size_t count,
        uint64_t* out_base, std::vector<uint32_t>* out_offsets) {
    if (count == 0) {
        *out_base = 0;
        out_offsets->clear();
        return Ret(0);
    }

    if (offsets == nullptr) {
        return Ret("CompactIds::init: offsets pointer is null");
    }

    out_offsets->clear();
    out_offsets->reserve(count);

    CHECK(validate_offsets(base, offsets, count));
    out_offsets->insert(out_offsets->end(), offsets, offsets + count);

    *out_base = base;
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

Ret read_exact(FILE* f, void* data, size_t size, const std::string& error_message) {
    if (size == 0) {
        return Ret(0);
    }
    if (fread(data, 1, size, f) != size) {
        return Ret(error_message);
    }
    return Ret(0);
}

size_t bitset_payload_size(uint32_t max_offset) {
    return (static_cast<size_t>(max_offset) + 8u) / 8u;
}

Ret decode_bitset_payload(const std::vector<uint8_t>& payload, uint32_t count, uint32_t max_offset,
        std::vector<uint32_t>* offsets) {
    offsets->clear();
    offsets->reserve(count);

    const size_t expected_payload_size = bitset_payload_size(max_offset);
    if (payload.size() != expected_payload_size) {
        return Ret("CompactIds::read: malformed bitset payload size");
    }
    if (!payload.empty()) {
        // Bits above max_offset in the last byte are padding and must stay zero.
        const uint32_t used_bits_in_last_byte = (max_offset + 1u) & 7u;
        if (used_bits_in_last_byte != 0u) {
            const uint8_t padding_mask = static_cast<uint8_t>(~((1u << used_bits_in_last_byte) - 1u));
            if ((payload.back() & padding_mask) != 0u) {
                return Ret("CompactIds::read: malformed bitset payload tail bits");
            }
        }
    }

    // Use a 64-bit loop bound to avoid uint32_t wraparound when max_offset
    // is UINT32_MAX.
    const uint64_t limit = static_cast<uint64_t>(max_offset) + 1u;
    for (uint64_t offset = 0; offset < limit; ++offset) {
        const size_t byte_index = static_cast<size_t>(offset) >> 3;
        const uint8_t mask = static_cast<uint8_t>(1u << (offset & 7u));
        if ((payload[byte_index] & mask) != 0u) {
            offsets->push_back(static_cast<uint32_t>(offset));
        }
    }

    if (offsets->size() != count) {
        return Ret("CompactIds::read: bitset count does not match header");
    }

    return Ret(0);
}

} // namespace

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

Ret CompactIds::init(const uint64_t* ids, size_t count) {
    try {
        uint64_t new_base = 0;
        std::vector<uint32_t> new_offsets;
        CHECK(validate_ids_and_fill_offsets(ids, count, &new_base, &new_offsets));
        base_ = new_base;
        offsets_ = std::move(new_offsets);
        return Ret(0);
    } catch (const std::exception& ex) {
        return Ret(std::string("CompactIds::init: ") + ex.what());
    }
}

Ret CompactIds::init(const std::vector<uint64_t>& ids) {
    return init(ids.data(), ids.size());
}

Ret CompactIds::init(uint64_t base, const uint32_t* offsets, size_t count) {
    try {
        uint64_t new_base = 0;
        std::vector<uint32_t> new_offsets;
        CHECK(validate_offsets_and_set_base(base, offsets, count, &new_base, &new_offsets));
        base_ = new_base;
        offsets_ = std::move(new_offsets);
        return Ret(0);
    } catch (const std::exception& ex) {
        return Ret(std::string("CompactIds::init: ") + ex.what());
    }
}

Ret CompactIds::init(uint64_t base, const std::vector<uint32_t>& offsets) {
    return init(base, offsets.data(), offsets.size());
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

Ret CompactIds::read(FILE* f) {
    try {
        if (f == nullptr) {
            return Ret("CompactIds::read: file handle is null");
        }

        SerializedHeader hdr{};
        CHECK(read_exact(f, &hdr, sizeof(hdr), "CompactIds::read: failed to read header"));

        const CompactIdsEncoding encoding = static_cast<CompactIdsEncoding>(hdr.encoding);
        if (encoding != CompactIdsEncoding::Offsets32 && encoding != CompactIdsEncoding::Bitset) {
            return Ret("CompactIds::read: unknown encoding");
        }

        if (hdr.count == 0) {
            if (hdr.max_offset != 0 || hdr.payload_size != 0 || hdr.base != 0) {
                return Ret("CompactIds::read: malformed empty payload header");
            }
            clear();
            return Ret(0);
        }

        if (encoding == CompactIdsEncoding::Offsets32) {
            const size_t expected_payload_size = static_cast<size_t>(hdr.count) * sizeof(uint32_t);
            if (hdr.payload_size != expected_payload_size) {
                return Ret("CompactIds::read: malformed offsets payload size");
            }
            std::vector<uint32_t> offsets(hdr.count);
            CHECK(read_exact(f, offsets.data(), hdr.payload_size, "CompactIds::read: failed to read offsets payload"));
            return init(hdr.base, std::move(offsets));
        }

        const size_t expected_payload_size = bitset_payload_size(hdr.max_offset);
        if (hdr.payload_size != expected_payload_size) {
            return Ret("CompactIds::read: malformed bitset payload size");
        }
        std::vector<uint8_t> payload(hdr.payload_size);
        CHECK(read_exact(f, payload.data(), payload.size(), "CompactIds::read: failed to read bitset payload"));

        std::vector<uint32_t> offsets;
        CHECK(decode_bitset_payload(payload, hdr.count, hdr.max_offset, &offsets));
        return init(hdr.base, std::move(offsets));
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
    if (empty()) {
        return 0;
    }
    return bitset_payload_size(max_offset());
}

CompactIdsEncoding CompactIds::preferred_encoding() const {
    if (empty()) {
        return CompactIdsEncoding::Offsets32;
    }
    return bitset_storage_size_bytes() < offsets_storage_size_bytes()
        ? CompactIdsEncoding::Bitset
        : CompactIdsEncoding::Offsets32;
}

size_t CompactIds::serialized_size_bytes() const {
    const size_t header_size = sizeof(SerializedHeader);
    if (empty()) {
        return header_size;
    }
    if (preferred_encoding() == CompactIdsEncoding::Bitset) {
        return header_size + bitset_storage_size_bytes();
    }
    return header_size + offsets_storage_size_bytes();
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

Ret CompactIds::write_offsets(FILE* f, const std::string& error_message) const {
    const std::string base_message = error_message.empty() ? "CompactIds::write_offsets failed" : error_message;
    if (f == nullptr) {
        return Ret(base_message + ": file handle is null");
    }
    if (offsets_.empty()) {
        return Ret(0);
    }
    if (fwrite(offsets_.data(), sizeof(uint32_t), offsets_.size(), f) != offsets_.size()) {
        return Ret(base_message);
    }
    return Ret(0);
}

Ret CompactIds::write(FILE* f, const std::string& error_message) const {
    const std::string base_message = error_message.empty() ? "CompactIds::write failed" : error_message;
    if (f == nullptr) {
        return Ret(base_message + ": file handle is null");
    }

    if (count() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        return Ret(base_message + ": id count exceeds uint32_t range");
    }

    SerializedHeader hdr{};
    hdr.encoding = static_cast<uint8_t>(preferred_encoding());
    hdr.count = static_cast<uint32_t>(count());
    hdr.max_offset = empty() ? 0 : max_offset();
    hdr.payload_size = 0;
    hdr.base = empty() ? 0 : base_;

    if (preferred_encoding() == CompactIdsEncoding::Bitset) {
        if (bitset_storage_size_bytes() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            return Ret(base_message + ": bitset payload exceeds uint32_t range");
        }
        hdr.payload_size = static_cast<uint32_t>(bitset_storage_size_bytes());
    } else {
        if (offsets_storage_size_bytes() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            return Ret(base_message + ": offsets payload exceeds uint32_t range");
        }
        hdr.payload_size = static_cast<uint32_t>(offsets_storage_size_bytes());
    }

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        return Ret(base_message);
    }

    if (empty()) {
        return Ret(0);
    }

    if (preferred_encoding() == CompactIdsEncoding::Offsets32) {
        return write_offsets(f, base_message);
    }

    std::vector<uint8_t> bitset(bitset_storage_size_bytes(), 0);
    for (uint32_t value : offsets_) {
        const size_t byte_index = static_cast<size_t>(value) >> 3;
        const uint8_t mask = static_cast<uint8_t>(1u << (value & 7u));
        bitset[byte_index] |= mask;
    }
    if (fwrite(bitset.data(), 1, bitset.size(), f) != bitset.size()) {
        return Ret(base_message);
    }
    return Ret(0);
}

} // namespace sketch2
