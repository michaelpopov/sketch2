// Implements CompactIdsOffsets, a sorted-id container backed by 32-bit offsets.

#include "compact_ids_offsets.h"
#include "compact_ids_shared.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace sketch2 {

namespace {

static_assert(sizeof(CompactIdsHeader) == 24, "CompactIdsHeader must stay compact");

Ret validate_offsets(uint64_t base, const uint32_t* offsets, size_t count);
Ret validate_offsets(uint64_t base, const uint32_t* offsets, size_t count) {
    uint32_t prev = 0;
    for (size_t i = 0; i < count; ++i) {
        const uint32_t current = offsets[i];
        if (i > 0 && prev >= current) {
            return Ret("CompactIdsOffsets::init: offsets must be strictly increasing");
        }
        if (base > std::numeric_limits<uint64_t>::max() - static_cast<uint64_t>(current)) {
            return Ret("CompactIdsOffsets::init: base plus offset overflows uint64_t");
        }
        prev = current;
    }

    return Ret(0);
}

} // namespace

void CompactIdsOffsets::Iterator::next() {
    if (eof()) {
        return;
    }
    ++index_;
}

bool CompactIdsOffsets::Iterator::eof() const {
    return ids_ == nullptr || index_ >= ids_->count();
}

uint64_t CompactIdsOffsets::Iterator::id() const {
    if (eof()) {
        throw std::out_of_range("CompactIdsOffsets::Iterator::id: index out of range");
    }
    return ids_->id(index_);
}

size_t CompactIdsOffsets::Iterator::index() const {
    if (eof()) {
        throw std::out_of_range("CompactIdsOffsets::Iterator::index: index out of range");
    }
    return index_;
}

Ret CompactIdsOffsets::init(const CompactIdsAccumulator& accumulator) {
    try {
        if (accumulator.size() == 0) {
            clear();
            return Ret(0);
        }
        if (accumulator.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            return Ret("CompactIdsOffsets::init: id count exceeds uint32_t range");
        }

        const uint64_t new_base = accumulator[0];
        std::vector<uint32_t> new_offsets;
        new_offsets.reserve(accumulator.size());
        uint64_t prev = new_base;
        for (size_t i = 0; i < accumulator.size(); ++i) {
            const uint64_t current = accumulator[i];
            if (i > 0 && prev >= current) {
                return Ret("CompactIdsOffsets::init: ids must be strictly increasing");
            }

            const uint64_t offset = current - new_base;
            if (offset > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
                return Ret("CompactIdsOffsets::init: id offset exceeds uint32_t range");
            }

            new_offsets.push_back(static_cast<uint32_t>(offset));
            prev = current;
        }

        base_ = new_base;
        owned_offsets_ = std::move(new_offsets);
        offsets_ = owned_offsets_.data();
        count_ = static_cast<uint32_t>(owned_offsets_.size());
        return Ret(0);
    } catch (const std::exception& ex) {
        return Ret(std::string("CompactIdsOffsets::init: ") + ex.what());
    }
}

Ret CompactIdsOffsets::init(const std::vector<uint64_t>& ids) {
    try {
        if (ids.empty()) {
            clear();
            return Ret(0);
        }

        if (ids.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            return Ret("CompactIdsOffsets::init: id count exceeds uint32_t range");
        }

        const uint64_t new_base = ids[0];
        std::vector<uint32_t> new_offsets;
        new_offsets.reserve(ids.size());
        uint64_t prev = new_base;
        for (size_t i = 0; i < ids.size(); ++i) {
            const uint64_t current = ids[i];
            if (i > 0 && prev >= current) {
                return Ret("CompactIdsOffsets::init: ids must be strictly increasing");
            }

            const uint64_t offset = current - new_base;
            if (offset > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
                return Ret("CompactIdsOffsets::init: id offset exceeds uint32_t range");
            }

            new_offsets.push_back(static_cast<uint32_t>(offset));
            prev = current;
        }
        base_ = new_base;
        owned_offsets_ = std::move(new_offsets);
        offsets_ = owned_offsets_.data();
        count_ = static_cast<uint32_t>(owned_offsets_.size());
        return Ret(0);
    } catch (const std::exception& ex) {
        return Ret(std::string("CompactIdsOffsets::init: ") + ex.what());
    }
}

Ret CompactIdsOffsets::map(const uint8_t* data, size_t size, size_t* bytes_consumed) {
    try {
        if (bytes_consumed != nullptr) {
            *bytes_consumed = 0;
        }
        if (data == nullptr) {
            return Ret("CompactIdsOffsets::map: data pointer is null");
        }
        if (size < sizeof(CompactIdsHeader)) {
            return Ret("CompactIdsOffsets::map: buffer too small to contain header");
        }

        CompactIdsHeader hdr{};
        std::memcpy(&hdr, data, sizeof(CompactIdsHeader));
        if (hdr.encoding != static_cast<uint8_t>(CompactIdsExtEncoding::Offsets32)) {
            return Ret("CompactIdsOffsets::map: unexpected encoding");
        }

        const size_t payload_size = static_cast<size_t>(hdr.payload_size);
        if (payload_size > size - sizeof(CompactIdsHeader)) {
            return Ret("CompactIdsOffsets::map: truncated payload");
        }

        const size_t expected_payload_size = static_cast<size_t>(hdr.count) * sizeof(uint32_t);
        if (payload_size != expected_payload_size) {
            return Ret("CompactIdsOffsets::map: malformed offsets payload size");
        }

        const size_t consumed = sizeof(CompactIdsHeader) + payload_size;
        if (hdr.count == 0) {
            if (hdr.miss_count != 0 || hdr.payload_size != 0 || hdr.base != 0) {
                return Ret("CompactIdsOffsets::map: malformed empty payload header");
            }
            clear();
        } else {
            const uint32_t* mapped_offsets =
                reinterpret_cast<const uint32_t*>(data + sizeof(CompactIdsHeader));
            CHECK(validate_offsets(hdr.base, mapped_offsets, hdr.count));
            if (mapped_offsets[hdr.count - 1] != hdr.miss_count) {
                return Ret("CompactIdsOffsets::map: max_offset does not match payload");
            }
            base_ = hdr.base;
            count_ = hdr.count;
            offsets_ = mapped_offsets;
            owned_offsets_.clear();
        }

        if (bytes_consumed != nullptr) {
            *bytes_consumed = consumed;
        }
        return Ret(0);
    } catch (const std::exception& ex) {
        return Ret(std::string("CompactIdsOffsets::map: ") + ex.what());
    }
}

void CompactIdsOffsets::clear() {
    base_ = 0;
    count_ = 0;
    offsets_ = nullptr;
    owned_offsets_.clear();
}

uint64_t CompactIdsOffsets::min_id() const {
    if (empty()) {
        throw std::out_of_range("CompactIdsOffsets::min_id: container is empty");
    }
    return base_ + offsets_[0];
}

uint64_t CompactIdsOffsets::max_id() const {
    if (empty()) {
        throw std::out_of_range("CompactIdsOffsets::max_id: container is empty");
    }
    return base_ + offsets_[count_ - 1];
}

uint32_t CompactIdsOffsets::offset(size_t index) const {
    if (index >= count()) {
        throw std::out_of_range("CompactIdsOffsets::offset: index out of range");
    }
    return offsets_[index];
}

uint32_t CompactIdsOffsets::max_offset() const {
    if (empty()) {
        throw std::out_of_range("CompactIdsOffsets::max_offset: container is empty");
    }
    return offsets_[count_ - 1];
}

size_t CompactIdsOffsets::serialized_size_bytes() const {
    return sizeof(CompactIdsHeader) + offsets_storage_size_bytes();
}

uint64_t CompactIdsOffsets::id(size_t index) const {
    if (index >= count()) {
        throw std::out_of_range("CompactIdsOffsets::id: index out of range");
    }
    return base_ + offsets_[index];
}

size_t CompactIdsOffsets::lower_bound_index(uint64_t value) const {
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
        offsets_, offsets_ + count_, static_cast<uint32_t>(relative));
    return static_cast<size_t>(it - offsets_);
}

size_t CompactIdsOffsets::index_of(uint64_t value) const {
    const size_t index = lower_bound_index(value);
    if (index >= count()) {
        return npos;
    }
    return id(index) == value ? index : npos;
}

bool CompactIdsOffsets::contains(uint64_t value) const {
    return index_of(value) != npos;
}

Ret CompactIdsOffsets::write(FILE* f, const std::string& error_message) const {
    const std::string base_message = error_message.empty() ? "CompactIdsOffsets::write failed" : error_message;
    if (f == nullptr) {
        return Ret(base_message + ": file handle is null");
    }

    CompactIdsHeader hdr{};
    hdr.encoding = static_cast<uint8_t>(CompactIdsExtEncoding::Offsets32);
    hdr.count = count_;
    hdr.miss_count = empty() ? 0 : max_offset();
    hdr.payload_size = static_cast<uint32_t>(offsets_storage_size_bytes());
    hdr.base = empty() ? 0 : base_;

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        return Ret(base_message);
    }
    if (empty()) {
        return Ret(0);
    }

    const uint32_t* output_offsets = owned_offsets_.empty() ? offsets_ : owned_offsets_.data();
    if (fwrite(output_offsets, sizeof(uint32_t), count_, f) != count_) {
        return Ret(base_message);
    }
    return Ret(0);
}

} // namespace sketch2
