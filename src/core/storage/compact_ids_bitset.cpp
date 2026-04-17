// Implements CompactIdsBitset, a sorted-id container backed by a bitset.

#include "compact_ids_bitset.h"
#include "compact_ids_shared.h"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace sketch2 {

namespace {

size_t bitset_payload_size(uint32_t max_offset) {
    return (static_cast<size_t>(max_offset) + 8u) / 8u;
}

bool test_bit(const uint8_t* bitset, uint32_t offset) {
    return (bitset[offset >> 3] & static_cast<uint8_t>(1u << (offset & 7u))) != 0;
}

size_t popcount_prefix(const uint8_t* bitset, uint32_t bit_count) {
    size_t count = 0;
    size_t byte_index = 0;
    const size_t full_bytes = static_cast<size_t>(bit_count) >> 3;

    for (; byte_index + sizeof(uint64_t) <= full_bytes; byte_index += sizeof(uint64_t)) {
        uint64_t word = 0;
        std::memcpy(&word, bitset + byte_index, sizeof(word));
        count += static_cast<size_t>(__builtin_popcountll(word));
    }

    for (; byte_index < full_bytes; ++byte_index) {
        count += static_cast<size_t>(__builtin_popcount(static_cast<unsigned int>(bitset[byte_index])));
    }

    const uint32_t tail_bits = bit_count & 7u;
    if (tail_bits != 0u) {
        const uint8_t mask = static_cast<uint8_t>((1u << tail_bits) - 1u);
        count += static_cast<size_t>(
            __builtin_popcount(static_cast<unsigned int>(bitset[full_bytes] & mask)));
    }

    return count;
}

} // namespace

CompactIdsBitset::Iterator::Iterator(const CompactIdsBitset* ids)
    : ids_(ids), index_(0), bit_pos_(ids == nullptr ? 0u : ids->next_set_bit(0u)) {}

void CompactIdsBitset::Iterator::next() {
    if (eof()) {
        return;
    }
    ++index_;
    bit_pos_ = ids_->next_set_bit(bit_pos_ + 1u);
}

bool CompactIdsBitset::Iterator::eof() const {
    return ids_ == nullptr || index_ >= ids_->count_;
}

uint64_t CompactIdsBitset::Iterator::id() const {
    if (eof()) {
        throw std::out_of_range("CompactIdsBitset::Iterator::id: index out of range");
    }
    return ids_->base_ + bit_pos_;
}

size_t CompactIdsBitset::Iterator::index() const {
    if (eof()) {
        throw std::out_of_range("CompactIdsBitset::Iterator::index: index out of range");
    }
    return index_;
}

Ret CompactIdsBitset::init(const CompactIdsAccumulator& accumulator) {
    try {
        return init_(accumulator);
    } catch (const std::exception& ex) {
        return Ret(std::string("CompactIdsBitset::init: ") + ex.what());
    }
}

Ret CompactIdsBitset::init_(const CompactIdsAccumulator& accumulator) {
    if (accumulator.size() == 0) {
        clear();
        return Ret(0);
    }

    if (accumulator.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        return Ret("CompactIdsBitset::init: id count exceeds uint32_t range");
    }

    const uint64_t new_base = accumulator[0];
    uint64_t prev = new_base;
    for (size_t i = 1; i < accumulator.size(); ++i) {
        const uint64_t current = accumulator[i];
        if (current <= prev) {
            return Ret("CompactIdsBitset::init: ids must be strictly increasing");
        }
        prev = current;
    }

    const uint64_t max_offset64 = accumulator[accumulator.size() - 1] - new_base;
    if (max_offset64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return Ret("CompactIdsBitset::init: id range exceeds uint32_t");
    }

    const uint32_t new_span = static_cast<uint32_t>(max_offset64) + 1u;
    std::vector<uint8_t> new_bitset(bitset_payload_size(static_cast<uint32_t>(max_offset64)), 0);
    for (size_t i = 0; i < accumulator.size(); ++i) {
        const uint32_t off = static_cast<uint32_t>(accumulator[i] - new_base);
        new_bitset[off >> 3] |= static_cast<uint8_t>(1u << (off & 7u));
    }

    base_ = new_base;
    count_ = static_cast<uint32_t>(accumulator.size());
    span_ = new_span;
    owned_bitset_ = std::move(new_bitset);
    bitset_size_ = static_cast<uint32_t>(owned_bitset_.size());
    bitset_ = owned_bitset_.data();
    return Ret(0);
}

Ret CompactIdsBitset::init(const uint64_t* ids, size_t size) {
    try {
        return init_(ids, size);
    } catch (const std::exception& ex) {
        return Ret(std::string("CompactIdsBitset::init: ") + ex.what());
    }
}

Ret CompactIdsBitset::init_(const uint64_t* ids, size_t size) {
    if (size == 0) {
        clear();
        return Ret(0);
    }
    if (ids == nullptr) {
        return Ret("CompactIdsBitset::init: ids pointer is null");
    }

    for (size_t i = 1; i < size; ++i) {
        if (ids[i] <= ids[i - 1]) {
            return Ret("CompactIdsBitset::init: ids must be strictly increasing");
        }
    }

    const uint64_t new_base = ids[0];
    const uint64_t max_offset64 = ids[size - 1] - new_base;
    if (max_offset64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return Ret("CompactIdsBitset::init: id range exceeds uint32_t");
    }
    if (size > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        return Ret("CompactIdsBitset::init: id count exceeds uint32_t range");
    }

    const uint32_t new_span = static_cast<uint32_t>(max_offset64) + 1u;
    std::vector<uint8_t> new_bitset(bitset_payload_size(static_cast<uint32_t>(max_offset64)), 0);
    for (size_t i = 0; i < size; ++i) {
        const uint64_t value = ids[i];
        const uint32_t off = static_cast<uint32_t>(value - new_base);
        new_bitset[off >> 3] |= static_cast<uint8_t>(1u << (off & 7u));
    }

    base_ = new_base;
    count_ = static_cast<uint32_t>(size);
    span_ = new_span;
    owned_bitset_ = std::move(new_bitset);
    bitset_size_ = static_cast<uint32_t>(owned_bitset_.size());
    bitset_ = owned_bitset_.data();
    return Ret(0);
}

void CompactIdsBitset::clear() {
    base_ = 0;
    count_ = 0;
    span_ = 0;
    bitset_ = nullptr;
    bitset_size_ = 0;
    owned_bitset_.clear();
}

size_t CompactIdsBitset::count() const {
    return count_;
}

bool CompactIdsBitset::empty() const {
    return count_ == 0;
}

size_t CompactIdsBitset::serialized_size_bytes() const {
    return sizeof(CompactIdsHeader) + bitset_size_;
}

uint64_t CompactIdsBitset::id(size_t index) const {
    if (index >= count_) {
        throw std::out_of_range("CompactIdsBitset::id: index out of range");
    }
    return base_ + select_bit(index);
}

uint64_t CompactIdsBitset::id_unchecked(size_t index) const {
    return base_ + select_bit(index);
}

size_t CompactIdsBitset::lower_bound_index(uint64_t value) const {
    if (empty() || value <= base_) {
        return 0;
    }

    const uint64_t off = value - base_;
    if (off >= span_) {
        return count_;
    }

    const uint32_t off32 = static_cast<uint32_t>(off);
    if (test_bit(bitset_, off32)) {
        return rank_bit(off32);
    }

    const uint32_t next = next_set_bit(off32 + 1u);
    if (next >= span_) {
        return count_;
    }
    return rank_bit(next);
}

Ret CompactIdsBitset::write(FILE* f, const std::string& error_message) const {
    const std::string base_message = error_message.empty()
        ? "CompactIdsBitset::write failed" : error_message;
    CHECK(write_compact_ids_header(
        f,
        CompactIdsExtEncoding::Bitset,
        count_,
        empty() ? 0u : span_ - 1u,
        bitset_size_,
        empty() ? 0 : base_,
        base_message));
    if (bitset_size_ > 0 && fwrite(bitset_, 1, bitset_size_, f) != bitset_size_) {
        return Ret(base_message);
    }

    return Ret(0);
}

Ret CompactIdsBitset::map(const uint8_t* data, size_t size, size_t* bytes_consumed) {
    try {
        return map_(data, size, bytes_consumed);
    } catch (const std::exception& ex) {
        return Ret(std::string("CompactIdsBitset::map: ") + ex.what());
    }
}

Ret CompactIdsBitset::map_(const uint8_t* data, size_t size, size_t* bytes_consumed) {
    CompactIdsMappedPrefix prefix{};
    CHECK(parse_compact_ids_mapped_prefix(
        data,
        size,
        CompactIdsExtEncoding::Bitset,
        "CompactIdsBitset::map",
        bytes_consumed,
        &prefix));
    const CompactIdsHeader& hdr = prefix.header;
    const size_t payload_size = prefix.payload_size;

    const uint32_t header_max_offset = hdr.aux_data;
    const size_t expected_payload = hdr.count == 0 ? 0 : bitset_payload_size(header_max_offset);
    if (payload_size != expected_payload) {
        return Ret("CompactIdsBitset::map: payload size does not match max_offset");
    }

    if (hdr.count == 0) {
        CHECK(validate_compact_ids_empty_header(
            hdr,
            "CompactIdsBitset::map: malformed empty header"));
        clear();
    } else {
        const uint32_t used_bits_in_last_byte = (header_max_offset + 1u) & 7u;
        if (used_bits_in_last_byte != 0u) {
            const uint8_t* payload = data + sizeof(CompactIdsHeader);
            const uint8_t padding_mask =
                static_cast<uint8_t>(~((1u << used_bits_in_last_byte) - 1u));
            if ((payload[payload_size - 1] & padding_mask) != 0u) {
                return Ret("CompactIdsBitset::map: malformed bitset payload tail bits");
            }
        }

        const size_t actual_count =
            popcount_prefix(data + sizeof(CompactIdsHeader), header_max_offset + 1u);
        if (actual_count != hdr.count) {
            return Ret("CompactIdsBitset::map: bitset count does not match header");
        }

        base_ = hdr.base;
        count_ = hdr.count;
        span_ = header_max_offset + 1u;
        owned_bitset_.clear();
        bitset_size_ = hdr.payload_size;
        bitset_ = bitset_size_ == 0 ? nullptr : data + sizeof(CompactIdsHeader);
    }

    if (bytes_consumed != nullptr) {
        *bytes_consumed = prefix.consumed;
    }
    return Ret(0);
}

CompactIdsBitset::Iterator CompactIdsBitset::begin() const {
    return Iterator(this);
}

uint32_t CompactIdsBitset::select_bit(size_t target) const {
    size_t seen = 0;
    size_t byte_index = 0;

    for (; byte_index + sizeof(uint64_t) <= bitset_size_; byte_index += sizeof(uint64_t)) {
        uint64_t word = 0;
        std::memcpy(&word, bitset_ + byte_index, sizeof(word));
        const size_t bits_in_word = static_cast<size_t>(__builtin_popcountll(word));
        if (target < seen + bits_in_word) {
            size_t local_target = target - seen;
            while (word != 0) {
                const uint32_t bit_index = static_cast<uint32_t>(__builtin_ctzll(word));
                if (local_target == 0) {
                    return static_cast<uint32_t>(byte_index * 8u + bit_index);
                }
                word &= (word - 1u);
                --local_target;
            }
        }
        seen += bits_in_word;
    }

    for (; byte_index < bitset_size_; ++byte_index) {
        uint8_t byte = bitset_[byte_index];
        const size_t bits_in_byte = static_cast<size_t>(
            __builtin_popcount(static_cast<unsigned int>(byte)));
        if (target < seen + bits_in_byte) {
            size_t local_target = target - seen;
            while (byte != 0) {
                const uint32_t bit_index = static_cast<uint32_t>(
                    __builtin_ctz(static_cast<unsigned int>(byte)));
                if (local_target == 0) {
                    return static_cast<uint32_t>(byte_index * 8u + bit_index);
                }
                byte = static_cast<uint8_t>(byte & static_cast<uint8_t>(byte - 1u));
                --local_target;
            }
        }
        seen += bits_in_byte;
    }

    throw std::out_of_range("CompactIdsBitset::select_bit: index out of range");
}

size_t CompactIdsBitset::rank_bit(uint32_t offset) const {
    return popcount_prefix(bitset_, offset);
}

uint32_t CompactIdsBitset::next_set_bit(uint32_t offset) const {
    if (offset >= span_) {
        return span_;
    }

    size_t byte_index = static_cast<size_t>(offset) >> 3;
    uint8_t byte = bitset_[byte_index];
    byte &= static_cast<uint8_t>(0xFFu << (offset & 7u));
    if (byte != 0) {
        return static_cast<uint32_t>(
            byte_index * 8u + __builtin_ctz(static_cast<unsigned int>(byte)));
    }

    ++byte_index;
    for (; byte_index < bitset_size_; ++byte_index) {
        if (bitset_[byte_index] != 0) {
            return static_cast<uint32_t>(
                byte_index * 8u + __builtin_ctz(static_cast<unsigned int>(bitset_[byte_index])));
        }
    }

    return span_;
}

} // namespace sketch2
