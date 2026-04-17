// Implements CompactIdsMisses, a sorted-id container backed by a miss-list.

#include "compact_ids_misses.h"
#include "compact_ids_shared.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace sketch2 {

namespace {

static_assert(sizeof(CompactIdsHeader) == 24, "CompactIdsHeader must stay compact");

Ret validate_misses(uint64_t base, const uint32_t* misses, uint32_t count, uint32_t miss_count) {
    const uint64_t total_span = static_cast<uint64_t>(count) + static_cast<uint64_t>(miss_count);
    if (total_span == 0) {
        return Ret(0);
    }
    if (total_span - 1u > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return Ret("CompactIdsMisses::map: id range exceeds uint32_t");
    }
    if (base > std::numeric_limits<uint64_t>::max() - (total_span - 1u)) {
        return Ret("CompactIdsMisses::map: base plus id span overflows uint64_t");
    }

    uint32_t prev = 0;
    for (uint32_t i = 0; i < miss_count; ++i) {
        const uint32_t current = misses[i];
        if (i > 0 && prev >= current) {
            return Ret("CompactIdsMisses::map: misses must be strictly increasing");
        }
        if (current == 0u || current + 1u >= total_span) {
            return Ret("CompactIdsMisses::map: misses must stay within the interior span");
        }
        prev = current;
    }

    return Ret(0);
}

} // namespace

void CompactIdsMisses::Iterator::next() {
    if (eof()) {
        return;
    }
    ++index_;
    ++current_;
    while (miss_idx_ < ids_->miss_count_ &&
           current_ - ids_->base_ == ids_->misses_[miss_idx_]) {
        ++current_;
        ++miss_idx_;
    }
}

bool CompactIdsMisses::Iterator::eof() const {
    return ids_ == nullptr || index_ >= ids_->count_;
}

uint64_t CompactIdsMisses::Iterator::id() const {
    if (eof()) {
        throw std::out_of_range("CompactIdsMisses::Iterator::id: index out of range");
    }
    return current_;
}

size_t CompactIdsMisses::Iterator::index() const {
    if (eof()) {
        throw std::out_of_range("CompactIdsMisses::Iterator::index: index out of range");
    }
    return index_;
}

Ret CompactIdsMisses::init(const CompactIdsAccumulator& accumulator) {
    try {
        return init_(accumulator);
    } catch (const std::exception& ex) {
        return Ret(std::string("CompactIdsMisses::init: ") + ex.what());
    }
}

Ret CompactIdsMisses::init_(const CompactIdsAccumulator& accumulator) {
    if (accumulator.size() == 0) {
        clear();
        return Ret(0);
    }

    if (accumulator.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        return Ret("CompactIdsMisses::init: id count exceeds uint32_t range");
    }

    const uint64_t new_base = accumulator[0];
    uint64_t prev = new_base;
    std::vector<uint32_t> new_misses;
    for (size_t i = 1; i < accumulator.size(); ++i) {
        const uint64_t current = accumulator[i];
        if (current <= prev) {
            return Ret("CompactIdsMisses::init: ids must be strictly increasing");
        }

        const uint64_t prev_off64 = prev - new_base;
        const uint64_t curr_off64 = current - new_base;
        if (curr_off64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            return Ret("CompactIdsMisses::init: id range exceeds uint32_t");
        }

        const uint32_t prev_off = static_cast<uint32_t>(prev_off64);
        const uint32_t curr_off = static_cast<uint32_t>(curr_off64);
        for (uint32_t off = prev_off + 1; off < curr_off; ++off) {
            new_misses.push_back(off);
        }
        prev = current;
    }

    base_ = new_base;
    count_ = static_cast<uint32_t>(accumulator.size());
    owned_misses_ = std::move(new_misses);
    miss_count_ = static_cast<uint32_t>(owned_misses_.size());
    misses_ = owned_misses_.data();
    return Ret(0);
}

Ret CompactIdsMisses::init(const uint64_t* ids, size_t size) {
    try {
        return init_(ids, size);
    } catch (const std::exception& ex) {
        return Ret(std::string("CompactIdsMisses::init: ") + ex.what());
    }
}

Ret CompactIdsMisses::init_(const uint64_t* ids, size_t size) {
    if (size == 0) {
        clear();
        return Ret(0);
    }
    if (ids == nullptr) {
        return Ret("CompactIdsMisses::init: ids pointer is null");
    }

    for (size_t i = 1; i < size; ++i) {
        if (ids[i] <= ids[i - 1]) {
            return Ret("CompactIdsMisses::init: ids must be strictly increasing");
        }
    }

    const uint64_t new_base = ids[0];
    const uint64_t span = ids[size - 1] - ids[0] + 1;

    if (span - 1 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return Ret("CompactIdsMisses::init: id range exceeds uint32_t");
    }

    if (size > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        return Ret("CompactIdsMisses::init: id count exceeds uint32_t range");
    }

    std::vector<uint32_t> new_misses;
    new_misses.reserve(static_cast<size_t>(span) - size);

    for (size_t i = 1; i < size; ++i) {
        const uint32_t prev_off = static_cast<uint32_t>(ids[i - 1] - new_base);
        const uint32_t curr_off = static_cast<uint32_t>(ids[i] - new_base);
        for (uint32_t off = prev_off + 1; off < curr_off; ++off) {
            new_misses.push_back(off);
        }
    }

    base_ = new_base;
    count_ = static_cast<uint32_t>(size);
    owned_misses_ = std::move(new_misses);
    miss_count_ = static_cast<uint32_t>(owned_misses_.size());
    misses_ = owned_misses_.data();
    return Ret(0);
}

void CompactIdsMisses::clear() {
    base_ = 0;
    count_ = 0;
    misses_ = nullptr;
    miss_count_ = 0;
    owned_misses_.clear();
}

size_t CompactIdsMisses::count() const {
    return count_;
}

bool CompactIdsMisses::empty() const {
    return count_ == 0;
}

size_t CompactIdsMisses::serialized_size_bytes() const {
    if (empty()) {
        return sizeof(CompactIdsHeader);
    }
    return sizeof(CompactIdsHeader) + static_cast<size_t>(miss_count_) * sizeof(uint32_t);
}

uint64_t CompactIdsMisses::id(size_t index) const {
    if (index >= count_) {
        throw std::out_of_range("CompactIdsMisses::id: index out of range");
    }
    if (miss_count_ == 0) {
        return base_ + index;
    }
    size_t lo = 0;
    size_t hi = miss_count_;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (static_cast<uint64_t>(misses_[mid]) - mid > index) hi = mid;
        else lo = mid + 1;
    }
    return base_ + index + lo;
}

uint64_t CompactIdsMisses::id_unchecked(size_t index) const {
    if (miss_count_ == 0) {
        return base_ + index;
    }
    size_t lo = 0;
    size_t hi = miss_count_;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (static_cast<uint64_t>(misses_[mid]) - mid > index) hi = mid;
        else lo = mid + 1;
    }
    return base_ + index + lo;
}

size_t CompactIdsMisses::lower_bound_index(uint64_t value) const {
    if (empty() || value <= base_) {
        return 0;
    }

    const uint64_t off = value - base_;
    const uint64_t total_span = static_cast<uint64_t>(count_) + miss_count_;
    if (off >= total_span) {
        return count_;
    }

    if (miss_count_ == 0) {
        return static_cast<size_t>(off);
    }

    const uint32_t off32 = static_cast<uint32_t>(off);
    const uint32_t* end = misses_ + miss_count_;
    const uint32_t* it = std::lower_bound(misses_, end, off32);
    const size_t m = static_cast<size_t>(it - misses_);

    if (it == end || *it != off32) {
        return static_cast<size_t>(off) - m;
    }

    size_t j = m;
    while (j < miss_count_ && misses_[j] == off32 + static_cast<uint32_t>(j - m)) {
        ++j;
    }
    const uint64_t next_present = static_cast<uint64_t>(misses_[j - 1]) + 1;
    if (next_present >= total_span) {
        return count_;
    }
    return static_cast<size_t>(next_present) - j;
}

Ret CompactIdsMisses::write(FILE* f, const std::string& error_message) const {
    const std::string base_message = error_message.empty()
        ? "CompactIdsMisses::write failed" : error_message;
    CHECK(write_compact_ids_header(
        f,
        CompactIdsExtEncoding::Misses32,
        count_,
        miss_count_,
        static_cast<uint32_t>(static_cast<size_t>(miss_count_) * sizeof(uint32_t)),
        empty() ? 0 : base_,
        base_message));

    if (miss_count_ > 0) {
        if (fwrite(misses_, sizeof(uint32_t), miss_count_, f) != miss_count_) {
            return Ret(base_message);
        }
    }

    return Ret(0);
}

Ret CompactIdsMisses::map(const uint8_t* data, size_t size, size_t* bytes_consumed) {
    try {
        return map_(data, size, bytes_consumed);
    } catch (const std::exception& ex) {
        return Ret(std::string("CompactIdsMisses::map: ") + ex.what());
    }
}

Ret CompactIdsMisses::map_(const uint8_t* data, size_t size, size_t* bytes_consumed) {
    CompactIdsMappedPrefix prefix{};
    CHECK(parse_compact_ids_mapped_prefix(
        data,
        size,
        CompactIdsExtEncoding::Misses32,
        "CompactIdsMisses::map",
        bytes_consumed,
        &prefix));
    const CompactIdsHeader& hdr = prefix.header;
    const size_t payload_size = prefix.payload_size;

    const uint32_t header_miss_count = hdr.aux_data;
    const size_t expected_payload = static_cast<size_t>(header_miss_count) * sizeof(uint32_t);
    if (payload_size != expected_payload) {
        return Ret("CompactIdsMisses::map: payload size does not match miss_count");
    }

    if (hdr.count == 0) {
        CHECK(validate_compact_ids_empty_header(
            hdr,
            "CompactIdsMisses::map: malformed empty header"));
        clear();
    } else {
        const uint8_t* payload = data + sizeof(CompactIdsHeader);
        const uint32_t* mapped_misses = nullptr;
        if (header_miss_count == 0) {
            mapped_misses = nullptr;
        } else {
            mapped_misses = reinterpret_cast<const uint32_t*>(payload);
        }
        CHECK(validate_misses(hdr.base, mapped_misses, hdr.count, header_miss_count));
        base_ = hdr.base;
        count_ = hdr.count;
        miss_count_ = header_miss_count;
        misses_ = mapped_misses;
        owned_misses_.clear();
    }

    if (bytes_consumed != nullptr) {
        *bytes_consumed = prefix.consumed;
    }
    return Ret(0);
}

CompactIdsMisses::Iterator CompactIdsMisses::begin() const {
    return Iterator(this);
}

} // namespace sketch2
