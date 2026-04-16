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

bool is_misses_encoding(uint8_t encoding) {
    return encoding == static_cast<uint8_t>(CompactIdsExtEncoding::Misses32);
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
    } catch (const std::exception& ex) {
        return Ret(std::string("CompactIdsMisses::init: ") + ex.what());
    }
}

Ret CompactIdsMisses::init(const std::vector<uint64_t>& ids) {
    try {
        if (ids.empty()) {
            clear();
            return Ret(0);
        }

        for (size_t i = 1; i < ids.size(); ++i) {
            if (ids[i] <= ids[i - 1]) {
                return Ret("CompactIdsMisses::init: ids must be strictly increasing");
            }
        }

        const uint64_t new_base = ids[0];
        const uint64_t span = ids.back() - ids[0] + 1;

        if (span - 1 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            return Ret("CompactIdsMisses::init: id range exceeds uint32_t");
        }

        if (ids.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            return Ret("CompactIdsMisses::init: id count exceeds uint32_t range");
        }

        std::vector<uint32_t> new_misses;
        new_misses.reserve(static_cast<size_t>(span) - ids.size());

        for (size_t i = 1; i < ids.size(); ++i) {
            const uint32_t prev_off = static_cast<uint32_t>(ids[i - 1] - new_base);
            const uint32_t curr_off = static_cast<uint32_t>(ids[i] - new_base);
            for (uint32_t off = prev_off + 1; off < curr_off; ++off) {
                new_misses.push_back(off);
            }
        }

        base_ = new_base;
        count_ = static_cast<uint32_t>(ids.size());
        owned_misses_ = std::move(new_misses);
        miss_count_ = static_cast<uint32_t>(owned_misses_.size());
        misses_ = owned_misses_.data();
        return Ret(0);
    } catch (const std::exception& ex) {
        return Ret(std::string("CompactIdsMisses::init: ") + ex.what());
    }
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

uint64_t CompactIdsMisses::base() const {
    return base_;
}

uint64_t CompactIdsMisses::min_id() const {
    if (empty()) {
        throw std::out_of_range("CompactIdsMisses::min_id: container is empty");
    }
    return base_;
}

uint64_t CompactIdsMisses::max_id() const {
    if (empty()) {
        throw std::out_of_range("CompactIdsMisses::max_id: container is empty");
    }
    return base_ + count_ - 1 + miss_count_;
}

uint32_t CompactIdsMisses::offset(size_t index) const {
    if (index >= count_) {
        throw std::out_of_range("CompactIdsMisses::offset: index out of range");
    }
    if (miss_count_ == 0) {
        return static_cast<uint32_t>(index);
    }
    size_t lo = 0;
    size_t hi = miss_count_;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (static_cast<uint64_t>(misses_[mid]) - mid > index) hi = mid;
        else lo = mid + 1;
    }
    return static_cast<uint32_t>(index + lo);
}

uint32_t CompactIdsMisses::max_offset() const {
    if (empty()) {
        throw std::out_of_range("CompactIdsMisses::max_offset: container is empty");
    }
    return static_cast<uint32_t>(count_ - 1 + miss_count_);
}

size_t CompactIdsMisses::offsets_storage_size_bytes() const {
    return static_cast<size_t>(miss_count_) * sizeof(uint32_t);
}

size_t CompactIdsMisses::serialized_size_bytes() const {
    if (empty()) {
        return sizeof(CompactIdsHeader);
    }
    return sizeof(CompactIdsHeader) + offsets_storage_size_bytes();
}

uint64_t CompactIdsMisses::id(size_t index) const {
    if (index >= count_) {
        throw std::out_of_range("CompactIdsMisses::id: index out of range");
    }
    if (miss_count_ == 0) {
        return base_ + index;
    }
    return base_ + offset(index);
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

size_t CompactIdsMisses::index_of(uint64_t value) const {
    if (empty() || value < base_) {
        return npos;
    }

    const uint64_t off = value - base_;
    const uint64_t total_span = static_cast<uint64_t>(count_) + miss_count_;
    if (off >= total_span) {
        return npos;
    }

    if (miss_count_ == 0) {
        return static_cast<size_t>(off);
    }

    const uint32_t off32 = static_cast<uint32_t>(off);
    const uint32_t* end = misses_ + miss_count_;
    const uint32_t* it = std::lower_bound(misses_, end, off32);
    if (it != end && *it == off32) {
        return npos;
    }

    const size_t m = static_cast<size_t>(it - misses_);
    return static_cast<size_t>(off) - m;
}

bool CompactIdsMisses::contains(uint64_t value) const {
    if (empty() || value < base_) {
        return false;
    }

    const uint64_t off = value - base_;
    const uint64_t total_span = static_cast<uint64_t>(count_) + miss_count_;
    if (off >= total_span) {
        return false;
    }

    if (miss_count_ == 0) {
        return true;
    }

    return !std::binary_search(misses_, misses_ + miss_count_, static_cast<uint32_t>(off));
}

Ret CompactIdsMisses::write(FILE* f, const std::string& error_message) const {
    const std::string base_message = error_message.empty()
        ? "CompactIdsMisses::write failed" : error_message;
    if (f == nullptr) {
        return Ret(base_message + ": file handle is null");
    }

    CompactIdsHeader hdr{};
    hdr.encoding = static_cast<uint8_t>(CompactIdsExtEncoding::Misses32);
    hdr.count = count_;
    hdr.miss_count = miss_count_;
    hdr.payload_size = static_cast<uint32_t>(offsets_storage_size_bytes());
    hdr.base = empty() ? 0 : base_;

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        return Ret(base_message);
    }

    if (miss_count_ > 0) {
        if (fwrite(misses_, sizeof(uint32_t), miss_count_, f) != miss_count_) {
            return Ret(base_message);
        }
    }

    return Ret(0);
}

Ret CompactIdsMisses::map(const uint8_t* data, size_t size, size_t* bytes_consumed) {
    try {
        if (bytes_consumed != nullptr) {
            *bytes_consumed = 0;
        }
        if (data == nullptr) {
            return Ret("CompactIdsMisses::map: data pointer is null");
        }
        if (size < sizeof(CompactIdsHeader)) {
            return Ret("CompactIdsMisses::map: buffer too small to contain header");
        }

        CompactIdsHeader hdr{};
        std::memcpy(&hdr, data, sizeof(CompactIdsHeader));
        if (!is_misses_encoding(hdr.encoding)) {
            return Ret("CompactIdsMisses::map: unexpected encoding");
        }

        const size_t payload_size = static_cast<size_t>(hdr.payload_size);
        if (payload_size > size - sizeof(CompactIdsHeader)) {
            return Ret("CompactIdsMisses::map: truncated payload");
        }

        const size_t expected_payload = static_cast<size_t>(hdr.miss_count) * sizeof(uint32_t);
        if (payload_size != expected_payload) {
            return Ret("CompactIdsMisses::map: payload size does not match miss_count");
        }

        const size_t consumed = sizeof(CompactIdsHeader) + payload_size;

        if (hdr.count == 0) {
            if (hdr.miss_count != 0 || hdr.payload_size != 0 || hdr.base != 0) {
                return Ret("CompactIdsMisses::map: malformed empty header");
            }
            clear();
        } else {
            base_ = hdr.base;
            count_ = hdr.count;
            miss_count_ = hdr.miss_count;
            owned_misses_.clear();
            misses_ = miss_count_ == 0 ? nullptr :
                reinterpret_cast<const uint32_t*>(data + sizeof(CompactIdsHeader));
        }

        if (bytes_consumed != nullptr) {
            *bytes_consumed = consumed;
        }
        return Ret(0);
    } catch (const std::exception& ex) {
        return Ret(std::string("CompactIdsMisses::map: ") + ex.what());
    }
}

CompactIdsMisses::Iterator CompactIdsMisses::begin() const {
    return Iterator(this);
}

} // namespace sketch2
