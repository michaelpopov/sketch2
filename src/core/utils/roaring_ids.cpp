// Implements RoaringIds, a uint32_t id container backed by CRoaring.

#include "roaring_ids.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace sketch2 {

namespace {

bool is_aligned_32(const void* ptr) {
    return reinterpret_cast<uintptr_t>(ptr) % 32 == 0;
}

} // namespace

void RoaringIds::BitmapDeleter::operator()(roaring::api::roaring_bitmap_t* bitmap) const {
    roaring::api::roaring_bitmap_free(bitmap);
}

roaring::api::roaring_bitmap_t* RoaringIds::bitmap() {
    return bitmap_.get();
}

const roaring::api::roaring_bitmap_t* RoaringIds::bitmap() const {
    return bitmap_.get();
}

RoaringIds::Iterator::Iterator(const RoaringIds* roaring_ids)
    : roaring_ids_(roaring_ids) {
    if (roaring_ids_ != nullptr && roaring_ids_->bitmap() != nullptr) {
        roaring::api::roaring_iterator_init(roaring_ids_->bitmap(), &iterator_);
    }
}

void RoaringIds::Iterator::next() {
    if (eof()) {
        return;
    }
    roaring::api::roaring_uint32_iterator_advance(&iterator_);
    ++index_;
}

bool RoaringIds::Iterator::eof() const {
    return roaring_ids_ == nullptr || !iterator_.has_value;
}

uint64_t RoaringIds::Iterator::id() const {
    if (eof()) {
        throw std::out_of_range("RoaringIds::Iterator::id: index out of range");
    }
    if (roaring_ids_->base_ >
        std::numeric_limits<uint64_t>::max() - iterator_.current_value) {
        throw std::overflow_error(
            "RoaringIds::Iterator::id: base plus id offset overflows uint64_t");
    }
    return roaring_ids_->base_ + iterator_.current_value;
}

size_t RoaringIds::Iterator::index() const {
    if (eof()) {
        throw std::out_of_range("RoaringIds::Iterator::index: index out of range");
    }
    return index_;
}

uint64_t RoaringIds::Iterator::operator*() const {
    return id();
}

RoaringIds::Iterator& RoaringIds::Iterator::operator++() {
    next();
    return *this;
}

bool RoaringIds::Iterator::operator==(const Iterator& other) const {
    if (eof() && other.eof()) {
        return true;
    }
    return roaring_ids_ == other.roaring_ids_ &&
        iterator_.has_value == other.iterator_.has_value &&
        iterator_.current_value == other.iterator_.current_value;
}

bool RoaringIds::Iterator::operator!=(const Iterator& other) const {
    return !(*this == other);
}

Ret RoaringIds::init_writable(uint64_t base) {
    BitmapPtr new_bitmap(roaring::api::roaring_bitmap_create());
    if (!new_bitmap) {
        return Ret("RoaringIds::init_writable: failed to allocate bitmap");
    }

    bitmap_ = std::move(new_bitmap);
    read_only_ = false;
    base_ = base;
    return Ret(0);
}

Ret RoaringIds::init_frozen_view(const uint8_t* data, size_t size, uint64_t base) {
    if (data == nullptr) {
        return Ret("RoaringIds::init_frozen_view: data pointer is null");
    }
    if (!is_aligned_32(data)) {
        return Ret("RoaringIds::init_frozen_view: frozen view buffer must be 32-byte aligned");
    }

    const roaring::api::roaring_bitmap_t* view =
        roaring::api::roaring_bitmap_frozen_view(reinterpret_cast<const char*>(data), size);
    if (view == nullptr) {
        return Ret("RoaringIds::init_frozen_view: invalid frozen view");
    }
    if (roaring::api::roaring_bitmap_get_cardinality(view) > 0 &&
        base > std::numeric_limits<uint64_t>::max() -
            roaring::api::roaring_bitmap_maximum(view)) {
        roaring::api::roaring_bitmap_free(view);
        return Ret("RoaringIds::init_frozen_view: base plus id offset overflows uint64_t");
    }

    BitmapPtr new_bitmap(const_cast<roaring::api::roaring_bitmap_t*>(view));
    bitmap_ = std::move(new_bitmap);
    read_only_ = true;
    base_ = base;
    return Ret(0);
}

Ret RoaringIds::add(uint64_t id) {
    if (bitmap() == nullptr) {
        return Ret("RoaringIds::add: bitmap is not initialized");
    }
    if (read_only_) {
        return Ret("RoaringIds::add: bitmap is read-only");
    }
    if (id < base_) {
        return Ret("RoaringIds::add: id is below base");
    }

    const uint64_t offset = id - base_;
    if (offset > std::numeric_limits<uint32_t>::max()) {
        return Ret("RoaringIds::add: id offset exceeds uint32_t range");
    }

    roaring::api::roaring_bitmap_add(bitmap(), static_cast<uint32_t>(offset));
    return Ret(0);
}

void RoaringIds::clear() {
    bitmap_.reset();
    read_only_ = false;
    base_ = 0;
}

size_t RoaringIds::count() const {
    if (bitmap() == nullptr) {
        return 0;
    }
    return static_cast<size_t>(roaring::api::roaring_bitmap_get_cardinality(bitmap()));
}

bool RoaringIds::empty() const {
    return count() == 0;
}

bool RoaringIds::contains(uint64_t id) const {
    if (bitmap() == nullptr || id < base_) {
        return false;
    }

    const uint64_t offset = id - base_;
    if (offset > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    return roaring::api::roaring_bitmap_contains(bitmap(), static_cast<uint32_t>(offset));
}

void RoaringIds::compact() {
    if (bitmap() == nullptr || read_only_) {
        return;
    }
    roaring::api::roaring_bitmap_run_optimize(bitmap());
    roaring::api::roaring_bitmap_shrink_to_fit(bitmap());
}

size_t RoaringIds::serialized_size_bytes() const {
    if (bitmap() == nullptr) {
        return 0;
    }
    return roaring::api::roaring_bitmap_frozen_size_in_bytes(bitmap());
}

uint64_t RoaringIds::id(size_t index) const {
    if (index >= count()) {
        throw std::out_of_range("RoaringIds::id: index out of range");
    }

    return id_unchecked(index);
}

uint64_t RoaringIds::id_unchecked(size_t index) const {
    uint32_t value = 0;
    if (!roaring::api::roaring_bitmap_select(bitmap(), static_cast<uint32_t>(index), &value)) {
        throw std::out_of_range("RoaringIds::id_unchecked: index out of range");
    }
    if (base_ > std::numeric_limits<uint64_t>::max() - value) {
        throw std::overflow_error(
            "RoaringIds::id_unchecked: base plus id offset overflows uint64_t");
    }
    return base_ + value;
}

size_t RoaringIds::lower_bound_index(uint64_t id) const {
    if (bitmap() == nullptr) {
        return 0;
    }
    if (id < base_) {
        return 0;
    }

    const uint64_t offset = id - base_;
    if (offset > std::numeric_limits<uint32_t>::max()) {
        return count();
    }

    const uint32_t offset32 = static_cast<uint32_t>(offset);
    const int64_t exact_index = roaring::api::roaring_bitmap_get_index(bitmap(), offset32);
    if (exact_index >= 0) {
        return static_cast<size_t>(exact_index);
    }

    return static_cast<size_t>(roaring::api::roaring_bitmap_rank(bitmap(), offset32));
}

bool RoaringIds::find_index(uint64_t id, size_t* index) const {
    if (index != nullptr) {
        *index = npos;
    }
    if (bitmap() == nullptr || id < base_) {
        return false;
    }

    const uint64_t offset = id - base_;
    if (offset > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    const int64_t exact_index =
        roaring::api::roaring_bitmap_get_index(bitmap(), static_cast<uint32_t>(offset));
    if (exact_index < 0) {
        return false;
    }

    if (index != nullptr) {
        *index = static_cast<size_t>(exact_index);
    }
    return true;
}

Ret RoaringIds::serialize(char* buffer) const {
    if (bitmap() == nullptr) {
        return Ret("RoaringIds::serialize: bitmap is not initialized");
    }
    if (buffer == nullptr) {
        return Ret("RoaringIds::serialize: buffer pointer is null");
    }
    if (!is_aligned_32(buffer)) {
        return Ret("RoaringIds::serialize: buffer must be 32-byte aligned");
    }

    roaring::api::roaring_bitmap_frozen_serialize(bitmap(), buffer);
    return Ret(0);
}

RoaringIds::Iterator RoaringIds::begin() const {
    return Iterator(this);
}

RoaringIds::Iterator RoaringIds::end() const {
    return Iterator(nullptr);
}

} // namespace sketch2
