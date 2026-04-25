// Implements RoaringIds, a uint32_t id container backed by CRoaring.

#include "roaring_ids.h"

#include <algorithm>
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
    base_ = base;
    return Ret(0);
}

void RoaringIds::reset_view() {
    bitmap_.reset();
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

roaring::api::roaring_bitmap_t* RoaringIdsBuilder::bitmap() {
    return bitmap_.get();
}

const roaring::api::roaring_bitmap_t* RoaringIdsBuilder::bitmap() const {
    return bitmap_.get();
}

Ret RoaringIdsBuilder::init(uint64_t base) {
    RoaringIds::BitmapPtr new_bitmap(roaring::api::roaring_bitmap_create());
    if (!new_bitmap) {
        return Ret("RoaringIdsBuilder::init: failed to allocate bitmap");
    }

    bitmap_ = std::move(new_bitmap);
    buffer_.clear();
    fill_size_ = 0;
    base_ = base;
    return Ret(0);
}

Ret RoaringIdsBuilder::init_buffered(uint64_t base, size_t buffer_size, bool needs_sorting) {
    if (buffer_size == 0) {
        return Ret("RoaringIdsBuilder::init_buffered: buffer size must be greater than 0");
    }
    CHECK(init(base));
    buffer_.resize(buffer_size);
    needs_sorting_ = needs_sorting;
    return Ret(0);
}

Ret RoaringIdsBuilder::init_copy(const RoaringIds& other, uint64_t base) {
    // An uninitialized other has no data: treat as empty, base is irrelevant.
    if (other.bitmap() == nullptr) {
        return init(base);
    }
    if (other.base_ != base) {
        return Ret("RoaringIdsBuilder::init_copy: base mismatch");
    }
    RoaringIds::BitmapPtr new_bitmap(roaring::api::roaring_bitmap_copy(other.bitmap()));
    if (!new_bitmap) {
        return Ret("RoaringIdsBuilder::init_copy: failed to clone bitmap");
    }
    bitmap_ = std::move(new_bitmap);
    buffer_.clear();
    fill_size_ = 0;
    base_ = base;
    return Ret(0);
}

Ret RoaringIdsBuilder::add(uint64_t id) {
    if (bitmap() == nullptr) {
        return Ret("RoaringIdsBuilder::add: bitmap is not initialized");
    }
    if (id < base_) {
        return Ret("RoaringIdsBuilder::add: id is below base");
    }

    const uint64_t offset = id - base_;
    if (offset > std::numeric_limits<uint32_t>::max()) {
        return Ret("RoaringIdsBuilder::add: id offset exceeds uint32_t range");
    }

    if (!buffer_.empty()) {
        buffer_[fill_size_] = id;
        ++fill_size_;
        if (fill_size_ == buffer_.size()) {
            CHECK(flush_buffer());
        }
        return Ret(0);
    }

    roaring::api::roaring_bitmap_add(bitmap(), static_cast<uint32_t>(offset));
    return Ret(0);
}

Ret RoaringIdsBuilder::load(const uint64_t* values, size_t size) {
    if (bitmap() == nullptr) {
        return Ret("RoaringIdsBuilder::load: bitmap is not initialized");
    }
    if (size == 0) {
        return Ret(0);
    }
    if (values == nullptr) {
        return Ret("RoaringIdsBuilder::load: values pointer is null");
    }
    if (values[0] < base_ || values[size - 1] < base_) {
        return Ret("RoaringIdsBuilder::load: id is below base");
    }
    if (values[size - 1] - base_ > std::numeric_limits<uint32_t>::max()) {
        return Ret("RoaringIdsBuilder::load: id offset exceeds uint32_t range");
    }

    CHECK(flush_buffer());
    return load_unbuffered(values, size);
}

Ret RoaringIdsBuilder::load_unbuffered(const uint64_t* values, size_t size) {
    roaring::api::roaring_bulk_context_t ctx = {};
    size_t i = 0;
    while (i < size) {
        size_t j = i + 1;
        while (j < size && values[j] == values[j - 1] + 1) {
            ++j;
        }
        const size_t len = j - i;

        if (len >= 4) {
            roaring::api::roaring_bitmap_add_range_closed(
                bitmap(),
                static_cast<uint32_t>(values[i] - base_),
                static_cast<uint32_t>(values[j - 1] - base_));
            // Non-bulk modification invalidates the bulk context.
            ctx = {};
        } else {
            for (size_t k = i; k < j; ++k) {
                roaring::api::roaring_bitmap_add_bulk(
                    bitmap(), &ctx,
                    static_cast<uint32_t>(values[k] - base_));
            }
        }
        i = j;
    }
    return Ret(0);
}

Ret RoaringIdsBuilder::union_in_place(const RoaringIds& other) {
    if (bitmap() == nullptr) {
        return Ret("RoaringIdsBuilder::union_in_place: bitmap is not initialized");
    }
    CHECK(flush_buffer());
    // Uninitialized other has no data: nothing to do, base is irrelevant.
    if (other.bitmap() == nullptr) {
        return Ret(0);
    }
    if (other.base_ != base_) {
        return Ret("RoaringIdsBuilder::union_in_place: base mismatch");
    }
    roaring::api::roaring_bitmap_or_inplace(bitmap(), other.bitmap());
    return Ret(0);
}

Ret RoaringIdsBuilder::andnot_in_place(const RoaringIds& other) {
    if (bitmap() == nullptr) {
        return Ret("RoaringIdsBuilder::andnot_in_place: bitmap is not initialized");
    }
    CHECK(flush_buffer());
    if (other.bitmap() == nullptr) {
        return Ret(0);
    }
    if (other.base_ != base_) {
        return Ret("RoaringIdsBuilder::andnot_in_place: base mismatch");
    }
    roaring::api::roaring_bitmap_andnot_inplace(bitmap(), other.bitmap());
    return Ret(0);
}

size_t RoaringIdsBuilder::count() const {
    if (bitmap() == nullptr) {
        return 0;
    }
    return static_cast<size_t>(roaring::api::roaring_bitmap_get_cardinality(bitmap())) + fill_size_;
}

bool RoaringIdsBuilder::empty() const {
    return count() == 0;
}

Ret RoaringIdsBuilder::flush_buffer() {
    if (fill_size_ == 0) {
        return Ret(0);
    }
    if (bitmap() == nullptr) {
        return Ret("RoaringIdsBuilder::flush_buffer: bitmap is not initialized");
    }

    if (needs_sorting_) {
        std::sort(buffer_.begin(), buffer_.begin() + fill_size_);
    }

    CHECK(load_unbuffered(buffer_.data(), fill_size_));
    fill_size_ = 0;
    return Ret(0);
}

Ret RoaringIdsBuilder::compact() {
    if (bitmap() == nullptr) {
        return Ret(0);
    }
    CHECK(flush_buffer());
    roaring::api::roaring_bitmap_run_optimize(bitmap());
    roaring::api::roaring_bitmap_shrink_to_fit(bitmap());
    return Ret(0);
}

RoaringIds RoaringIdsBuilder::build() && {
    const Ret ret = compact();
    if (ret.code() != 0) {
        throw std::runtime_error(ret.message());
    }
    RoaringIds ids;
    ids.bitmap_ = std::move(bitmap_);
    ids.base_ = base_;
    buffer_.clear();
    fill_size_ = 0;
    base_ = 0;
    return ids;
}

} // namespace sketch2
