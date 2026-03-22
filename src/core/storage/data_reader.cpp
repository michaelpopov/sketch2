// Implements memory-mapped reading of persisted data and delta files.

#include "data_reader.h"
#include "core/storage/data_file_layout.h"
#include <algorithm>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cassert>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>

namespace sketch2 {

// --- Iterator ---

DataReader::Iterator::Iterator(const DataReader* reader, const DataReader* delta_reader, size_t index, const uint64_t* ids)
    : reader_(reader), delta_reader_(delta_reader), index_(index), ids_(ids),
      count_(reader_->count() + (delta_reader_ ? delta_reader_->count() : 0)) {}

void DataReader::Iterator::next() {
    ++index_;
    while (index_ < reader_->count() && reader_->is_hidden(index_)) {
        ++index_;
    }
}

bool DataReader::Iterator::eof() const {
    return index_ >= count_;
}

const uint8_t* DataReader::Iterator::data() const {
    if (index_ >= count_) {
        throw std::out_of_range("DataReader::Iterator::data: index out of range");
    }

    if (index_ >= reader_->count()) {
        assert(delta_reader_);
        const size_t ind = index_ - reader_->count();
        return delta_reader_->at(ind);
    }

    return reader_->at(index_);
}

float DataReader::Iterator::cosine_inv_norm() const {
    if (index_ >= count_) {
        throw std::out_of_range("DataReader::Iterator::cosine_inv_norm: index out of range");
    }

    if (index_ >= reader_->count()) {
        assert(delta_reader_);
        const size_t ind = index_ - reader_->count();
        return delta_reader_->cosine_inv_norm(ind);
    }

    return reader_->cosine_inv_norm(index_);
}

uint64_t DataReader::Iterator::id() const {
    if (index_ >= count_) {
        throw std::out_of_range("DataReader::Iterator::id: index out of range");
    }

    if (index_ >= reader_->count()) {
        assert(delta_reader_);
        const size_t ind = index_ - reader_->count();
        return delta_reader_->ids_[ind];
    }

    return ids_[index_];
}

// --- OrderedIterator ---

void DataReader::OrderedIterator::next() {
    ++index_;
    if (!reader_ || source_ != Source::Base) {
        return;
    }
    while (index_ < reader_->count() && reader_->is_hidden(index_)) {
        ++index_;
    }
}

bool DataReader::OrderedIterator::eof() const {
    if (!reader_) {
        return true;
    }
    if (source_ == Source::Base) {
        return index_ >= reader_->count();
    }
    return !reader_->delta_ || index_ >= reader_->delta_->count();
}

const uint8_t* DataReader::OrderedIterator::data() const {
    if (eof()) {
        throw std::out_of_range("DataReader::OrderedIterator::data: index out of range");
    }

    if (source_ == Source::Base) {
        return reader_->at(index_);
    }

    return reader_->delta_->at(index_);
}

float DataReader::OrderedIterator::cosine_inv_norm() const {
    if (eof()) {
        throw std::out_of_range("DataReader::OrderedIterator::cosine_inv_norm: index out of range");
    }

    if (source_ == Source::Base) {
        return reader_->cosine_inv_norm(index_);
    }

    return reader_->delta_->cosine_inv_norm(index_);
}

uint64_t DataReader::OrderedIterator::id() const {
    if (eof()) {
        throw std::out_of_range("DataReader::OrderedIterator::id: index out of range");
    }

    if (source_ == Source::Base) {
        return reader_->ids_[index_];
    }

    return reader_->delta_->ids_[index_];
}

// --- DataReader ---

DataReader::~DataReader() {
    if (map_) {
        munmap(const_cast<uint8_t*>(map_), map_len_);
    }
}

Ret DataReader::init(const std::string &path, std::unique_ptr<DataReader> delta) {
    try {
        return init_(path, std::move(delta));
    } catch (const std::exception& e) {
        return Ret(e.what());
    }
}

// Memory-maps a binary data file, validates its layout, and caches pointers to
// the vector, cosine, id, and delete sections. When a delta reader is attached,
// it also builds a visibility bitset for base rows shadowed by newer updates.
Ret DataReader::init_(const std::string& path, std::unique_ptr<DataReader> delta) {
    if (map_) {
        return Ret("DataReader is initialized already.");
    }
    if (delta && !delta->check_consistency()) {
        return Ret("DataReader: delta is inconsistent");
    }

    path_ = path;

    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return Ret("DataReader: failed to open file: " + path);
    }
    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return Ret("DataReader: failed to stat file: " + path);
    }
    map_len_ = static_cast<size_t>(st.st_size);
    
    if (map_len_ < sizeof(DataFileHeader)) {
        close(fd);
        return Ret("DataReader: file too small to contain a valid header");
    }
    void* m = mmap(nullptr, map_len_, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) {
        return Ret("DataReader: failed to mmap file: " + path);
    }
    madvise(m, map_len_, MADV_SEQUENTIAL);
    map_ = static_cast<uint8_t*>(m);
    auto fail = [this](const std::string& message) -> Ret {
        munmap(const_cast<uint8_t*>(map_), map_len_);
        map_ = nullptr;
        map_len_ = 0;
        hdr_ = nullptr;
        ids_ = nullptr;
        cosine_inv_norms_ = nullptr;
        deleted_ids_ = nullptr;
        size_ = 0;
        stride_ = 0;
        return Ret(message);
    };

    hdr_ = reinterpret_cast<const DataFileHeader *>(map_);
    if (hdr_->base.magic != kMagic) return fail("DataReader: invalid magic number");
    if (hdr_->base.kind != static_cast<uint16_t>(FileType::Data)) return fail("DataReader: not a data file");
    if (hdr_->base.version != kVersion) return fail("DataReader: unsupported file version");

    try {
        type_ = data_type_from_int(hdr_->type);
    } catch (const std::exception& ex) {
        return fail(ex.what());
    }
    const size_t elem_size = data_type_size(type_);
    if (elem_size == 0) {
        return fail("DataReader: invalid element type size");
    }

    const size_t dim = static_cast<size_t>(hdr_->dim);
    if (dim < 4) {
        return fail("DataReader: dimension too small");
    }

    size_ = dim * elem_size;
    stride_ = static_cast<size_t>(hdr_->vector_stride);
    const size_t count = static_cast<size_t>(hdr_->count);
    const size_t deleted_count = static_cast<size_t>(hdr_->deleted_count);

    if ((hdr_->flags & ~kDataFileHasCosineInvNorms) != 0u) {
        return fail("DataReader: unsupported data-file flags");
    }

    const IdsLayout ids_layout = compute_ids_layout(*hdr_, count);
    const size_t ids_bytes = (deleted_count + count) * sizeof(uint64_t);
    if (hdr_->data_offset < sizeof(DataFileHeader) || (hdr_->data_offset % kDataAlignment) != 0) {
        return fail("DataReader: invalid data offset alignment");
    }
    if (stride_ < size_ || (stride_ % kDataAlignment) != 0) {
        return fail("DataReader: invalid vector stride");
    }
    if (ids_layout.ids_offset % alignof(uint64_t) != 0) {
        return fail("DataReader: invalid ids offset alignment");
    }
    if (map_len_ != ids_layout.ids_offset + ids_bytes) {
        return fail("DataReader: truncated or malformed data file");
    }

    if (delta) {
        if (!delta->hdr_) return fail("DataReader: invalid delta");
        if (type_ != delta->type()) return fail("DataReader: invalid delta type");
        if (size_ != delta->size()) return fail("DataReader: invalid delta dim");
        if (stride_ != delta->stride()) return fail("DataReader: invalid delta stride");
        if (has_cosine_inv_norms() != delta->has_cosine_inv_norms()) {
            return fail("DataReader: invalid delta cosine inverse-norm layout");
        }
    }

    cosine_inv_norms_ = ids_layout.cosine_inv_norms_bytes == 0
        ? nullptr
        : reinterpret_cast<const float*>(map_ + ids_layout.cosine_inv_norms_offset);
    ids_ = reinterpret_cast<const uint64_t*>(map_ + ids_layout.ids_offset);
    deleted_ids_ = ids_ + count;
    delta_  = std::move(delta);

    if (delta_) {
        bitset_.resize(hdr_->count);
        CHECK(init_delta());
    }

    assert_invariants_();
    return Ret(0);
}

// Marks base-file rows hidden when the attached delta either overwrites or
// deletes the same id, allowing iteration to skip superseded records cheaply.
Ret DataReader::init_delta() {
    if (!hdr_ || !ids_ || !delta_) {
        return Ret("DataReader::init_delta: reader is not initialized");
    }

    auto mark_hidden = [this](const uint64_t* other_ids, size_t other_count) {
        for (size_t i = 0, j = 0; i < hdr_->count; ++i) {
            const uint64_t id = ids_[i];
            while (j < other_count && other_ids[j] < id) {
                ++j;
            }

            if (j >= other_count) {
                break;
            }

            if (other_ids[j] == id) {
                bitset_.set(i);
            }
        }
    };

    mark_hidden(delta_->deleted_ids_, delta_->deleted_count());
    mark_hidden(delta_->ids_, delta_->count());

    return Ret(0);
}

// Checks layout-derived state in debug builds so mmap pointers, strides, ids,
// and delta metadata stay internally consistent.
void DataReader::assert_invariants_() const {
#ifndef NDEBUG
    if (!hdr_) {
        assert(map_ == nullptr);
        assert(ids_ == nullptr);
        assert(cosine_inv_norms_ == nullptr);
        assert(deleted_ids_ == nullptr);
        assert(size_ == 0);
        assert(stride_ == 0);
        return;
    }

    assert(map_ != nullptr);
    assert(hdr_->base.magic == kMagic);
    assert(hdr_->base.kind == static_cast<uint16_t>(FileType::Data));
    assert(hdr_->base.version == kVersion);
    assert(size_ == compute_vector_size(type_, hdr_->dim));
    assert(stride_ == hdr_->vector_stride);
    assert(stride_ >= size_);
    assert((hdr_->data_offset % kDataAlignment) == 0);
    assert((stride_ % kDataAlignment) == 0);
    assert(ids_ != nullptr);
    assert(deleted_ids_ == ids_ + hdr_->count);

    if (data_file_has_cosine_inv_norms(*hdr_)) {
        assert(hdr_->count == 0 || cosine_inv_norms_ != nullptr);
    } else {
        assert(cosine_inv_norms_ == nullptr);
    }

    if (delta_) {
        assert(bitset_.size() == hdr_->count);
        assert(type_ == delta_->type());
        assert(size_ == delta_->size());
        assert(stride_ == delta_->stride());
        assert(has_cosine_inv_norms() == delta_->has_cosine_inv_norms());
    } else {
        assert(bitset_.size() == 0);
    }
#endif
}

DataType DataReader::type() const {
    if (!hdr_) {
        throw std::runtime_error("DataReader::type: reader is not initialized");
    }
    return type_;
}

size_t DataReader::dim() const {
    if (!hdr_) {
        throw std::runtime_error("DataReader::dim: reader is not initialized");
    }
    return hdr_->dim;
}

size_t DataReader::size() const {
    if (!hdr_) {
        throw std::runtime_error("DataReader::size: reader is not initialized");
    }
    return size_;
}

size_t DataReader::count() const {
    if (!hdr_) {
        throw std::runtime_error("DataReader::count: reader is not initialized");
    }
    return hdr_->count;
}

bool DataReader::has_cosine_inv_norms() const {
    if (!hdr_) {
        throw std::runtime_error("DataReader::has_cosine_inv_norms: reader is not initialized");
    }
    return data_file_has_cosine_inv_norms(*hdr_);
}

DataReader::Iterator DataReader::begin() const {
    size_t index = 0;
    while (index < count() && is_hidden(index)) {
        ++index;
    }
    return Iterator(this, delta_ ? delta_.get() : nullptr, index, ids_);
}

DataReader::OrderedIterator DataReader::base_begin() const {
    size_t index = 0;
    while (index < count() && is_hidden(index)) {
        ++index;
    }
    return OrderedIterator(this, OrderedIterator::Source::Base, index);
}

DataReader::OrderedIterator DataReader::delta_begin() const {
    return OrderedIterator(this, OrderedIterator::Source::Delta, 0);
}

uint64_t DataReader::id(size_t index) const {
    if (index >= count()) {
        throw std::out_of_range("DataReader::id: index out of range");
    }
    return ids_[index];
}

float DataReader::cosine_inv_norm(size_t index) const {
    if (index >= count()) {
        throw std::out_of_range("DataReader::cosine_inv_norm: index out of range");
    }
    if (!cosine_inv_norms_) {
        return 0.0f;
    }
    return cosine_inv_norms_[index];
}

const uint8_t* DataReader::at(size_t index) const {
    if (index >= count()) {
        throw std::out_of_range("DataReader::at: index out of range");
    }

    if (index < bitset_.size() && bitset_.get(index)) {
        return nullptr;
    }

    return map_ + hdr_->data_offset + index * stride_;
}

// Looks up an id in the base file and falls back to the attached delta when the
// base row is absent or hidden by newer updates.
const uint8_t* DataReader::get(uint64_t id) const {
    const size_t n = count();
    const uint64_t* first = ids_;
    const uint64_t* last  = ids_ + n;
    const uint64_t* it    = std::lower_bound(first, last, id);

    if (it == last) {
        if (delta_) {
            return delta_->get(id);
        }
        return nullptr;
    }

    const size_t index = static_cast<size_t>(it - first);
    if (ids_[index] != id) {
        if (delta_) {
            return delta_->get(id);
        }
        return nullptr;
    }

    if (is_hidden(index)) {
        if (delta_) {
            return delta_->get(id);
        }
        return nullptr;
    }

    return map_ + hdr_->data_offset + index * stride_;
}

bool DataReader::is_hidden(size_t index) const {
    return (index < bitset_.size() && bitset_.get(index));
}

uint64_t DataReader::deleted_id(size_t index) const {
    if (index >= deleted_count()) {
        throw std::out_of_range("DataReader::deleted_id: index out of range");
    }
    return deleted_ids_[index];
}

// Verifies that ids and deleted ids are strictly sorted and disjoint, which is
// required for binary search, merge logic, and hidden-row bookkeeping.
bool DataReader::check_consistency() const {
    if (!hdr_) {
        return false;
    }

    const size_t ids_count = count();
    const size_t deleted_count_ = deleted_count();

    for (size_t i = 1; i < deleted_count_; ++i) {
        if (deleted_ids_[i - 1] >= deleted_ids_[i]) {
            return false;
        }
    }

    for (size_t i = 1; i < ids_count; ++i) {
        if (ids_[i - 1] >= ids_[i]) {
            return false;
        }
    }

    size_t i = 0;
    size_t j = 0;
    while (i < ids_count && j < deleted_count_) {
        const uint64_t id = ids_[i];
        const uint64_t deleted_id = deleted_ids_[j];

        if (id == deleted_id) {
            return false;
        }
        if (id < deleted_id) {
            ++i;
        } else {
            ++j;
        }
    }
    return true;
}

} // namespace sketch2
