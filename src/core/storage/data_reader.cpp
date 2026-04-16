// Implements memory-mapped reading of persisted data and delta files.

#include "data_reader.h"
#include "core/storage/data_file_layout.h"
#include <algorithm>
#include <sys/stat.h>
#include <cassert>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>

namespace sketch2 {

// --- Iterator ---

DataReader::Iterator::Iterator(const DataReader* reader, const DataReader* delta_reader, size_t index)
    : reader_(reader), delta_reader_(delta_reader), index_(index),
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
        return delta_reader_->ids_.id(ind);
    }

    return reader_->ids_.id(index_);
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
        return reader_->ids_.id(index_);
    }

    return reader_->delta_->ids_.id(index_);
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
    if (initialized_) {
        return Ret("DataReader is initialized already.");
    }
    if (delta && !delta->check_consistency()) {
        return Ret("DataReader: delta is inconsistent");
    }

    path_ = path;

    int fd = -1;
    size_t file_size = 0;
    auto fail = [this, &fd](const std::string& message) {
        if (fd >= 0) {
            close(fd);
            fd = -1;
        }
        reset_state_();
        return Ret(message);
    };

    fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return Ret("DataReader: failed to open file: " + path);
    }

    Ret ret = read_header_(fd, path, &file_size);
    if (ret.code() != 0) {
        return fail(ret.message());
    }

    DataMetadataLayout metadata_layout{};
    try {
        ret = validate_header_and_layout_(file_size, &metadata_layout);
        if (ret.code() != 0) {
            return fail(ret.message());
        }
        ret = validate_delta_(delta);
        if (ret.code() != 0) {
            return fail(ret.message());
        }
        ret = map_regions_(fd, file_size, metadata_layout);
        if (ret.code() != 0) {
            return fail(ret.message());
        }
    } catch (const std::exception& ex) {
        return fail(std::string("DataReader: failed to initialize mapped metadata: ") + ex.what());
    }

    close(fd);
    fd = -1;

    initialized_ = true;
    delta_  = std::move(delta);

    if (delta_) {
        changed_bitset_.resize(hdr_.count);
        ret = init_delta();
        if (ret.code() != 0) {
            return fail(ret.message());
        }
    }

    assert_invariants_();
    return Ret(0);
}

void DataReader::reset_state_() {
    vectors_region_.reset();
    ids_region_.reset();
    deleted_ids_region_.reset();
    norms_region_.reset();
    hdr_ = {};
    initialized_ = false;
    ids_.clear();
    norms_ = nullptr;
    deleted_ids_.clear();
    vector_size_ = 0;
    stride_ = 0;
    changed_bitset_.resize(0);
    delta_.reset();
}

Ret DataReader::read_header_(int fd, const std::string& path, size_t* file_size) {
    if (fd < 0 || file_size == nullptr) {
        return Ret("DataReader: missing fd or file-size output for header read");
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        return Ret("DataReader: failed to stat file: " + path);
    }

    *file_size = static_cast<size_t>(st.st_size);
    if (*file_size < sizeof(DataFileHeader)) {
        return Ret("DataReader: file too small to contain a valid header");
    }

    const ssize_t header_bytes = pread(fd, &hdr_, sizeof(hdr_), 0);
    if (header_bytes != static_cast<ssize_t>(sizeof(hdr_))) {
        hdr_ = {};
        return Ret("DataReader: failed to read file header");
    }

    return Ret(0);
}

Ret DataReader::validate_header_and_layout_(size_t file_size, DataMetadataLayout* metadata_layout) {
    if (metadata_layout == nullptr) {
        return Ret("DataReader: missing metadata layout output");
    }

    if (hdr_.base.magic != kMagic) return Ret("DataReader: invalid magic number");
    if (hdr_.base.kind != static_cast<uint16_t>(FileType::Data)) return Ret("DataReader: not a data file");
    if (hdr_.base.version != kVersion) {
        return Ret("DataReader: unsupported file version");
    }

    type_ = data_type_from_int(hdr_.type);
    const size_t elem_size = data_type_size(type_);
    if (elem_size == 0) {
        return Ret("DataReader: invalid element type size");
    }

    const size_t dim = static_cast<size_t>(hdr_.dim);
    if (dim < 4) {
        return Ret("DataReader: dimension too small");
    }

    vector_size_ = dim * elem_size;
    stride_ = static_cast<size_t>(hdr_.vector_stride);
    if ((hdr_.flags & ~kDataFileHasCosineInvNorms) != 0u) {
        return Ret("DataReader: unsupported data-file flags");
    }
    if (hdr_.data_offset < sizeof(DataFileHeader)
            || (hdr_.data_offset % static_cast<uint64_t>(kDataRegionAlignment)) != 0) {
        return Ret("DataReader: invalid data offset alignment");
    }
    if (stride_ < vector_size_ || (stride_ % kDataAlignment) != 0) {
        return Ret("DataReader: invalid vector stride");
    }

    *metadata_layout = compute_data_metadata_layout(hdr_, hdr_.count);
    if (hdr_.vectors_bytes != metadata_layout->vectors_bytes
            || hdr_.cosine_inv_norms_offset != metadata_layout->cosine_inv_norms_offset
            || hdr_.cosine_inv_norms_bytes != metadata_layout->cosine_inv_norms_bytes
            || hdr_.ids_offset != metadata_layout->ids_trailer_offset
            || hdr_.deleted_ids_offset != compute_deleted_ids_offset(hdr_.ids_offset, hdr_.ids_bytes)) {
        return Ret("DataReader: malformed section layout in header");
    }
    metadata_layout->deleted_ids_offset = hdr_.deleted_ids_offset;
    metadata_layout->deleted_ids_bytes = hdr_.deleted_ids_bytes;
    metadata_layout->deleted_ids_padding = compute_deleted_ids_padding(hdr_.ids_offset, hdr_.ids_bytes);

    const size_t end_of_deleted_ids =
        static_cast<size_t>(hdr_.deleted_ids_offset) + static_cast<size_t>(hdr_.deleted_ids_bytes);
    if (file_size < metadata_layout->ids_trailer_offset || end_of_deleted_ids != file_size) {
        return Ret("DataReader: truncated or malformed data file");
    }

    return Ret(0);
}

Ret DataReader::validate_delta_(const std::unique_ptr<DataReader>& delta) const {
    if (!delta) {
        return Ret(0);
    }
    if (!delta->initialized_) return Ret("DataReader: invalid delta");
    if (type_ != delta->type_) return Ret("DataReader: invalid delta type");
    if (vector_size_ != delta->vector_size_) return Ret("DataReader: invalid delta dim");
    if (stride_ != delta->stride_) return Ret("DataReader: invalid delta stride");
    if (data_file_has_cosine_inv_norms(hdr_) != data_file_has_cosine_inv_norms(delta->hdr_)) {
        return Ret("DataReader: invalid delta cosine inverse-norm layout");
    }
    return Ret(0);
}

Ret DataReader::map_regions_(int fd, size_t file_size, const DataMetadataLayout& metadata_layout) {
    if (metadata_layout.vectors_bytes > 0) {
        CHECK(vectors_region_.init(fd, hdr_.data_offset, metadata_layout.vectors_bytes));
    }
    if (metadata_layout.cosine_inv_norms_bytes > 0) {
        CHECK(norms_region_.init(fd, metadata_layout.cosine_inv_norms_offset,
            metadata_layout.cosine_inv_norms_bytes));
    }

    CHECK(ids_region_.init(fd, hdr_.ids_offset, hdr_.ids_bytes));
    size_t active_ids_bytes = 0;
    size_t exact_active_ids_bytes = 0;
    CHECK(ids_.map(ids_region_.data(), ids_region_.size(), &exact_active_ids_bytes));
    if (exact_active_ids_bytes != ids_region_.size()) {
        return Ret("DataReader: malformed ids trailer size");
    }
    active_ids_bytes = exact_active_ids_bytes;
    if (active_ids_bytes != hdr_.ids_bytes) {
        return Ret("DataReader: ids trailer size does not match header");
    }

    CHECK(deleted_ids_region_.init(fd, hdr_.deleted_ids_offset, hdr_.deleted_ids_bytes));
    size_t exact_deleted_ids_bytes = 0;
    CHECK(deleted_ids_.map(
        deleted_ids_region_.data(), deleted_ids_region_.size(), &exact_deleted_ids_bytes));
    if (exact_deleted_ids_bytes != deleted_ids_region_.size()) {
        return Ret("DataReader: malformed ids trailer size");
    }
    const size_t deleted_ids_bytes = exact_deleted_ids_bytes;

    const size_t parsed_file_size = static_cast<size_t>(hdr_.deleted_ids_offset) + deleted_ids_bytes;
    if (parsed_file_size != file_size) {
        return Ret("DataReader: malformed ids trailer size");
    }
    if (ids_.count() != hdr_.count || deleted_ids_.count() != hdr_.deleted_count) {
        return Ret("DataReader: ids trailer count does not match header");
    }

    norms_ = metadata_layout.cosine_inv_norms_bytes > 0
        ? reinterpret_cast<const float*>(norms_region_.data())
        : nullptr;
    return Ret(0);
}

// Marks base-file rows hidden when the attached delta either overwrites or
// deletes the same id, allowing iteration to skip superseded records cheaply.
Ret DataReader::init_delta() {
    if (!initialized_ || !delta_) {
        return Ret("DataReader::init_delta: reader is not initialized");
    }

    auto mark_hidden = [this](const CompactIdsExt& other_ids) {
        const size_t base_count = ids_.count();
        const size_t other_count = other_ids.count();
        for (size_t i = 0, j = 0; i < base_count; ++i) {
            const uint64_t id = ids_.id_unchecked(i);
            while (j < other_count && other_ids.id_unchecked(j) < id) {
                ++j;
            }

            if (j >= other_count) {
                break;
            }

            if (other_ids.id_unchecked(j) == id) {
                changed_bitset_.set(i);
            }
        }
    };

    mark_hidden(delta_->deleted_ids_);
    mark_hidden(delta_->ids_);

    return Ret(0);
}

// Checks layout-derived state in debug builds so mmap pointers, strides, ids,
// and delta metadata stay internally consistent.
void DataReader::assert_invariants_() const {
#ifndef NDEBUG
    if (!initialized_) {
        assert(vectors_region_.empty());
        assert(ids_region_.empty());
        assert(deleted_ids_region_.empty());
        assert(norms_region_.empty());
        assert(ids_.empty());
        assert(norms_ == nullptr);
        assert(deleted_ids_.empty());
        assert(vector_size_ == 0);
        assert(stride_ == 0);
        return;
    }

    assert(hdr_.base.magic == kMagic);
    assert(hdr_.base.kind == static_cast<uint16_t>(FileType::Data));
    assert(hdr_.base.version == kVersion);
    assert(vector_size_ == compute_vector_size(type_, hdr_.dim));
    assert(stride_ == hdr_.vector_stride);
    assert(stride_ >= vector_size_);
    assert((hdr_.data_offset % static_cast<uint64_t>(kDataRegionAlignment)) == 0);
    assert((stride_ % kDataAlignment) == 0);
    assert(ids_.count() == hdr_.count);
    assert(deleted_ids_.count() == hdr_.deleted_count);
    assert(!ids_region_.empty());
    assert(!deleted_ids_region_.empty());

    const DataMetadataLayout metadata_layout = compute_data_metadata_layout(hdr_, hdr_.count);
    assert(vectors_region_.size() == metadata_layout.vectors_bytes);
    assert(norms_region_.size() == metadata_layout.cosine_inv_norms_bytes);
    assert(hdr_.vectors_bytes == metadata_layout.vectors_bytes);
    assert(hdr_.cosine_inv_norms_offset == metadata_layout.cosine_inv_norms_offset);
    assert(hdr_.cosine_inv_norms_bytes == metadata_layout.cosine_inv_norms_bytes);
    assert(hdr_.ids_offset == metadata_layout.ids_trailer_offset);
    assert(hdr_.ids_bytes == ids_region_.size());
    assert(hdr_.deleted_ids_offset ==
        static_cast<uint64_t>(compute_deleted_ids_offset(hdr_.ids_offset, hdr_.ids_bytes)));
    assert(hdr_.deleted_ids_bytes == deleted_ids_region_.size());

    if (data_file_has_cosine_inv_norms(hdr_)) {
        assert(norms_ == reinterpret_cast<const float*>(norms_region_.data()));
    } else {
        assert(norms_ == nullptr);
    }

    if (delta_) {
        assert(changed_bitset_.size() == hdr_.count);
        assert(type_ == delta_->type());
        assert(vector_size_ == delta_->size());
        assert(stride_ == delta_->stride());
        assert(has_cosine_inv_norms() == delta_->has_cosine_inv_norms());
    } else {
        assert(changed_bitset_.size() == 0);
    }
#endif
}

DataType DataReader::type() const {
    if (!initialized_) {
        throw std::runtime_error("DataReader::type: reader is not initialized");
    }
    return type_;
}

size_t DataReader::dim() const {
    if (!initialized_) {
        throw std::runtime_error("DataReader::dim: reader is not initialized");
    }
    return hdr_.dim;
}

size_t DataReader::size() const {
    if (!initialized_) {
        throw std::runtime_error("DataReader::size: reader is not initialized");
    }
    return vector_size_;
}

size_t DataReader::count() const {
    if (!initialized_) {
        throw std::runtime_error("DataReader::count: reader is not initialized");
    }
    return static_cast<size_t>(hdr_.count);
}

bool DataReader::has_cosine_inv_norms() const {
    if (!initialized_) {
        throw std::runtime_error("DataReader::has_cosine_inv_norms: reader is not initialized");
    }
    return data_file_has_cosine_inv_norms(hdr_);
}

DataReader::Iterator DataReader::begin() const {
    size_t index = 0;
    while (index < count() && is_hidden(index)) {
        ++index;
    }
    return Iterator(this, delta_ ? delta_.get() : nullptr, index);
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
    return ids_.id(index);
}

float DataReader::cosine_inv_norm(size_t index) const {
    if (index >= count()) {
        throw std::out_of_range("DataReader::cosine_inv_norm: index out of range");
    }
    if (!norms_) {
        return 0.0f;
    }
    return norms_[index];
}

const uint8_t* DataReader::at(size_t index) const {
    if (index >= count()) {
        throw std::out_of_range("DataReader::at: index out of range");
    }

    if (index < changed_bitset_.size() && changed_bitset_.get(index)) {
        return nullptr;
    }

    return vectors_region_.data() + index * stride_;
}

// Looks up an id in the base file and falls back to the attached delta when the
// base row is absent or hidden by newer updates.
const uint8_t* DataReader::get(uint64_t id) const {
    const size_t index = ids_.lower_bound_index(id);
    if (index >= ids_.count()) {
        if (delta_) {
            return delta_->get(id);
        }
        return nullptr;
    }

    if (ids_.id(index) != id) {
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

    return vectors_region_.data() + index * stride_;
}

bool DataReader::is_hidden(size_t index) const {
    return (index < changed_bitset_.size() && changed_bitset_.get(index));
}

uint64_t DataReader::deleted_id(size_t index) const {
    if (index >= deleted_count()) {
        throw std::out_of_range("DataReader::deleted_id: index out of range");
    }
    return deleted_ids_.id(index);
}

// Verifies that ids and deleted ids are strictly sorted and disjoint, which is
// required for binary search, merge logic, and hidden-row bookkeeping.
bool DataReader::check_consistency() const {
    if (!initialized_) {
        return false;
    }

    const size_t ids_count = count();
    const size_t deleted_count_ = deleted_count();

    for (size_t i = 1; i < deleted_count_; ++i) {
        if (deleted_ids_.id_unchecked(i - 1) >= deleted_ids_.id_unchecked(i)) {
            return false;
        }
    }

    for (size_t i = 1; i < ids_count; ++i) {
        if (ids_.id_unchecked(i - 1) >= ids_.id_unchecked(i)) {
            return false;
        }
    }

    size_t i = 0;
    size_t j = 0;
    while (i < ids_count && j < deleted_count_) {
        const uint64_t id = ids_.id_unchecked(i);
        const uint64_t deleted_id = deleted_ids_.id_unchecked(j);

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
