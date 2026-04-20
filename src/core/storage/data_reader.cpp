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
    if (index_ < reader_->count_unchecked()) {
        index_ = reader_->next_visible_base_index_unchecked(index_);
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

float DataReader::Iterator::get_norm() const {
    if (index_ >= count_) {
        throw std::out_of_range("DataReader::Iterator::get_norm: index out of range");
    }

    if (index_ >= reader_->count()) {
        assert(delta_reader_);
        const size_t ind = index_ - reader_->count();
        return delta_reader_->get_norm(ind);
    }

    return reader_->get_norm(index_);
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

Ret DataReader::init(const std::string &path, std::unique_ptr<DataReader> delta) {
    try {
        return init_(path, std::move(delta));
    } catch (const std::exception& e) {
        return Ret(e.what());
    }
}

// Memory-maps a binary data file, validates its layout, and caches pointers to
// the vector, id, and delete sections. When a delta reader is attached,
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
    hdr_ = {};
    initialized_ = false;
    ids_.clear();
    deleted_ids_.clear();
    vector_size_ = 0;
    norm_offset_in_record_ = 0;
    stride_ = 0;
    changed_bitset_.resize(0);
    has_hidden_rows_ = false;
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

    const DataRecordLayout record_layout =
        compute_data_record_layout(type_, hdr_.dim, data_file_has_norms(hdr_));
    vector_size_ = record_layout.vector_size;
    norm_offset_in_record_ = record_layout.norm_offset;
    stride_ = static_cast<size_t>(hdr_.vector_stride);
    if ((hdr_.flags & ~kDataFileNormKindMask) != 0u) {
        return Ret("DataReader: unsupported data-file flags");
    }
    if (!data_file_has_valid_norm_flags(hdr_)) {
        return Ret("DataReader: invalid stored norm flags");
    }
    if (hdr_.data_offset < sizeof(DataFileHeader)
            || (hdr_.data_offset % static_cast<uint64_t>(kDataRegionAlignment)) != 0) {
        return Ret("DataReader: invalid data offset alignment");
    }
    if (stride_ < vector_size_ || (stride_ % kDataAlignment) != 0) {
        return Ret("DataReader: invalid vector stride");
    }
    if (data_file_has_norms(hdr_) && stride_ != record_layout.stride) {
        return Ret("DataReader: invalid inline norm layout");
    }

    *metadata_layout = compute_data_metadata_layout(hdr_, hdr_.count);
    if (hdr_.vectors_bytes != metadata_layout->vectors_bytes
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
    if (hdr_.min_range_id != delta->hdr_.min_range_id) return Ret("DataReader: invalid delta min_range_id");
    if (norm_offset_in_record_ != delta->norm_offset_in_record_) {
        return Ret("DataReader: invalid delta norm layout");
    }
    if (data_file_norm_flags(hdr_) != data_file_norm_flags(delta->hdr_)) {
        return Ret("DataReader: invalid delta norm layout");
    }
    return Ret(0);
}

Ret DataReader::map_regions_(int fd, size_t file_size, const DataMetadataLayout& metadata_layout) {
    if (metadata_layout.vectors_bytes > 0) {
        CHECK(vectors_region_.init(fd, hdr_.data_offset, metadata_layout.vectors_bytes, true));
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

    return Ret(0);
}

// Marks base-file rows hidden when the attached delta either overwrites or
// deletes the same id, allowing iteration to skip superseded records cheaply.
Ret DataReader::init_delta() {
    if (!initialized_ || !delta_) {
        return Ret("DataReader::init_delta: reader is not initialized");
    }

    has_hidden_rows_ = false;
    auto mark_hidden = [this](const CompactIds& other_ids) {
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
                has_hidden_rows_ = true;
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
        assert(ids_.empty());
        assert(deleted_ids_.empty());
        assert(vector_size_ == 0);
        assert(norm_offset_in_record_ == 0);
        assert(stride_ == 0);
        return;
    }

    assert(hdr_.base.magic == kMagic);
    assert(hdr_.base.kind == static_cast<uint16_t>(FileType::Data));
    assert(hdr_.base.version == kVersion);
    assert(vector_size_ == compute_vector_size(type_, hdr_.dim));
    assert(norm_offset_in_record_ ==
        compute_data_record_layout(type_, hdr_.dim, data_file_has_norms(hdr_)).norm_offset);
    assert(stride_ == hdr_.vector_stride);
    assert(data_file_has_valid_norm_flags(hdr_));
    assert(stride_ >= vector_size_);
    assert((hdr_.data_offset % static_cast<uint64_t>(kDataRegionAlignment)) == 0);
    assert((stride_ % kDataAlignment) == 0);
    assert(ids_.count() == hdr_.count);
    assert(deleted_ids_.count() == hdr_.deleted_count);
    assert(!ids_region_.empty());
    assert(!deleted_ids_region_.empty());

    const DataMetadataLayout metadata_layout = compute_data_metadata_layout(hdr_, hdr_.count);
    assert(vectors_region_.size() == metadata_layout.vectors_bytes);
    assert(hdr_.vectors_bytes == metadata_layout.vectors_bytes);
    assert(hdr_.ids_offset == metadata_layout.ids_trailer_offset);
    assert(hdr_.ids_bytes == ids_region_.size());
    assert(hdr_.deleted_ids_offset ==
        static_cast<uint64_t>(compute_deleted_ids_offset(hdr_.ids_offset, hdr_.ids_bytes)));
    assert(hdr_.deleted_ids_bytes == deleted_ids_region_.size());

    if (data_file_has_norms(hdr_)) {
        assert(stride_ >= norm_offset_in_record_ + sizeof(float));
    }

    if (delta_) {
        assert(changed_bitset_.size() == hdr_.count);
        assert(has_hidden_rows_ == changed_bitset_.any());
        assert(type_ == delta_->type());
        assert(vector_size_ == delta_->size());
        assert(stride_ == delta_->stride());
        assert(hdr_.min_range_id == delta_->hdr_.min_range_id);
        assert(norm_offset_in_record_ == delta_->norm_offset_in_record_);
        assert(norm_flags() == delta_->norm_flags());
    } else {
        assert(changed_bitset_.size() == 0);
        assert(!has_hidden_rows_);
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

uint64_t DataReader::min_range_id() const {
    if (!initialized_) {
        throw std::runtime_error("DataReader::min_range_id: reader is not initialized");
    }
    return hdr_.min_range_id;
}

uint32_t DataReader::norm_flags() const {
    if (!initialized_) {
        throw std::runtime_error("DataReader::norm_flags: reader is not initialized");
    }
    return data_file_norm_flags(hdr_);
}

bool DataReader::has_norms() const {
    if (!initialized_) {
        throw std::runtime_error("DataReader::has_norms: reader is not initialized");
    }
    return data_file_has_norms(hdr_);
}

bool DataReader::has_matching_stored_norms(DistFunc dist_func) const {
    if (!initialized_) {
        throw std::runtime_error("DataReader::has_matching_stored_norms: reader is not initialized");
    }
    return data_file_matches_stored_norms_dist(hdr_, dist_func);
}

DataReader::Iterator DataReader::begin() const {
    return Iterator(this, delta_ ? delta_.get() : nullptr, first_visible_base_index_unchecked());
}

DataReader::OrderedIterator DataReader::base_begin() const {
    return OrderedIterator(this, OrderedIterator::Source::Base, first_visible_base_index_unchecked());
}

DataReader::OrderedIterator DataReader::delta_begin() const {
    return OrderedIterator(this, OrderedIterator::Source::Delta, 0);
}

uint64_t DataReader::id(size_t index) const {
    if (index >= count()) {
        throw std::out_of_range("DataReader::id: index out of range");
    }
    return id_unchecked(index);
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
