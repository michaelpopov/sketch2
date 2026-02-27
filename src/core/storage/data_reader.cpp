#include "data_reader.h"
#include <algorithm>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cassert>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace sketch2 {

// --- Iterator ---

DataReader::Iterator::Iterator(const DataReader* reader, size_t index, const uint64_t* ids)
    : reader_(reader), index_(index), ids_(ids) {}

void DataReader::Iterator::next() {
    ++index_;
    while (index_ < reader_->count() && reader_->is_deleted(index_)) {
        ++index_;
    }
}

bool DataReader::Iterator::eof() const {
    return index_ >= reader_->count();
}

const uint8_t* DataReader::Iterator::data() const {
    return reader_->at(index_);
}

uint64_t DataReader::Iterator::id() const {
    return ids_[index_];
}

// --- DataReader ---

DataReader::~DataReader() {
    if (map_) {
        munmap(const_cast<uint8_t*>(map_), map_len_);
    }
}

Ret DataReader::init(const std::string &path, ReaderMode mode, std::unique_ptr<DataReader> delta) {
    try {
        return init_(path, mode, std::move(delta));
    } catch (const std::exception& e) {
        return Ret(e.what());
    }
}

Ret DataReader::init_(const std::string& path, ReaderMode mode, std::unique_ptr<DataReader> delta) {
    if (map_) {
        return Ret("DataReader is initialized already.");
    }

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
    void* m = mmap(nullptr, map_len_, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) {
        return Ret("DataReader: failed to mmap file: " + path);
    }
    madvise(m, map_len_, MADV_SEQUENTIAL);
    madvise(m, map_len_, MADV_WILLNEED);
    map_ = static_cast<uint8_t*>(m);
    auto fail = [this](const std::string& message) -> Ret {
        munmap(const_cast<uint8_t*>(map_), map_len_);
        map_ = nullptr;
        map_len_ = 0;
        hdr_ = nullptr;
        ids_ = nullptr;
        size_ = 0;
        return Ret(message);
    };

    hdr_ = reinterpret_cast<const DataFileHeader *>(map_);
    if (hdr_->magic != kMagic) return fail("DataReader: invalid magic number");
    if (hdr_->kind != static_cast<uint16_t>(FileType::Data)) return fail("DataReader: not a data file");
    if (hdr_->version != kVersion) return fail("DataReader: unsupported file version");

    type_ = data_type_from_int(hdr_->type);
    validate_type(type_);
    const size_t elem_size = data_type_size(type_);
    if (elem_size == 0) {
        return fail("DataReader: invalid element type size");
    }

    const size_t dim = static_cast<size_t>(hdr_->dim);
    size_ = dim * elem_size;
    const size_t count = static_cast<size_t>(hdr_->count);
    const size_t deleted_count = static_cast<size_t>(hdr_->deleted_count);

    const size_t vectors_bytes = count * size_;
    const size_t ids_bytes = (deleted_count + count) * sizeof(uint64_t);
    if (map_len_ != sizeof(DataFileHeader) + vectors_bytes + ids_bytes) {
        return fail("DataReader: truncated or malformed data file");
    }

    if (delta) {
        if (!delta->hdr_) return fail("DataReader: invalid delta");
        if (type_ != delta->type()) return fail("DataReader: invalid delta type");
        if (size_ != delta->size()) return fail("DataReader: invalid delta dim");
    }

    ids_ = reinterpret_cast<const uint64_t*>(map_ + sizeof(DataFileHeader) + vectors_bytes);
    deleted_ids_ = ids_ + count;
    mode_   = mode;
    delta_  = std::move(delta);

    if (delta_) {
        CHECK(init_deleted_bitset());
        CHECK(init_updates_override());
    }

    return Ret(0);
}

Ret DataReader::init_deleted_bitset() {
    if (!hdr_ || !ids_ || !delta_) {
        return Ret("DataReader::init_deleted_bitset: reader is not initialized");
    }

    const uint64_t* del_ids = delta_->deleted_ids_;
    size_t del_count = delta_->deleted_count();

    bitset_.resize(hdr_->count);

    const void* zero = nullptr;
 
    for (size_t i = 0, j = 0; i < hdr_->count; ++i) {
        const uint64_t id = ids_[i];
        while (j < del_count && del_ids[j] < id) {
            ++j;
        }

        if (j >= del_count) {
            break;
        }

        if (del_ids[j] == id) {
            bitset_[i] = true;
            if (mode_ == ReaderMode::Reference) {
                assert(size_ >= sizeof(void*));
                uint8_t* vec_ptr = map_ + sizeof(DataFileHeader) + i * size_;
                memcpy(vec_ptr, &zero, sizeof(zero));
            }
        }
    }

    return Ret(0);
}

Ret DataReader::init_updates_override() {
    if (!hdr_ || !ids_ || !delta_) {
        return Ret("DataReader::init_updates_override: reader is not initialized");
    }

    const uint64_t* upd_ids = delta_->ids_;
    size_t upd_count = delta_->count();

    for (size_t i = 0, j = 0; i < hdr_->count; ++i) {
        const uint64_t id = ids_[i];
        while (j < upd_count && upd_ids[j] < id) {
            ++j;
        }

        if (j >= upd_count) {
            break;
        }
        
        if (upd_ids[j] == id) {
            uint8_t* vec_ptr = map_ + sizeof(DataFileHeader) + i * size_;
            const uint8_t* upd_ptr = delta_->at(j);
            if (mode_ == ReaderMode::InPlace) {
                memcpy(vec_ptr, upd_ptr, size_);
            } else if (mode_ == ReaderMode::Reference) {
                assert(size_ >= sizeof(upd_ptr));
                bitset_[i] = true;
                memcpy(vec_ptr, &upd_ptr, sizeof(upd_ptr));
            }
        }
    }

    return Ret(0);
}

DataType DataReader::type() const {
    return type_;
}

size_t DataReader::dim() const {
    return hdr_->dim;
}

size_t DataReader::size() const {
    return size_;
}

size_t DataReader::count() const {
    return hdr_->count;
}

DataReader::Iterator DataReader::begin() const {
    size_t index = 0;
    while (index < count() && is_deleted(index)) {
        ++index;
    }
    return Iterator(this, index, ids_);
}

uint64_t DataReader::id(size_t index) const {
    if (index >= count()) {
        throw std::out_of_range("DataReader::id: index out of range");
    }
    return ids_[index];
}

const uint8_t* DataReader::at(size_t index) const {
    if (index >= count()) {
        throw std::out_of_range("DataReader::at: index out of range");
    }

    // In case of ReaderMode::InPlace always return the mmaped vector.
    if (mode_ == ReaderMode::InPlace) {
        return map_ + sizeof(DataFileHeader) + index * size();
    }

    // In case of ReaderMode::Reference check is a vector overwritten.
    // If not, then return the mmaped vector.
    if (index >= bitset_.size() || !bitset_[index]) {
        return map_ + sizeof(DataFileHeader) + index * size();
    }

    // In case of ReaderMode::Reference when a vector is overwritten
    // get the pointer to a new value and return it.
    const uint8_t* vec = map_ + sizeof(DataFileHeader) + index * size();
    const uint8_t* ptr;
    memcpy(&ptr, vec, sizeof(ptr));
    return ptr;
}

const uint8_t* DataReader::get(uint64_t id) const {
    const size_t n = count();
    const uint64_t* first = ids_;
    const uint64_t* last  = ids_ + n;
    const uint64_t* it    = std::lower_bound(first, last, id);
    if (it == last || *it != id) {
        return nullptr;
    }
    const size_t index = static_cast<size_t>(it - first);
    return is_deleted(index) ? nullptr : at(index);
}

bool DataReader::is_deleted(size_t index) const {
    if (index >= bitset_.size() || !bitset_[index]) {
        return false;
    }
    if (mode_ == ReaderMode::InPlace) {
        return true;
    }
    // Reference mode: deleted if the first 8 bytes (a pointer) are null
    const uint8_t* vec = map_ + sizeof(DataFileHeader) + index * size();
    const uint8_t* ptr;
    memcpy(&ptr, vec, sizeof(ptr));
    return ptr == nullptr;
}

uint64_t DataReader::deleted_id(size_t index) const {
    if (index >= deleted_count()) {
        throw std::out_of_range("DataReader::deleted_id: index out of range");
    }
    return deleted_ids_[index];
}

bool DataReader::check_consistency() const {
    if (!hdr_) {
        return false;
    }

    size_t i = 0;
    size_t j = 0;
    const size_t ids_count = count();
    const size_t deleted_count_ = deleted_count();

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
