#include "data_reader.h"
#include <algorithm>
#include <sys/mman.h>
#include <sys/stat.h>
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

Ret DataReader::init(const std::string &path, ReaderMode mode, const std::vector<bool> *bitset) {
    try {
        return init_(path, mode, bitset);
    } catch (const std::exception& e) {
        return Ret(e.what());
    }
}

Ret DataReader::init_(const std::string& path, ReaderMode mode,
                     const std::vector<bool>* bitset) {
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
    void* m = mmap(nullptr, map_len_, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) {
        return Ret("DataReader: failed to mmap file: " + path);
    }
    map_ = static_cast<const uint8_t*>(m);
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
    if (hdr_->magic != kMagic)
    {
        return fail("DataReader: invalid magic number");
    }
    if (hdr_->kind != static_cast<uint16_t>(FileType::Data)) {
        return fail("DataReader: not a data file");
    }
    if (hdr_->version != kVersion) {
        return fail("DataReader: unsupported file version");
    }

    type_ = data_type_from_int(hdr_->type);
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
        /********************************************************
        std::cout << "\n\n\n";
        std::cout << "hdr_->type=" << hdr_->type << "\n";
        std::cout << "elem_size=" << elem_size << "\n";
        std::cout << "dim=" << dim << "\n";
        std::cout << "size_=" << size_ << "\n";
        std::cout << "count=" << count << "\n";
        std::cout << "\n";
        std::cout << "map_len_=" << map_len_ << "\n";
        std::cout << "sizeof(DataFileHeader)=" << sizeof(DataFileHeader) << "\n";
        std::cout << "vectors_bytes=" << vectors_bytes << "\n";
        std::cout << "ids_bytes=" << ids_bytes << "\n";
        std::cout << "\n\n\n";
        *********************************************************/
        return fail("DataReader: truncated or malformed data file");
    }

    ids_ = reinterpret_cast<const uint64_t*>(map_ + sizeof(DataFileHeader) + vectors_bytes);
    deleted_ids_ = ids_ + count;
    mode_   = mode;
    bitset_ = bitset;

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
    return map_ + sizeof(DataFileHeader) + index * size();
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
    if (!bitset_ || index >= bitset_->size() || !(*bitset_)[index]) {
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

} // namespace sketch2
