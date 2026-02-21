#include "data_reader.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>

namespace sketch2 {

// --- helpers ---

static DataType from_file_data_type(uint16_t t) {
    switch (t) {
        case 1:  return DataType::f16;
        case 2:  return DataType::i32;
        default: return DataType::f32;
    }
}

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

Ret DataReader::init(const std::string& path, ReaderMode mode,
                     const std::vector<bool>* bitset) {
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
    hdr_ = reinterpret_cast<const DataFileHeader*>(map_);

    if (hdr_->magic != kMagic) {
        return Ret("DataReader: invalid magic number");
    }
    if (hdr_->kind != static_cast<uint16_t>(FileType::Data)) {
        return Ret("DataReader: not a data file");
    }

    type_   = from_file_data_type(hdr_->type);
    size_   = hdr_->dim * to_size(type_);
    ids_    = reinterpret_cast<const uint64_t*>(
        map_ + sizeof(DataFileHeader) + count() * size_);
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

const uint8_t* DataReader::at(size_t index) const {
    if (index >= count()) {
        throw std::out_of_range("DataReader::at: index out of range");
    }
    return map_ + sizeof(DataFileHeader) + index * size();
}

const uint8_t* DataReader::get(uint64_t id) const {
    for (size_t i = 0; i < count(); ++i) {
        if (ids_[i] == id) {
            return is_deleted(i) ? nullptr : at(i);
        }
    }
    return nullptr;
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

} // namespace sketch2
