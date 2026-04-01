// Implements a buffered writer for indexed-binary input files.

#include "input_writer.h"

#include "core/utils/shared_consts.h"
#include "core/utils/string_utils.h"

#include <cstring>
#include <stdexcept>

namespace sketch2 {

namespace {

uint32_t crc32_update(uint32_t crc, const uint8_t* data, size_t size) {
    crc = ~crc;
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            const uint32_t mask = -(crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

} // namespace

InputWriter::~InputWriter() {
    (void)close_file();
}

Ret InputWriter::init(DataType type, size_t dim, const std::string& path) {
    try {
        return init_(type, dim, path);
    } catch (const std::exception& e) {
        return Ret(e.what());
    }
}

Ret InputWriter::init_(DataType type, size_t dim, const std::string& path) {
    if (file_ != nullptr) {
        return Ret("InputWriter is initialized already");
    }
    if (dim < kMinDimension || dim > kMaxDimension) {
        return Ret("InputWriter: invalid dimension");
    }

    type_ = type;
    dim_ = dim;
    vector_size_ = dim_ * data_type_size(type_);
    if (vector_size_ == 0) {
        return Ret("InputWriter: invalid data type");
    }

    const size_t max_block_bytes =
        sizeof(uint64_t) + kIndexedBinaryBlockItems * (sizeof(uint64_t) + vector_size_) + sizeof(IndexedBlockFooter);
    block_buffer_.assign(max_block_bytes, 0);
    file_buffer_.assign(max_block_bytes, 0);
    vector_buffer_.assign(vector_size_, 0);
    block_used_ = sizeof(uint64_t); // reserve bitset at the start of the block
    block_items_ = 0;
    total_items_ = 0;
    block_bitset_ = 0;

    file_ = std::fopen(path.c_str(), "wb");
    if (file_ == nullptr) {
        return Ret("InputWriter: failed to open file");
    }
    if (setvbuf(file_, reinterpret_cast<char*>(file_buffer_.data()), _IOFBF, file_buffer_.size()) != 0) {
        std::fclose(file_);
        file_ = nullptr;
        return Ret("InputWriter: failed to configure file buffer");
    }

    if (std::fprintf(file_, "%s,%zu,%s\n", data_type_to_string(type_), dim_, BinIndexedFileMarker) < 0) {
        std::fclose(file_);
        file_ = nullptr;
        return Ret("InputWriter: failed to write header");
    }

    return Ret(0);
}

Ret InputWriter::append_record_header(uint64_t id, bool deleted) {
    if (file_ == nullptr) {
        return Ret("InputWriter: file is not initialized");
    }
    if (block_items_ >= kIndexedBinaryBlockItems) {
        return Ret("InputWriter: block is full");
    }

    if (deleted) {
        block_bitset_ |= (uint64_t{1} << block_items_);
    }

    std::memcpy(block_buffer_.data() + block_used_, &id, sizeof(id));
    block_used_ += sizeof(id);
    ++block_items_;
    ++total_items_;
    return Ret(0);
}

Ret InputWriter::write_vector(uint64_t id, const char* vector) {
    if (vector == nullptr) {
        return Ret("InputWriter::write_vector: vector is null");
    }

    const bool comma_delimited = check_comma_format(vector);
    const Ret parse_ret = comma_delimited
        ? parse_vector(vector_buffer_.data(), vector_buffer_.size(), type_, static_cast<uint16_t>(dim_), vector)
        : parse_vector_spaces(vector_buffer_.data(), vector_buffer_.size(), type_, static_cast<uint16_t>(dim_), vector);
    if (parse_ret.code() != 0) {
        return parse_ret;
    }

    CHECK(append_record_header(id, false));
    std::memcpy(block_buffer_.data() + block_used_, vector_buffer_.data(), vector_size_);
    block_used_ += vector_size_;
    return finish_record();
}

Ret InputWriter::write_deleted(uint64_t id) {
    CHECK(append_record_header(id, true));
    return finish_record();
}

Ret InputWriter::finish_record() {
    if (block_items_ == kIndexedBinaryBlockItems) {
        return flush_block(true);
    }
    return Ret(0);
}

Ret InputWriter::flush_block(bool write_footer) {
    if (file_ == nullptr) {
        return Ret("InputWriter: file is not initialized");
    }
    if (block_items_ == 0) {
        return Ret(0);
    }

    std::memcpy(block_buffer_.data(), &block_bitset_, sizeof(block_bitset_));
    size_t bytes_to_write = block_used_;
    if (write_footer) {
        IndexedBlockFooter footer{
            static_cast<uint32_t>(total_items_),
            crc32_update(0, block_buffer_.data(), block_used_)
        };
        std::memcpy(block_buffer_.data() + block_used_, &footer, sizeof(footer));
        bytes_to_write += sizeof(footer);
    }

    if (std::fwrite(block_buffer_.data(), 1, bytes_to_write, file_) != bytes_to_write) {
        return Ret("InputWriter: failed to write block");
    }
    if (std::fflush(file_) != 0) {
        return Ret("InputWriter: failed to flush file");
    }

    block_used_ = sizeof(uint64_t);
    block_items_ = 0;
    block_bitset_ = 0;
    return Ret(0);
}

Ret InputWriter::close_file() {
    if (file_ == nullptr) {
        return Ret(0);
    }

    Ret flush_ret = flush_block(false);
    const int close_rc = std::fclose(file_);
    file_ = nullptr;
    if (flush_ret.code() != 0) {
        return flush_ret;
    }
    if (close_rc != 0) {
        return Ret("InputWriter: failed to close file");
    }

    return Ret(0);
}

} // namespace sketch2
