// Implements an indexed-binary input writer.

#include "input_writer.h"

#include "core/utils/shared_consts.h"
#include "core/utils/string_utils.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <unistd.h>

namespace sketch2 {

namespace {

Ret write_all(int fd, const void* data, size_t size, const char* context) {
    const auto* ptr = static_cast<const uint8_t*>(data);
    size_t written = 0;
    while (written < size) {
        const ssize_t rc = ::write(fd, ptr + written, size - written);
        if (rc < 0) {
            if (errno == EINTR) {
                continue;
            }
            return Ret(std::string(context) + ": " + std::strerror(errno));
        }
        if (rc == 0) {
            return Ret(std::string(context) + ": short write");
        }
        written += static_cast<size_t>(rc);
    }
    return Ret(0);
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
    if (fd_ >= 0) {
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
    vector_buffer_.assign(vector_size_, 0);
    block_used_ = sizeof(uint64_t); // reserve bitset at the start of the block
    block_items_ = 0;
    total_items_ = 0;
    block_bitset_ = 0;

    fd_ = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd_ < 0) {
        return Ret("InputWriter: failed to open file");
    }

    char header[64];
    const int header_len = std::snprintf(
        header, sizeof(header), "%s,%zu,%s\n", data_type_to_string(type_), dim_, BinIndexedFileMarker);
    if (header_len <= 0 || static_cast<size_t>(header_len) >= sizeof(header)) {
        (void)::close(fd_);
        fd_ = -1;
        return Ret("InputWriter: failed to format header");
    }
    const Ret header_ret = write_all(fd_, header, static_cast<size_t>(header_len), "InputWriter: failed to write header");
    if (header_ret.code() != 0) {
        (void)::close(fd_);
        fd_ = -1;
        return header_ret;
    }

    return Ret(0);
}

Ret InputWriter::append_record_header(uint64_t id, bool deleted) {
    if (fd_ < 0) {
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

    if (!comma_delimited_detected_) {
        comma_delimited_detected_ = true;
        comma_delimited_ = check_comma_format(vector);
    }

    const Ret parse_ret = comma_delimited_
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
    if (fd_ < 0) {
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

    const Ret write_ret = write_all(fd_, block_buffer_.data(), bytes_to_write, "InputWriter: failed to write block");
    if (write_ret.code() != 0) {
        return write_ret;
    }

    const int flush_rc = fdatasync(fd_);
    if (flush_rc != 0) {
        return Ret(std::string("InputWriter: failed to store data: ") + strerror(errno));
    }

    block_used_ = sizeof(uint64_t);
    block_items_ = 0;
    block_bitset_ = 0;
    return Ret(0);
}

Ret InputWriter::close_file() {
    if (fd_ < 0) {
        return Ret(0);
    }

    Ret flush_ret = flush_block(false);
    const int fsync_rc = ::fsync(fd_);
    const int close_rc = ::close(fd_);
    fd_ = -1;
    if (flush_ret.code() != 0) {
        return flush_ret;
    }
    if (fsync_rc != 0 || close_rc != 0) {
        return Ret("InputWriter: failed to close file");
    }

    return Ret(0);
}

} // namespace sketch2
