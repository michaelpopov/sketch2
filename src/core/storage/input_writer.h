// Declares a writer for indexed-binary input files.

#pragma once

#include "utils/shared_types.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sketch2 {

// InputWriter exists to produce indexed-binary import files in the same format
// consumed by InputReader. It buffers up to one 64-item block in memory,
// writes a footer for full blocks, and leaves a trailing partial block
// footerless as defined by the format.
class InputWriter {
public:
    ~InputWriter();

    Ret init(DataType type, size_t dim, const std::string& path);
    Ret write_vector(uint64_t id, const char* vector);
    Ret write_deleted(uint64_t id);
    Ret close_file();

private:
    int fd_ = -1;
    DataType type_ = DataType::f32;
    size_t dim_ = 0;
    size_t vector_size_ = 0;
    size_t block_used_ = 0;
    size_t block_items_ = 0;
    size_t total_items_ = 0;
    uint64_t block_bitset_ = 0;
    std::vector<uint8_t> block_buffer_;
    std::vector<uint8_t> vector_buffer_;

    Ret init_(DataType type, size_t dim, const std::string& path);
    Ret append_record_header(uint64_t id, bool deleted);
    Ret finish_record();
    Ret flush_block(bool write_footer);
};

} // namespace sketch2
