#pragma once
#include <cstdint>

namespace sketch2 {

enum class FileType : uint16_t {
    Data, Delta,
};

enum class DataType : uint16_t {
    Float, Float16,
};

struct BaseFileHeader {
    uint32_t magic;
    uint16_t type;     // file type
    uint16_t version;  // file format version
};

struct DataFileHeader : public BaseFileHeader {
    uint64_t min_id;
    uint64_t max_id;
    uint32_t count;
    uint16_t data;     // data type
    uint16_t dim;
};

} // namespace sketch2
