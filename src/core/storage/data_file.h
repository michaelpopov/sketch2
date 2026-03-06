#pragma once
#include <cstdint>

namespace sketch2 {

static constexpr uint32_t kMagic   = 0x534B5632; // "SKV2"
static constexpr uint16_t kVersion = 2;
static constexpr uint32_t kDataAlignment = 32;

enum class FileType : uint16_t {
    Data,
};

struct BaseFileHeader {
    uint32_t magic;
    uint16_t kind;     // file type
    uint16_t version;  // file format version
};

struct DataFileHeader {
    BaseFileHeader base;
    uint64_t min_id;
    uint64_t max_id;
    uint32_t count;
    uint32_t deleted_count;
    uint16_t type;     // data type
    uint16_t dim;
    uint32_t data_offset; // offset from file start to vectors section
};

} // namespace sketch2
