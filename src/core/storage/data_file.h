#pragma once
#include "utils/shared_consts.h"
#include <cstdint>
#include <cstddef>

namespace sketch2 {

template <typename T>
constexpr T align_up(T value, T alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

enum class FileType : uint16_t {
    Data,
    Wal,
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

struct WalFileHeader {
    BaseFileHeader base;
    uint16_t type;     // data type
    uint16_t dim;
    uint32_t reserved;
};

struct WalRecordHeader {
    uint32_t size;     // full record size including header
    uint8_t op;
    uint8_t reserved[3];
    uint64_t id;
    uint32_t checksum;
    uint32_t reserved2;
};

} // namespace sketch2
