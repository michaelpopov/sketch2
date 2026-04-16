// Defines the binary on-disk file headers and shared file-format helpers.

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
    uint32_t vectors_bytes; // total bytes in the vectors section
    uint32_t vector_stride; // bytes between consecutive persisted vectors, including padding
    uint32_t flags; // optional section flags, e.g. cosine inverse norms
    uint32_t cosine_inv_norms_offset; // offset from file start to the optional norms section
    uint32_t cosine_inv_norms_bytes; // size of the optional norms section
    uint32_t ids_offset; // offset from file start to active ids section
    uint32_t ids_bytes; // size of active ids section
    uint32_t deleted_ids_offset; // offset from file start to deleted ids section
    uint32_t deleted_ids_bytes; // size of deleted ids section
    uint32_t reserved = 0;
};

// Data file payload contract (v8):
// 1) aligned vector records
// 2) region-alignment padding
// 3) optional cosine inverse norms for active vectors
// 4) region-alignment padding
// 5) CompactIdsOffsets(active ids)
// 6) region-alignment padding
// 7) CompactIdsOffsets(deleted ids)

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
