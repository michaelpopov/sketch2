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
    uint64_t data_offset; // offset from file start to the aligned vector-record region
    uint64_t vectors_bytes; // total bytes in the vector-record region
    uint32_t vector_stride; // bytes between consecutive persisted records, including inline norm/padding
    uint32_t flags; // optional per-record metadata flags, e.g. stored inline norms
    uint64_t ids_offset; // offset from file start to active ids trailer
    uint64_t ids_bytes; // size of active ids section
    uint64_t deleted_ids_offset; // offset from file start to deleted ids section
    uint64_t deleted_ids_bytes; // size of deleted ids section
    uint64_t min_range_id;
};

static_assert(sizeof(DataFileHeader) == 104, "Unexpected DataFileHeader size");

// Data file payload contract (v13):
// 1) aligned vector records with optional inline norm
// 2) region-alignment padding
// 3) frozen RoaringIds(active ids), omitted when count is zero
// 4) region-alignment padding
// 5) frozen RoaringIds(deleted ids), omitted when deleted_count is zero

} // namespace sketch2
