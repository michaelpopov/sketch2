#pragma once
#include <cstdint>

namespace sketch2 {

enum class FileType : uint16_t {
    Data, Delta, DataCatalogue,
    // Sample, Residuals,
    // ClusterMap, IndexCatalogue, Centroids, PQCentroids, PQData,
};

enum class DataType : uint16_t {
    Float, Float16,
};

struct BaseFileHeader {
    uint32_t magic;
    uint16_t file_type;
    uint16_t format_version;
};

struct DataBaseHeader : public BaseFileHeader {
    uint32_t file_version;
    uint32_t file_index;
    uint16_t data_type;
    uint16_t dim;
    uint32_t count;
    uint64_t min_id;
    uint64_t max_id;
};

} // namespace sketch2
