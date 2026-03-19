// Declares dataset metadata and the lean Dataset base class.

#pragma once
#include "core/compute/compute.h"
#include "utils/shared_consts.h"
#include "utils/shared_types.h"
#include <string>
#include <vector>

namespace sketch2 {

struct DatasetMetadata {
    std::vector<std::string> dirs;
    DataType type = DataType::f32;
    DistFunc dist_func = DistFunc::L1;
    uint64_t dim = 4;
    uint64_t range_size = kRangeSize;
    uint64_t data_merge_ratio = 2; // merge data files when the new file is less than
                                   // data_merge_ratio times smaller than the existing file
    uint64_t accumulator_size = kAccumulatorBufferSize;
};

struct DatasetItem {
    uint64_t id = 0;
    std::string data_file_path;
    std::string delta_file_path;
};

// Free functions used by dataset_reader.cpp and dataset_writer.cpp.
Ret collect_dataset_items(const DatasetMetadata& metadata, std::vector<DatasetItem>* items);
std::string dataset_owner_lock_path(const DatasetMetadata& metadata);

// Dataset is the lean base class: metadata storage and init only.
// Read infrastructure lives in DatasetReader; write infrastructure in DatasetWriter.
class Dataset {
public:
    Dataset() = default;
    virtual ~Dataset() = default;

    Ret init(const DatasetMetadata& metadata);

    Ret init(const std::vector<std::string>& dirs, uint64_t range_size,
        DataType type = DataType::f32, uint64_t dim = 4,
        uint64_t accumulator_size = kAccumulatorBufferSize,
        DistFunc dist_func = DistFunc::L1);

    // Initialize with values from ini file.
    Ret init(const std::string& path);

    DataType type() const { return metadata_.type; }
    DistFunc dist_func() const { return metadata_.dist_func; }
    uint64_t dim() const { return metadata_.dim; }
    uint64_t range_size() const { return metadata_.range_size; }
    const std::vector<std::string>& dirs() const { return metadata_.dirs; }

protected:
    DatasetMetadata metadata_;
    std::string item_path_base(uint64_t file_id) const;

private:
    Ret init_(const std::string& path);
};

} // namespace sketch2
