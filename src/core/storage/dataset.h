#pragma once
#include "utils/shared_types.h"
#include <memory>
#include <string>
#include <vector>

namespace sketch2 {

constexpr const char* kMetadataFileName = "sketch2.metadata";

class DataReader;
class InputReader;
class DatasetReader;

using DataReaderPtr = std::unique_ptr<DataReader>;
using DatasetReaderPtr = std::unique_ptr<DatasetReader>;

struct DatasetMetadata {
    std::vector<std::string> dirs;
    DataType type = DataType::f32;
    uint64_t dim = 4;
    uint64_t range_size = 10'000;
    uint64_t data_merge_ratio = 2; // merge data files when the new file is less than
                                   // data_merge_ratio times smaller than the existing file
};

class Dataset {
public:
    Ret init(const DatasetMetadata& metadata);

    // Initialize directly with a list of directories and id-range size.
    // Vectors with id in [file_id*range_size, (file_id+1)*range_size) go to file <file_id>.data
    // placed in directory dirs[file_id % dirs.size()].
    Ret init(const std::vector<std::string>& dirs, uint64_t range_size,
        DataType type = DataType::f32, uint64_t dim = 4);

    // Initialize with values from ini file.
    Ret init(const std::string& path);

    // Read input_path with InputReader, split by id range, and write one
    // data file per range using DataWriter.
    Ret store(const std::string& input_path);

    DatasetReaderPtr reader() const;

    DataType type() const { return metadata_.type; }
    uint16_t dim() const { return metadata_.dim; }

private:
    DatasetMetadata metadata_;

    Ret init_(const std::string& path);

    Ret store_(const std::string& input_path);
    Ret store_and_merge(const InputReader& reader, uint64_t file_id, uint64_t range_start, uint64_t range_end);

    bool check_data_file_merge(const DataReader& data_reader, const DataReader& output_reader);
    bool check_data_delta_merge(const DataReader& data_reader, const DataReader& delta_reader);

    Ret  merge_data_file(const DataReader& data_reader, const DataReader& output_reader,
        const std::string& output_path_base, const std::string& ext);
    Ret  merge_delta_file(const DataReader& delta_reader, const DataReader& output_reader,
        const std::string& output_path_base);
};

class DatasetReader {
public:
    Ret init(const std::vector<std::string>& dirs);
    std::pair<DataReaderPtr, Ret> next();
private:
    struct Item {
        uint64_t id;
        std::string data_file_path;
        std::string delta_file_path;
    };

private:
    std::vector<std::string> dirs_;
    std::vector<Item> items_;
    int current_ = -1;
};

} // namespace sketch2
