#pragma once
#include "accumulator.h"
#include "core/utils/file_lock.h"
#include "utils/shared_types.h"
#include <memory>
#include <string>
#include <vector>

namespace sketch2 {

constexpr const char* kMetadataFileName = "sketch2.metadata";
static constexpr size_t kAccumulatorBufferSize = 64 * 1024;
static constexpr int kRangeSize = 10'000;

class DataReader;
class InputReader;
class DatasetReader;
using DataReaderPtr = std::unique_ptr<DataReader>;
using DatasetReaderPtr = std::unique_ptr<DatasetReader>;

enum class DatasetMode {
    Owner,
    Guest,
};

struct DatasetMetadata {
    std::vector<std::string> dirs;
    DataType type = DataType::f32;
    uint64_t dim = 4;
    uint64_t range_size = kRangeSize;
    uint64_t data_merge_ratio = 2; // merge data files when the new file is less than
                                   // data_merge_ratio times smaller than the existing file
    uint64_t accumulator_size = kAccumulatorBufferSize;
};

class Dataset {
public:
    Dataset() = default;
    ~Dataset();

    Ret init(const DatasetMetadata& metadata);

    // Initialize directly with a list of directories and id-range size.
    // Vectors with id in [file_id*range_size, (file_id+1)*range_size) go to file <file_id>.data
    // placed in directory dirs[file_id % dirs.size()].
    Ret init(const std::vector<std::string>& dirs, uint64_t range_size,
        DataType type = DataType::f32, uint64_t dim = 4,
        uint64_t accumulator_size = kAccumulatorBufferSize);

    // Initialize with values from ini file.
    Ret init(const std::string& path);
    Ret set_guest_mode();

    // Read input_path with InputReader, split by id range, and write one
    // data file per range using DataWriter.
    Ret store(const std::string& input_path);
    Ret store_accumulator();
    Ret merge();

    DatasetReaderPtr reader() const;
    std::pair<DataReaderPtr, Ret> get(uint64_t id) const;

    DataType type() const { return metadata_.type; }
    uint64_t dim() const { return metadata_.dim; }

    Ret add_vector(uint64_t id, const uint8_t* data);
    Ret delete_vector(uint64_t id);

private:
    DatasetMetadata metadata_;
    DatasetMode mode_ = DatasetMode::Owner;
    std::unique_ptr<FileLockGuard> owner_lock_;
    std::unique_ptr<Accumulator> accumulator_;

    Ret init_(const std::string& path);

    Ret store_(const std::string& input_path);
    Ret store_accumulator_();
    Ret merge_();
    Ret store_and_merge(const InputReader& reader, uint64_t file_id, uint64_t range_start, uint64_t range_end);
    Ret store_and_merge_accumulator(uint64_t file_id, const std::vector<uint64_t>& ids, const std::vector<uint64_t>& deleted_ids);
    Ret init_accumulator_();

    bool check_data_file_merge(const DataReader& data_reader, const DataReader& output_reader);
    bool check_data_delta_merge(const DataReader& data_reader, const DataReader& delta_reader);

    Ret  merge_data_file(const DataReader& data_reader, const DataReader& output_reader,
        const std::string& output_path_base, const std::string& ext);
    Ret  merge_delta_file(const DataReader& delta_reader, const DataReader& output_reader,
        const std::string& output_path_base);
    Ret require_owner_() const;
    Ret ensure_owner_lock_();
};

class DatasetReader {
public:
    struct Item {
        uint64_t id;
        std::string data_file_path;
        std::string delta_file_path;
    };

    Ret init(DatasetMetadata metadata);
    std::pair<DataReaderPtr, Ret> next();

    // Get a DataReader that accesses a file with a range containing id.
    std::pair<DataReaderPtr, Ret> get(uint64_t id);

private:
    DatasetMetadata metadata_;
    std::vector<Item> items_;
    int current_ = -1;

    Ret init_items_();
};

} // namespace sketch2
