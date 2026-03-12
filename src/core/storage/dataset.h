#pragma once
#include "accumulator.h"
#include "core/compute/compute.h"
#include "core/utils/file_lock.h"
#include "utils/shared_consts.h"
#include "utils/shared_types.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sketch2 {

class DataReader;
class InputReader;
class DatasetReader;
using DataReaderPtr = std::shared_ptr<DataReader>;
using DatasetReaderPtr = std::unique_ptr<DatasetReader>;

enum class DatasetMode {
    Owner,
    Guest,
};

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

class Dataset {
public:
    class AccumulatorIterator {
    public:
        void next();
        bool eof() const;
        uint64_t id() const;
        const uint8_t* data() const;

    private:
        friend class Dataset;
        explicit AccumulatorIterator(Accumulator::Iterator iterator)
            : iterator_(std::move(iterator)) {}

        Accumulator::Iterator iterator_;
    };

    Dataset() = default;
    ~Dataset();

    Ret init(const DatasetMetadata& metadata);

    // Initialize directly with a list of directories and id-range size.
    // Vectors with id in [file_id*range_size, (file_id+1)*range_size) go to file <file_id>.data
    // placed in directory dirs[file_id % dirs.size()].
    Ret init(const std::vector<std::string>& dirs, uint64_t range_size,
        DataType type = DataType::f32, uint64_t dim = 4,
        uint64_t accumulator_size = kAccumulatorBufferSize,
        DistFunc dist_func = DistFunc::L1);

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
    DistFunc dist_func() const { return metadata_.dist_func; }
    uint64_t dim() const { return metadata_.dim; }
    uint64_t range_size() const { return metadata_.range_size; }
    const std::vector<std::string>& dirs() const { return metadata_.dirs; }

    Ret add_vector(uint64_t id, const uint8_t* data);
    Ret delete_vector(uint64_t id);
    bool is_deleted(uint64_t id) const;
    AccumulatorIterator accumulator_begin() const;
    size_t accumulator_vectors_count() const { return accumulator_ ? accumulator_->vectors_count() : 0; }
    size_t accumulator_deleted_count() const { return accumulator_ ? accumulator_->deleted_count() : 0; }

private:
    DatasetMetadata metadata_;
    DatasetMode mode_ = DatasetMode::Owner;
    std::unique_ptr<FileLockGuard> owner_lock_;
    std::unique_ptr<Accumulator> accumulator_;
    mutable bool items_cache_valid_ = false;
    mutable std::vector<DatasetItem> items_cache_;
    mutable std::unordered_map<uint64_t, DataReaderPtr> reader_cache_;

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
    Ret ensure_items_cache_() const;
    const DatasetItem* find_item_(uint64_t file_id) const;
    std::pair<DataReaderPtr, Ret> get_cached_reader_(const DatasetItem& item) const;
    void invalidate_data_caches_();

    friend class DatasetReader;
};

class DatasetReader {
public:
    Ret init(const Dataset* dataset, std::vector<DatasetItem> items);
    std::pair<DataReaderPtr, Ret> next();

    // Get a DataReader that accesses a file with a range containing id.
    std::pair<DataReaderPtr, Ret> get(uint64_t id);

private:
    const Dataset* dataset_ = nullptr;
    std::vector<DatasetItem> items_;
    int current_ = -1;
};

} // namespace sketch2
