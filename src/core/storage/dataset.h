#pragma once
#include "utils/shared_types.h"
#include <string>
#include <vector>

namespace sketch2 {

class DataReader;
class InputReader;

class Dataset {
public:
    // Initialize directly with a list of directories and id-range size.
    // Vectors with id in [file_id*range_size, (file_id+1)*range_size) go to file <file_id>.data
    // placed in directory dirs[file_id % dirs.size()].
    Ret init(const std::vector<std::string>& dirs, uint64_t range_size);

    // Read input_path with InputReader, split by id range, and write one
    // data file per range using DataWriter.
    Ret load(const std::string& input_path);

private:
    std::vector<std::string> dirs_;
    uint64_t range_size_ = 0;

    Ret load_(const std::string& input_path);
    Ret load_and_merge(const InputReader& reader, uint64_t file_id, uint64_t range_start, uint64_t range_end);

    bool check_data_file_merge(const DataReader& data_reader, const DataReader& output_reader);
    bool check_data_delta_merge(const DataReader& data_reader, const DataReader& delta_reader);

    Ret  merge_data_file(const DataReader& data_reader, const DataReader& output_reader,
        const std::string& output_path_base, const std::string& ext);
    Ret  merge_delta_file(const DataReader& delta_reader, const DataReader& output_reader,
        const std::string& output_path_base);
};

} // namespace sketch2
