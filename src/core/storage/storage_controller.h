#pragma once
#include "utils/shared_types.h"
#include <string>
#include <vector>

namespace sketch2 {

class StorageController {
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
    bool valid_data_file(const std::string& output_path);
    bool check_data_file_merge(const std::string& data_path, const std::string& output_path);
    Ret  merge_data_file(const std::string& data_path, const std::string& output_path);
    Ret  merge_delta_file(const std::string& delta_path, const std::string& output_path);
    bool check_data_delta_merge(const std::string& data_path, const std::string& delta_path);
    Ret  merge_data_delta_file(const std::string& data_path, const std::string& delta_path);
    bool check_merge(const std::string& source_path, const std::string& update_path, uint64_t ratio);
};

} // namespace sketch2
