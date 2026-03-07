#pragma once
#include "accumulator.h"
#include "data_reader.h"
#include <string>
#include <vector>

namespace sketch2 {

class DataMerger {
public:
    Ret merge_data_file(const DataReader& source, const DataReader& updater, const std::string& path);
    Ret merge_data_file(
        const DataReader& source, const Accumulator& updater,
        const std::vector<uint64_t>& ids, const std::vector<uint64_t>& deleted_ids,
        const std::string& path);
    Ret merge_delta_file(const DataReader& source, const DataReader& updater, const std::string& path);
    Ret merge_delta_file(
        const DataReader& source, const Accumulator& updater,
        const std::vector<uint64_t>& ids, const std::vector<uint64_t>& deleted_ids,
        const std::string& path);

private:
    Ret merge_data_file_(const DataReader& source, const DataReader& updater, const std::string& path);
    Ret merge_data_file_(
        const DataReader& source, const Accumulator& updater,
        const std::vector<uint64_t>& ids, const std::vector<uint64_t>& deleted_ids,
        const std::string& path);
    Ret merge_delta_file_(const DataReader& source, const DataReader& updater, const std::string& path);
    Ret merge_delta_file_(
        const DataReader& source, const Accumulator& updater,
        const std::vector<uint64_t>& ids, const std::vector<uint64_t>& deleted_ids,
        const std::string& path);
};

} // namespace sketch2
