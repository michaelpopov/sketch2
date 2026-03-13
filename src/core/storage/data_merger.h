// Declares the merge API for combining persisted and buffered storage data.

#pragma once
#include "accumulator.h"
#include "data_reader.h"
#include <string>
#include <vector>

namespace sketch2 {

// DataMerger exists to reconcile sorted base data, delta files, and buffered
// accumulator updates into new on-disk files. It provides the merge operations
// that keep persisted storage compact while preserving deletes and replacements.
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
