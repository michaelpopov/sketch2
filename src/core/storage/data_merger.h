// Declares the merge API for combining persisted storage files.

#pragma once
#include "data_reader.h"
#include <string>
#include <vector>

namespace sketch2 {

class InputReaderView;

// DataMerger helps compact range data by merging persisted data files and delta
// files while preserving deletes and optional cosine metadata.
class DataMerger {
public:
    Ret merge_data_file(const DataReader& source, const DataReader& updater, const std::string& path);
    Ret merge_delta_file(const DataReader& source, const DataReader& updater, const std::string& path);
    Ret merge_data_file(const DataReader& source, const InputReaderView& updater, const std::string& path);
    Ret merge_delta_file(const DataReader& source, const InputReaderView& updater, const std::string& path);

private:
    Ret merge_data_file_(const DataReader& source, const DataReader& updater, const std::string& path);
    Ret merge_delta_file_(const DataReader& source, const DataReader& updater, const std::string& path);
    Ret merge_data_file_(const DataReader& source, const InputReaderView& updater, const std::string& path);
    Ret merge_delta_file_(const DataReader& source, const InputReaderView& updater, const std::string& path);
};

} // namespace sketch2
