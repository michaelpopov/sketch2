#pragma once
#include "data_reader.h"
#include <string>
#include <unordered_set>
#include <vector>

namespace sketch2 {

class DataMerger {
public:
    Ret merge_data_file(const DataReader& source, const DataReader& updater, const std::string& path);
    Ret merge_delta_file(const DataReader& source, const DataReader& updater, const std::string& path);

private:
    struct Item {
        uint64_t id;
        const uint8_t* data;
    };

    void load_update_records(const DataReader& updater, std::vector<Item>& updater_items);
    void load_update_deletes(const DataReader& updater, std::unordered_set<uint64_t>& deletes);
    void write_data(FILE* f, const uint8_t* data, size_t size);

    Ret merge_data_file_(const DataReader& source, const DataReader& updater, const std::string& path);
    Ret merge_delta_file_(const DataReader& source, const DataReader& updater, const std::string& path);
};

} // namespace sketch2
