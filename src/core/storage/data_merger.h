#pragma once
#include "data_reader.h"
#include <string>
#include <unordered_set>
#include <vector>

namespace sketch2 {

class DataMerger {
public:
    Ret init(const std::string& source_path, const std::string& update_path);
    Ret merge_data_files(const std::string& path);
    Ret merge_delta_file(const std::string& path);
    Ret merge_data_delta_files(const std::string& path);

private:
    struct Item {
        uint64_t id;
        const uint8_t* data;
    };

private:
    DataReader source_;
    DataReader updater_;

    std::unordered_set<uint64_t> deletes_;
    std::vector<Item> updater_items_;

    void load_update_records();
    void load_update_deletes();
    void write_data(FILE* f, const uint8_t* data);

    Ret merge_data_files_(const std::string& path);
};

} // namespace sketch2
