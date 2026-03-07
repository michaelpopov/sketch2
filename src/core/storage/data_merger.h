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
    struct Item {
        uint64_t id;
        const uint8_t* data;
    };

    void load_update_records(const DataReader& updater, std::vector<Item>& updater_items);
    void load_update_deletes(const DataReader& updater, std::vector<uint64_t>& deletes);
    void write_data(FILE* f, const uint8_t* data, size_t size);

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
