#include "data_merger.h"
#include <algorithm>
#include <experimental/scope>
#include <filesystem>
#include <stdio.h>
#include <errno.h>
#include <string.h>

namespace sketch2 {

Ret DataMerger::merge_data_file(const DataReader& source, const DataReader& updater, const std::string& path) {
    if (source.dim() != updater.dim() || source.type() != updater.type()) {
        return Ret("DataMerger::merge_data_file: incompatible source and updater");
    }

    Ret ret(0);
    try {
        ret = merge_data_file_(source, updater, path);
    } catch (const std::exception& ex) {
        ret = Ret(ex.what());
    }

    if (ret != 0 && std::filesystem::exists(path)) {
        std::filesystem::remove(path);
    }

    return ret;
}

Ret DataMerger::merge_data_file_(const DataReader& source, const DataReader& updater, const std::string& path) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        return Ret(strerror(errno));
    }
    std::experimental::scope_exit file_guard([f]() { fclose(f); });

    DataFileHeader hdr;
    hdr.magic         = kMagic;
    hdr.kind          = static_cast<uint16_t>(FileType::Data);
    hdr.version       = kVersion;
    hdr.min_id        = 0;
    hdr.max_id        = 0;
    hdr.count         = 0;
    hdr.deleted_count = 0;
    hdr.type          = static_cast<uint16_t>(data_type_to_int(source.type()));
    hdr.dim           = static_cast<uint16_t>(source.dim());
    hdr.padding       = 0;

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        return Ret("DataMerger::merge_data_files: failed to write header");
    }

    std::vector<Item> updater_items;
    load_update_records(updater, updater_items);

    std::unordered_set<uint64_t> deletes;
    load_update_deletes(updater, deletes);

    std::vector<uint64_t> ids;
    ids.reserve(source.count() + updater_items.size());

    for (size_t i = 0, j = 0; i < source.count() || j < updater_items.size(); ) {
        const bool has_source = i < source.count();
        const bool has_update = j < updater_items.size();

        // Deletes can only mark deleted items in the source.
        // If an item is deleted, skip it and move forward.
        if (has_source) {
            const uint64_t sourceid = source.id(i);
            if (deletes.find(sourceid) != deletes.end()) {
                i++;
                continue;
            }
        }

        // Update items must not contain same id in the data list and in the
        // list of deleted items. If this happens, it is an error.
        if (has_update) {
            const uint64_t update_id = updater_items[j].id;
            if (deletes.find(update_id) != deletes.end()) {
                return Ret("DataMerger::merge_data_files: updated id is also deleted");
            }
        }

        if (has_source && has_update) {
            const uint64_t sourceid = source.id(i);
            const uint64_t update_id = updater_items[j].id;

            if (sourceid < update_id) {
                ids.push_back(sourceid);
                write_data(f, source.at(i), source.size());
                ++i;
            } else if (sourceid > update_id) {
                ids.push_back(update_id);
                write_data(f, updater_items[j].data, source.size());
                ++j;
            } else { // if sourceid == update_id, update_id wins
                ids.push_back(update_id);
                write_data(f, updater_items[j].data, source.size());
                ++i;
                ++j;
            }
            continue;
        }

        if (has_source) {
            const uint64_t sourceid = source.id(i);
            ids.push_back(sourceid);
            write_data(f, source.at(i), source.size());
            ++i;
        } else {
            const uint64_t update_id = updater_items[j].id;
            ids.push_back(update_id);
            write_data(f, updater_items[j].data, source.size());
            ++j;
        }
    }

    if (ids.empty()) {
        return Ret(0);
    }

    if (fwrite(ids.data(), sizeof(uint64_t), ids.size(), f) != ids.size()) {
        return Ret("DataMerger::merge_data_files: failed to write ids to merge file");
    }

    // Overwrite header with updated values.
    if (0 != fseek(f, 0, SEEK_SET)) {
        return Ret("DataMerger::merge_data_files: failed to rewind to header");
    }

    hdr.min_id = ids.front();
    hdr.max_id = ids.back();
    hdr.count  = static_cast<uint32_t>(ids.size());
    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        return Ret("DataMerger::merge_data_files: failed to write header");
    }

    return Ret(0);
}

void DataMerger::load_update_records(const DataReader& updater, std::vector<Item>& updater_items) {
    updater_items.reserve(updater.count());
    
    for (auto iter = updater.begin(); !iter.eof(); iter.next()) {
        Item item {
            .id = iter.id(),
            .data = iter.data(),
        };
        updater_items.push_back(item);
    }
}

void DataMerger::load_update_deletes(const DataReader& updater, std::unordered_set<uint64_t>& deletes) {
    for (size_t i = 0; i < updater.deleted_count(); i++) {
        deletes.insert(updater.deleted_id(i));
    }
}

void DataMerger::write_data(FILE* f, const uint8_t* data, size_t size) {
    if (fwrite(data, size, 1, f) != 1) {
        throw std::runtime_error("DataMerger::write_data: failed to write merge file");
    }
}

Ret DataMerger::merge_delta_file(const DataReader& source, const DataReader& updater, const std::string& path) {
    if (source.dim() != updater.dim() || source.type() != updater.type()) {
        return Ret("DataMerger::merge_delta_file: incompatible source and updater");
    }

    Ret ret(0);
    try {
        ret = merge_delta_file_(source, updater, path);
    } catch (const std::exception& ex) {
        ret = Ret(ex.what());
    }

    if (ret != 0 && std::filesystem::exists(path)) {
        std::filesystem::remove(path);
    }

    return ret;
}

Ret DataMerger::merge_delta_file_(const DataReader& source, const DataReader& updater, const std::string& path) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        return Ret(strerror(errno));
    }
    std::experimental::scope_exit file_guard([f]() { fclose(f); });

    DataFileHeader hdr;
    hdr.magic         = kMagic;
    hdr.kind          = static_cast<uint16_t>(FileType::Data);
    hdr.version       = kVersion;
    hdr.min_id        = 0;
    hdr.max_id        = 0;
    hdr.count         = 0;
    hdr.deleted_count = 0;
    hdr.type          = static_cast<uint16_t>(data_type_to_int(source.type()));
    hdr.dim           = static_cast<uint16_t>(source.dim());
    hdr.padding       = 0;

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        return Ret("DataMerger::merge_delta_file: failed to write header");
    }

    std::unordered_set<uint64_t> deletes;
    std::vector<Item> updater_items;

    load_update_records(updater, updater_items);

    // If updater contains records with ids that appear in deletes,
    // then remove these ids from deletes because updater brings new
    // values.
    load_update_deletes(source, deletes);
    for (const auto& item: updater_items) {
        deletes.erase(item.id);
    }

    load_update_deletes(updater, deletes);

    std::vector<uint64_t> ids;
    ids.reserve(source.count() + updater_items.size());

    for (size_t i = 0, j = 0; i < source.count() || j < updater_items.size(); ) {
        const bool has_source = i < source.count();
        const bool has_update = j < updater_items.size();

        // Deletes can only mark deleted items in the source.
        // If an item is deleted, skip it and move forward.
        if (has_source) {
            const uint64_t sourceid = source.id(i);
            if (deletes.find(sourceid) != deletes.end()) {
                i++;
                continue;
            }
        }

        // Update items must not contain same id in the data list and in the
        // list of deleted items. If this happens, it is an error.
        if (has_update) {
            const uint64_t update_id = updater_items[j].id;
            if (deletes.find(update_id) != deletes.end()) {
                return Ret("DataMerger::merge_delta_file: updated id is also deleted");
            }
        }

        if (has_source && has_update) {
            const uint64_t sourceid = source.id(i);
            const uint64_t update_id = updater_items[j].id;

            if (sourceid < update_id) {
                ids.push_back(sourceid);
                write_data(f, source.at(i), source.size());
                ++i;
            } else if (sourceid > update_id) {
                ids.push_back(update_id);
                write_data(f, updater_items[j].data, source.size());
                ++j;
            } else { // if sourceid == update_id, update_id wins
                ids.push_back(update_id);
                write_data(f, updater_items[j].data, source.size());
                ++i;
                ++j;
            }
            continue;
        }

        if (has_source) {
            const uint64_t sourceid = source.id(i);
            ids.push_back(sourceid);
            write_data(f, source.at(i), source.size());
            ++i;
        } else {
            const uint64_t update_id = updater_items[j].id;
            ids.push_back(update_id);
            write_data(f, updater_items[j].data, source.size());
            ++j;
        }
    }

    if (!ids.empty()) {
        if (fwrite(ids.data(), sizeof(uint64_t), ids.size(), f) != ids.size()) {
            return Ret("DataMerger::merge_delta_file: failed to write ids to merge file");
        }
    }

    if (!deletes.empty()) {
        std::vector<uint64_t> deletes_array;
        deletes_array.reserve(deletes.size());
        for (const auto& d: deletes) {
            deletes_array.push_back(d);
        }
        std::sort(deletes_array.begin(), deletes_array.end());
        if (fwrite(deletes_array.data(), sizeof(uint64_t), deletes_array.size(), f) != deletes_array.size()) {
            return Ret("DataMerger::merge_delta_file: failed to write deletes_array to merge file");
        }

        hdr.deleted_count = static_cast<uint32_t>(deletes_array.size());
    }

    // Overwrite header with updated values.
    if (0 != fseek(f, 0, SEEK_SET)) {
        return Ret("DataMerger::merge_delta_file: failed to rewind to header");
    }

    hdr.min_id = ids.empty() ? 0 : ids.front();
    hdr.max_id = ids.empty() ? 0 : ids.back();
    hdr.count  = static_cast<uint32_t>(ids.size());
    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        return Ret("DataMerger::merge_delta_file: failed to write header");
    }

    return Ret(0);
}


} // namespace sketch2
