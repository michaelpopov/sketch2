#include "data_merger.h"
#include <experimental/scope>
#include <stdio.h>
#include <errno.h>
#include <string.h>

namespace sketch2 {

Ret DataMerger::init(const std::string& source_path, const std::string& update_path) {
    const auto source_ret = source_.init(source_path);
    if (source_ret != 0) {
        return source_ret;
    }

    const auto update_ret = updater_.init(update_path);
    if (update_ret != 0) {
        return update_ret;
    }

    if (source_.dim() != updater_.dim() || source_.type() != updater_.type()) {
        return Ret("DataMerger::init: mismatching datasets.");
    }

    return Ret(0);
}

Ret DataMerger::merge_data_files(const std::string& path) {
    try {
        return merge_data_files_(path);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret DataMerger::merge_data_files_(const std::string& path) {
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
    hdr.type          = static_cast<uint16_t>(data_type_to_int(source_.type()));
    hdr.dim           = static_cast<uint16_t>(source_.dim());
    hdr.padding       = 0;

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        return Ret("DataMerger::merge_data_files: failed to write header");
    }

    load_update_records();
    load_update_deletes();

    std::vector<uint64_t> ids;
    ids.reserve(source_.count() + updater_items_.size());

    for (size_t i = 0, j = 0; i < source_.count() || j < updater_items_.size(); ) {
        const bool has_source = i < source_.count();
        const bool has_update = j < updater_items_.size();

        // Deletes can only mark deleted items in the source.
        // If an item is deleted, skip it and move forward.
        if (has_source) {
            const uint64_t source_id = source_.id(i);
            if (deletes_.find(source_id) != deletes_.end()) {
                i++;
                continue;
            }
        }

        // Update items must not contain same id in the data list and in the
        // list of deleted items. If this happens, it is an error.
        if (has_update) {
            const uint64_t update_id = updater_items_[j].id;
            if (deletes_.find(update_id) != deletes_.end()) {
                return Ret("DataMerger::merge_data_files: updated id is also deleted");
            }
        }

        if (has_source && has_update) {
            const uint64_t source_id = source_.id(i);
            const uint64_t update_id = updater_items_[j].id;

            if (source_id < update_id) {
                ids.push_back(source_id);
                write_data(f, source_.at(i));
                ++i;
            } else if (source_id > update_id) {
                ids.push_back(update_id);
                write_data(f, updater_items_[j].data);
                ++j;
            } else { // if source_id == update_id, update_id wins
                ids.push_back(update_id);
                write_data(f, updater_items_[j].data);
                ++i;
                ++j;
            }
            continue;
        }

        if (has_source) {
            const uint64_t source_id = source_.id(i);
            ids.push_back(source_id);
            write_data(f, source_.at(i));
            ++i;
        } else {
            const uint64_t update_id = updater_items_[j].id;
            ids.push_back(update_id);
            write_data(f, updater_items_[j].data);
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
    hdr.count  = ids.size();
    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        return Ret("DataMerger::merge_data_files: failed to write header");
    }

    return Ret(0);
}

Ret DataMerger::merge_delta_file(const std::string& path) {
    (void)path;
    return Ret(0);
}

Ret DataMerger::merge_data_delta_files(const std::string& path) {
    (void)path;
    return Ret(0);
}

void DataMerger::load_update_records() {
    updater_items_.clear();
    updater_items_.reserve(updater_.count());
    
    for (auto iter = updater_.begin(); !iter.eof(); iter.next()) {
        Item item {
            .id = iter.id(),
            .data = iter.data(),
        };
        updater_items_.push_back(item);
    }
}

void DataMerger::load_update_deletes() {
    deletes_.clear();
    for (size_t i = 0; i < updater_.deleted_count(); i++) {
        deletes_.insert(updater_.deleted_id(i));
    }
}

void DataMerger::write_data(FILE* f, const uint8_t* data) {
    if (fwrite(data, source_.size(), 1, f) != 1) {
        throw std::runtime_error("DataMerger::write_data: failed to write merge file");
    }
}

} // namespace sketch2
