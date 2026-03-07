#include "data_merger.h"
#include "core/storage/data_file_layout.h"
#include <algorithm>
#include <experimental/scope>
#include <filesystem>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>

namespace sketch2 {

namespace {

struct MergeItem {
    uint64_t id;
    const uint8_t* data;
};

constexpr size_t kFileBufferSize = 4 * 1024 * 1024;

void set_merge_file_buffer(FILE* f, std::vector<char>* file_buffer) {
    file_buffer->resize(kFileBufferSize);
    (void)setvbuf(f, file_buffer->data(), _IOFBF, file_buffer->size());
}

Ret flush_and_close_merge_file(FILE** f, const char* context) {
    const int n1 = fflush(*f);
    const int n2 = fsync(fileno(*f));
    const int n3 = fclose(*f);
    *f = nullptr;
    if (n1 != 0 || n2 != 0 || n3 != 0) {
        return Ret(std::string(context) + ": failed to flush and close merge file");
    }
    return Ret(0);
}

void write_data(FILE* f, const uint8_t* data, size_t size) {
    if (data == nullptr || size == 0) {
        throw std::runtime_error("DataMerger::write_data: invalid arguments");
    }
    if (fwrite(data, size, 1, f) != 1) {
        throw std::runtime_error("DataMerger::write_data: failed to write merge file");
    }
}

std::vector<MergeItem> load_update_records(const DataReader& updater) {
    std::vector<MergeItem> updater_items;
    updater_items.reserve(updater.count());
    for (auto iter = updater.begin(); !iter.eof(); iter.next()) {
        updater_items.push_back(MergeItem {
            .id = iter.id(),
            .data = iter.data(),
        });
    }
    return updater_items;
}

std::vector<MergeItem> load_update_records(const Accumulator& updater, const std::vector<uint64_t>& ids) {
    std::vector<MergeItem> updater_items;
    updater_items.reserve(ids.size());
    for (uint64_t id : ids) {
        updater_items.push_back(MergeItem {
            .id = id,
            .data = updater.get_vector(id),
        });
    }
    return updater_items;
}

std::vector<uint64_t> load_deleted_ids(const DataReader& updater) {
    std::vector<uint64_t> deletes;
    deletes.reserve(updater.deleted_count());
    for (size_t i = 0; i < updater.deleted_count(); ++i) {
        deletes.push_back(updater.deleted_id(i));
    }
    return deletes;
}

std::vector<uint64_t> build_delta_deletes(const DataReader& source,
        const std::vector<MergeItem>& updater_items,
        const std::vector<uint64_t>& updater_deletes) {
    std::vector<uint64_t> deletes;
    const std::vector<uint64_t> source_deletes = load_deleted_ids(source);

    size_t ui = 0;
    for (uint64_t sid : source_deletes) {
        while (ui < updater_items.size() && updater_items[ui].id < sid) {
            ++ui;
        }
        if (ui < updater_items.size() && updater_items[ui].id == sid) {
            continue;
        }
        deletes.push_back(sid);
    }

    std::vector<uint64_t> merged_deletes;
    merged_deletes.reserve(deletes.size() + updater_deletes.size());
    std::merge(
        deletes.begin(), deletes.end(),
        updater_deletes.begin(), updater_deletes.end(),
        std::back_inserter(merged_deletes));
    merged_deletes.erase(
        std::unique(merged_deletes.begin(), merged_deletes.end()),
        merged_deletes.end());
    return merged_deletes;
}

template <typename EmitFn>
Ret merge_records(const DataReader& source,
        const std::vector<MergeItem>& updater_items,
        const std::vector<uint64_t>& deletes,
        const std::string& conflict_message,
        EmitFn emit,
        std::vector<uint64_t>* output_ids) {
    output_ids->clear();
    output_ids->reserve(source.count() + updater_items.size());

    for (size_t i = 0, j = 0, di = 0, dj = 0; i < source.count() || j < updater_items.size(); ) {
        const bool has_source = i < source.count();
        const bool has_update = j < updater_items.size();

        if (has_source) {
            const uint64_t source_id = source.id(i);
            while (di < deletes.size() && deletes[di] < source_id) {
                ++di;
            }
            if (di < deletes.size() && deletes[di] == source_id) {
                ++i;
                continue;
            }
        }

        if (has_update) {
            const uint64_t update_id = updater_items[j].id;
            while (dj < deletes.size() && deletes[dj] < update_id) {
                ++dj;
            }
            if (dj < deletes.size() && deletes[dj] == update_id) {
                return Ret(conflict_message);
            }
        }

        if (has_source && has_update) {
            const uint64_t source_id = source.id(i);
            const uint64_t update_id = updater_items[j].id;
            if (source_id < update_id) {
                output_ids->push_back(source_id);
                emit(source.at(i), source.size());
                ++i;
            } else if (source_id > update_id) {
                output_ids->push_back(update_id);
                emit(updater_items[j].data, source.size());
                ++j;
            } else {
                output_ids->push_back(update_id);
                emit(updater_items[j].data, source.size());
                ++i;
                ++j;
            }
            continue;
        }

        if (has_source) {
            const uint64_t source_id = source.id(i);
            output_ids->push_back(source_id);
            emit(source.at(i), source.size());
            ++i;
        } else {
            const uint64_t update_id = updater_items[j].id;
            output_ids->push_back(update_id);
            emit(updater_items[j].data, source.size());
            ++j;
        }
    }

    return Ret(0);
}

template <typename FinalizeFn>
Ret merge_to_file(const DataReader& source,
        const std::string& path,
        const char* context,
        FinalizeFn finalize) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        return Ret(strerror(errno));
    }
    std::vector<char> file_buffer;
    set_merge_file_buffer(f, &file_buffer);
    std::experimental::scope_exit file_guard([&f]() {
        if (f) {
            fclose(f);
        }
    });

    DataFileHeader hdr = make_data_header(0, 0, 0, 0, source.type(), static_cast<uint16_t>(source.dim()));
    CHECK(write_header_and_data_padding(f, hdr, context));
    CHECK(finalize(f, &hdr));
    return flush_and_close_merge_file(&f, context);
}

} // namespace

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

    if (ret.code() != 0 && std::filesystem::exists(path)) {
        std::filesystem::remove(path);
    }

    return ret;
}

Ret DataMerger::merge_data_file(
        const DataReader& source, const Accumulator& updater,
        const std::vector<uint64_t>& ids, const std::vector<uint64_t>& deleted_ids,
        const std::string& path) {
    Ret ret(0);
    try {
        ret = merge_data_file_(source, updater, ids, deleted_ids, path);
    } catch (const std::exception& ex) {
        ret = Ret(ex.what());
    }

    if (ret.code() != 0 && std::filesystem::exists(path)) {
        std::filesystem::remove(path);
    }

    return ret;
}

Ret DataMerger::merge_data_file_(const DataReader& source, const DataReader& updater, const std::string& path) {
    if (source.dim() != updater.dim() || source.type() != updater.type()) {
        return Ret("DataMerger::merge_data_file: incompatible source and updater");
    }
    if (source.has_delta() || updater.has_delta()) {
        return Ret("DataMerger::merge_data_file: source and updater must not have deltas");
    }

    const std::vector<MergeItem> updater_items = load_update_records(updater);
    const std::vector<uint64_t> deletes = load_deleted_ids(updater);

    return merge_to_file(source, path, "DataMerger::merge_data_files",
        [&](FILE* f, DataFileHeader* hdr) -> Ret {
            std::vector<uint64_t> output_ids;
            CHECK(merge_records(source, updater_items, deletes,
                "DataMerger::merge_data_files: updated id is also deleted",
                [&](const uint8_t* data, size_t size) { write_data(f, data, size); },
                &output_ids));

            const IdsLayout ids_layout = compute_ids_layout(*hdr, output_ids.size(), source.size());
            CHECK(write_zero_padding(f, ids_layout.ids_padding,
                "DataMerger::merge_data_files: failed to write id alignment padding"));
            CHECK(write_u64_array(f, output_ids,
                "DataMerger::merge_data_files: failed to write ids to merge file"));

            hdr->min_id = output_ids.empty() ? 0 : output_ids.front();
            hdr->max_id = output_ids.empty() ? 0 : output_ids.back();
            hdr->count = static_cast<uint32_t>(output_ids.size());
            return rewrite_header(f, *hdr, "DataMerger::merge_data_files");
        });
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

    if (ret.code() != 0 && std::filesystem::exists(path)) {
        std::filesystem::remove(path);
    }

    return ret;
}

Ret DataMerger::merge_delta_file(
        const DataReader& source, const Accumulator& updater,
        const std::vector<uint64_t>& ids, const std::vector<uint64_t>& deleted_ids,
        const std::string& path) {
    Ret ret(0);
    try {
        ret = merge_delta_file_(source, updater, ids, deleted_ids, path);
    } catch (const std::exception& ex) {
        ret = Ret(ex.what());
    }

    if (ret.code() != 0 && std::filesystem::exists(path)) {
        std::filesystem::remove(path);
    }

    return ret;
}

Ret DataMerger::merge_delta_file_(const DataReader& source, const DataReader& updater, const std::string& path) {
    if (source.has_delta() || updater.has_delta()) {
        return Ret("DataMerger::merge_delta_file: source and updater must not have deltas");
    }

    const std::vector<MergeItem> updater_items = load_update_records(updater);
    const std::vector<uint64_t> updater_deletes = load_deleted_ids(updater);
    const std::vector<uint64_t> deletes = build_delta_deletes(source, updater_items, updater_deletes);

    return merge_to_file(source, path, "DataMerger::merge_delta_file",
        [&](FILE* f, DataFileHeader* hdr) -> Ret {
            std::vector<uint64_t> output_ids;
            CHECK(merge_records(source, updater_items, deletes,
                "DataMerger::merge_delta_file: updated id is also deleted",
                [&](const uint8_t* data, size_t size) { write_data(f, data, size); },
                &output_ids));

            const IdsLayout ids_layout = compute_ids_layout(*hdr, output_ids.size(), source.size());
            CHECK(write_zero_padding(f, ids_layout.ids_padding,
                "DataMerger::merge_delta_file: failed to write id alignment padding"));
            CHECK(write_u64_array(f, output_ids,
                "DataMerger::merge_delta_file: failed to write ids to merge file"));
            CHECK(write_u64_array(f, deletes,
                "DataMerger::merge_delta_file: failed to write deletes_array to merge file"));

            hdr->deleted_count = static_cast<uint32_t>(deletes.size());
            hdr->min_id = output_ids.empty() ? 0 : output_ids.front();
            hdr->max_id = output_ids.empty() ? 0 : output_ids.back();
            hdr->count = static_cast<uint32_t>(output_ids.size());
            return rewrite_header(f, *hdr, "DataMerger::merge_delta_file");
        });
}

Ret DataMerger::merge_data_file_(
        const DataReader& source, const Accumulator& updater,
        const std::vector<uint64_t>& ids, const std::vector<uint64_t>& deleted_ids,
        const std::string& path) {
    if (source.has_delta()) {
        return Ret("DataMerger::merge_data_file: source and updater must not have deltas");
    }

    const std::vector<MergeItem> updater_items = load_update_records(updater, ids);

    return merge_to_file(source, path, "DataMerger::merge_data_files",
        [&](FILE* f, DataFileHeader* hdr) -> Ret {
            std::vector<uint64_t> output_ids;
            CHECK(merge_records(source, updater_items, deleted_ids,
                "DataMerger::merge_data_files: updated id is also deleted",
                [&](const uint8_t* data, size_t size) { write_data(f, data, size); },
                &output_ids));

            const IdsLayout ids_layout = compute_ids_layout(*hdr, output_ids.size(), source.size());
            CHECK(write_zero_padding(f, ids_layout.ids_padding,
                "DataMerger::merge_data_files: failed to write id alignment padding"));
            CHECK(write_u64_array(f, output_ids,
                "DataMerger::merge_data_files: failed to write ids to merge file"));

            hdr->min_id = output_ids.empty() ? 0 : output_ids.front();
            hdr->max_id = output_ids.empty() ? 0 : output_ids.back();
            hdr->count = static_cast<uint32_t>(output_ids.size());
            return rewrite_header(f, *hdr, "DataMerger::merge_data_files");
        });
}

Ret DataMerger::merge_delta_file_(
        const DataReader& source, const Accumulator& updater,
        const std::vector<uint64_t>& ids, const std::vector<uint64_t>& deleted_ids,
        const std::string& path) {
    if (source.has_delta()) {
        return Ret("DataMerger::merge_delta_file: source and updater must not have deltas");
    }

    const std::vector<MergeItem> updater_items = load_update_records(updater, ids);
    const std::vector<uint64_t> deletes = build_delta_deletes(source, updater_items, deleted_ids);

    return merge_to_file(source, path, "DataMerger::merge_delta_file",
        [&](FILE* f, DataFileHeader* hdr) -> Ret {
            std::vector<uint64_t> output_ids;
            CHECK(merge_records(source, updater_items, deletes,
                "DataMerger::merge_delta_file: updated id is also deleted",
                [&](const uint8_t* data, size_t size) { write_data(f, data, size); },
                &output_ids));

            const IdsLayout ids_layout = compute_ids_layout(*hdr, output_ids.size(), source.size());
            CHECK(write_zero_padding(f, ids_layout.ids_padding,
                "DataMerger::merge_delta_file: failed to write id alignment padding"));
            CHECK(write_u64_array(f, output_ids,
                "DataMerger::merge_delta_file: failed to write ids to merge file"));
            CHECK(write_u64_array(f, deletes,
                "DataMerger::merge_delta_file: failed to write deletes_array to merge file"));

            hdr->deleted_count = static_cast<uint32_t>(deletes.size());
            hdr->min_id = output_ids.empty() ? 0 : output_ids.front();
            hdr->max_id = output_ids.empty() ? 0 : output_ids.back();
            hdr->count = static_cast<uint32_t>(output_ids.size());
            return rewrite_header(f, *hdr, "DataMerger::merge_delta_file");
        });
}

} // namespace sketch2
