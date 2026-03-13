#include "data_merger.h"
#include "core/storage/data_file_layout.h"
#include "core/utils/shared_consts.h"
#include <algorithm>
#include <cassert>
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
    float cosine_inv_norm;
};

class CosineInvNormOutput {
public:
    explicit CosineInvNormOutput(bool enabled) : enabled_(enabled) {}

    void reserve(size_t count) {
        if (enabled_) {
            values_.reserve(count);
        }
    }

    void push(float value) {
        if (enabled_) {
            values_.push_back(value);
        }
    }

    void assert_matches(size_t count) const {
#ifndef NDEBUG
        assert(!enabled_ || values_.size() == count);
#else
        (void)count;
#endif
    }

    Ret write(FILE* f, const char* context) const {
        return write_f32_array(
            f,
            values_,
            std::string(context) + ": failed to write cosine inverse norms");
    }

private:
    bool enabled_ = false;
    std::vector<float> values_;
};

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

// Materializes visible updater rows into a sorted merge stream that keeps both
// vector bytes and optional cosine metadata together.
std::vector<MergeItem> load_update_records(const DataReader& updater) {
    std::vector<MergeItem> updater_items;
    updater_items.reserve(updater.count());
    for (auto iter = updater.begin(); !iter.eof(); iter.next()) {
        assert(iter.data() != nullptr);
        updater_items.push_back(MergeItem {
            .id = iter.id(),
            .data = iter.data(),
            .cosine_inv_norm = iter.cosine_inv_norm(),
        });
    }
#ifndef NDEBUG
    for (size_t i = 1; i < updater_items.size(); ++i) {
        assert(updater_items[i - 1].id < updater_items[i].id);
    }
#endif
    return updater_items;
}

// Builds a sorted merge stream from the accumulator for only the ids relevant
// to a single file range.
std::vector<MergeItem> load_update_records(const Accumulator& updater, const std::vector<uint64_t>& ids) {
    std::vector<MergeItem> updater_items;
    updater_items.reserve(ids.size());
    for (uint64_t id : ids) {
        const uint8_t* data = updater.get_vector(id);
        assert(data != nullptr);
        updater_items.push_back(MergeItem {
            .id = id,
            .data = data,
            .cosine_inv_norm = updater.get_vector_cosine_inv_norm(id),
        });
    }
#ifndef NDEBUG
    for (size_t i = 1; i < updater_items.size(); ++i) {
        assert(updater_items[i - 1].id < updater_items[i].id);
    }
#endif
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

// Computes the delete set for a merged delta file. Existing deletes survive
// unless the updater reintroduces that id, and new deletes are then unioned in.
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
// Merges the sorted source stream and sorted updater stream into one ordered
// output. Deletes suppress matching ids, updater rows replace same-id source
// rows, and emit() writes each surviving record to the destination format.
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
                CHECK(emit(source.at(i), source.cosine_inv_norm(i), source.size()));
                ++i;
            } else if (source_id > update_id) {
                output_ids->push_back(update_id);
                CHECK(emit(updater_items[j].data, updater_items[j].cosine_inv_norm, source.size()));
                ++j;
            } else {
                output_ids->push_back(update_id);
                CHECK(emit(updater_items[j].data, updater_items[j].cosine_inv_norm, source.size()));
                ++i;
                ++j;
            }
            continue;
        }

        if (has_source) {
            const uint64_t source_id = source.id(i);
            output_ids->push_back(source_id);
            CHECK(emit(source.at(i), source.cosine_inv_norm(i), source.size()));
            ++i;
        } else {
            const uint64_t update_id = updater_items[j].id;
            output_ids->push_back(update_id);
            CHECK(emit(updater_items[j].data, updater_items[j].cosine_inv_norm, source.size()));
            ++j;
        }
    }

#ifndef NDEBUG
    assert(std::is_sorted(output_ids->begin(), output_ids->end()));
    assert(std::adjacent_find(output_ids->begin(), output_ids->end()) == output_ids->end());
#endif
    return Ret(0);
}

template <typename FinalizeFn>
// Creates a merge output file with a provisional header, lets finalize() stream
// the merged body, and then flushes the file once finalize() has patched header fields.
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

    DataFileHeader hdr = make_data_header(
        0, 0, 0, 0, source.type(), static_cast<uint16_t>(source.dim()), source.has_cosine_inv_norms());
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

// Rewrites a full data file by merging persisted rows with another sorted file
// of updates/deletes, producing a compact output with no tombstone section.
Ret DataMerger::merge_data_file_(const DataReader& source, const DataReader& updater, const std::string& path) {
    if (source.dim() != updater.dim() || source.type() != updater.type()) {
        return Ret("DataMerger::merge_data_file: incompatible source and updater");
    }
    if (source.has_cosine_inv_norms() != updater.has_cosine_inv_norms()) {
        return Ret("DataMerger::merge_data_file: incompatible cosine inverse-norm layout");
    }
    if (source.has_delta() || updater.has_delta()) {
        return Ret("DataMerger::merge_data_file: source and updater must not have deltas");
    }

    const std::vector<MergeItem> updater_items = load_update_records(updater);
    const std::vector<uint64_t> deletes = load_deleted_ids(updater);

    return merge_to_file(source, path, "DataMerger::merge_data_files",
        [&](FILE* f, DataFileHeader* hdr) -> Ret {
            std::vector<uint64_t> output_ids;
            CosineInvNormOutput output_cosine_inv_norms((hdr->flags & kDataFileHasCosineInvNorms) != 0u);
            output_cosine_inv_norms.reserve(source.count() + updater_items.size());
            CHECK(merge_records(source, updater_items, deletes,
                "DataMerger::merge_data_files: updated id is also deleted",
                [&](const uint8_t* data, float cosine_inv_norm, size_t size) -> Ret {
                    CHECK(write_vector_record(f, data, size, hdr->vector_stride,
                        "DataMerger::merge_data_files"));
                    output_cosine_inv_norms.push(cosine_inv_norm);
                    return Ret(0);
                },
                &output_ids));

            output_cosine_inv_norms.assert_matches(output_ids.size());
            const IdsLayout ids_layout = compute_ids_layout(*hdr, output_ids.size());
            CHECK(output_cosine_inv_norms.write(f, "DataMerger::merge_data_files"));
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

// Rewrites a delta file while preserving delta semantics: live updates stay in
// the record stream and the merged tombstone set is carried forward separately.
Ret DataMerger::merge_delta_file_(const DataReader& source, const DataReader& updater, const std::string& path) {
    if (source.has_cosine_inv_norms() != updater.has_cosine_inv_norms()) {
        return Ret("DataMerger::merge_delta_file: incompatible cosine inverse-norm layout");
    }
    if (source.has_delta() || updater.has_delta()) {
        return Ret("DataMerger::merge_delta_file: source and updater must not have deltas");
    }

    const std::vector<MergeItem> updater_items = load_update_records(updater);
    const std::vector<uint64_t> updater_deletes = load_deleted_ids(updater);
    const std::vector<uint64_t> deletes = build_delta_deletes(source, updater_items, updater_deletes);

    return merge_to_file(source, path, "DataMerger::merge_delta_file",
        [&](FILE* f, DataFileHeader* hdr) -> Ret {
            std::vector<uint64_t> output_ids;
            CosineInvNormOutput output_cosine_inv_norms((hdr->flags & kDataFileHasCosineInvNorms) != 0u);
            output_cosine_inv_norms.reserve(source.count() + updater_items.size());
            CHECK(merge_records(source, updater_items, deletes,
                "DataMerger::merge_delta_file: updated id is also deleted",
                [&](const uint8_t* data, float cosine_inv_norm, size_t size) -> Ret {
                    CHECK(write_vector_record(f, data, size, hdr->vector_stride,
                        "DataMerger::merge_delta_file"));
                    output_cosine_inv_norms.push(cosine_inv_norm);
                    return Ret(0);
                },
                &output_ids));

            output_cosine_inv_norms.assert_matches(output_ids.size());
            const IdsLayout ids_layout = compute_ids_layout(*hdr, output_ids.size());
            CHECK(output_cosine_inv_norms.write(f, "DataMerger::merge_delta_file"));
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

// Merges a persisted data file with range-scoped accumulator contents, using
// the supplied sorted ids/deletes instead of reading an updater DataReader.
Ret DataMerger::merge_data_file_(
        const DataReader& source, const Accumulator& updater,
        const std::vector<uint64_t>& ids, const std::vector<uint64_t>& deleted_ids,
        const std::string& path) {
    if (source.has_cosine_inv_norms() != updater.has_cosine_inv_norms()) {
        return Ret("DataMerger::merge_data_file: incompatible cosine inverse-norm layout");
    }
    if (source.has_delta()) {
        return Ret("DataMerger::merge_data_file: source and updater must not have deltas");
    }

    const std::vector<MergeItem> updater_items = load_update_records(updater, ids);

    return merge_to_file(source, path, "DataMerger::merge_data_files",
        [&](FILE* f, DataFileHeader* hdr) -> Ret {
            std::vector<uint64_t> output_ids;
            CosineInvNormOutput output_cosine_inv_norms((hdr->flags & kDataFileHasCosineInvNorms) != 0u);
            output_cosine_inv_norms.reserve(source.count() + updater_items.size());
            CHECK(merge_records(source, updater_items, deleted_ids,
                "DataMerger::merge_data_files: updated id is also deleted",
                [&](const uint8_t* data, float cosine_inv_norm, size_t size) -> Ret {
                    CHECK(write_vector_record(f, data, size, hdr->vector_stride,
                        "DataMerger::merge_data_files"));
                    output_cosine_inv_norms.push(cosine_inv_norm);
                    return Ret(0);
                },
                &output_ids));

            output_cosine_inv_norms.assert_matches(output_ids.size());
            const IdsLayout ids_layout = compute_ids_layout(*hdr, output_ids.size());
            CHECK(output_cosine_inv_norms.write(f, "DataMerger::merge_data_files"));
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

// Merges a persisted delta file with range-scoped accumulator contents while
// rebuilding both the ordered live rows and the delta delete section.
Ret DataMerger::merge_delta_file_(
        const DataReader& source, const Accumulator& updater,
        const std::vector<uint64_t>& ids, const std::vector<uint64_t>& deleted_ids,
        const std::string& path) {
    if (source.has_cosine_inv_norms() != updater.has_cosine_inv_norms()) {
        return Ret("DataMerger::merge_delta_file: incompatible cosine inverse-norm layout");
    }
    if (source.has_delta()) {
        return Ret("DataMerger::merge_delta_file: source and updater must not have deltas");
    }

    const std::vector<MergeItem> updater_items = load_update_records(updater, ids);
    const std::vector<uint64_t> deletes = build_delta_deletes(source, updater_items, deleted_ids);

    return merge_to_file(source, path, "DataMerger::merge_delta_file",
        [&](FILE* f, DataFileHeader* hdr) -> Ret {
            std::vector<uint64_t> output_ids;
            CosineInvNormOutput output_cosine_inv_norms((hdr->flags & kDataFileHasCosineInvNorms) != 0u);
            output_cosine_inv_norms.reserve(source.count() + updater_items.size());
            CHECK(merge_records(source, updater_items, deletes,
                "DataMerger::merge_delta_file: updated id is also deleted",
                [&](const uint8_t* data, float cosine_inv_norm, size_t size) -> Ret {
                    CHECK(write_vector_record(f, data, size, hdr->vector_stride,
                        "DataMerger::merge_delta_file"));
                    output_cosine_inv_norms.push(cosine_inv_norm);
                    return Ret(0);
                },
                &output_ids));

            output_cosine_inv_norms.assert_matches(output_ids.size());
            const IdsLayout ids_layout = compute_ids_layout(*hdr, output_ids.size());
            CHECK(output_cosine_inv_norms.write(f, "DataMerger::merge_delta_file"));
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
