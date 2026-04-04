// Implements merge operations for base data files and delta files.

#include "data_merger.h"
#include "core/compute/compute.h"
#include "core/storage/data_file_layout.h"
#include "core/storage/input_reader.h"
#include "core/utils/log.h"
#include "core/utils/shared_consts.h"
#include "core/utils/string_utils.h"
#include "core/utils/timer.h"
#include <algorithm>
#include <cassert>
#include <filesystem>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>

namespace sketch2 {

namespace {

enum class MergeItemStorage {
    Binary,
    Text,
};

struct MergeItemBinaryData {
    const uint8_t* data;
    float cosine_inv_norm;
};

struct MergeItemTextData {
    const char* start;
    const char* end;
};

struct MergeItem {
    uint64_t id;
    union {
        MergeItemBinaryData binary;
        MergeItemTextData text;
    };
};

struct MaterializedInputUpdater {
    // Every MergeItem in one updater uses the same representation, so the mode
    // and text delimiter choice live at the container level instead of being
    // repeated in each item. `compute_cosine_inv_norms` records the caller's
    // expectation so debug builds can assert that materialization-time and
    // write-time cosine handling stay in sync.
    MergeItemStorage storage = MergeItemStorage::Binary;
    bool text_is_comma_delimited = true;
    bool compute_cosine_inv_norms = false;
    std::vector<MergeItem> items;
    std::vector<uint64_t> deletes;
};

MergeItem make_binary_merge_item(uint64_t id, const uint8_t* data, float cosine_inv_norm) {
    MergeItem item{};
    item.id = id;
    item.binary = MergeItemBinaryData{
        .data = data,
        .cosine_inv_norm = cosine_inv_norm,
    };
    return item;
}

MergeItem make_text_merge_item(uint64_t id, const char* start, const char* end) {
    MergeItem item{};
    item.id = id;
    item.text = MergeItemTextData{
        .start = start,
        .end = end,
    };
    return item;
}

class CosineInvNormOutput {
public:
    explicit CosineInvNormOutput(bool enabled) : enabled_(enabled) {}

    // The cosine side-array is optional in the on-disk format. This helper lets
    // the merge code always "push" norms while internally becoming a no-op when
    // the current dataset layout does not persist them.
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

// Small forward declarations so the RAII helpers below can call the low-level
// file utilities that are defined later in this anonymous namespace.
void set_merge_file_buffer(FILE* f, std::vector<char>* file_buffer);
Ret flush_and_close_merge_file(FILE** f, const char* context);

class MergeFile {
public:
    MergeFile() = default;
    MergeFile(const MergeFile&) = delete;
    MergeFile& operator=(const MergeFile&) = delete;
    MergeFile(MergeFile&&) = delete;
    MergeFile& operator=(MergeFile&&) = delete;

    // Opens the destination file and writes a provisional header immediately.
    // The final counts/min/max ids are not known yet, so they are patched in at
    // the end once the merge body and trailing sections have been written.
    Ret open(const DataReader& source, const std::string& path, const char* context) {
        f_ = fopen(path.c_str(), "wb");
        if (!f_) {
            return Ret(strerror(errno));
        }

        set_merge_file_buffer(f_, &file_buffer_);
        header_ = make_data_header(
            0, 0, 0, 0, source.type(), static_cast<uint16_t>(source.dim()), source.has_cosine_inv_norms());
        Ret ret = write_header_and_data_padding(f_, header_, context);
        if (ret.code() != 0) {
            fclose(f_);
            f_ = nullptr;
        }
        return ret;
    }

    ~MergeFile() {
        if (f_) {
            fclose(f_);
        }
    }

    FILE* file() const { return f_; }
    DataFileHeader* header() { return &header_; }

    // Completes the durable write path. The pointer-reset prevents the dtor from
    // closing the same descriptor twice after ownership has conceptually ended.
    Ret flush_and_close(const char* context) {
        return flush_and_close_merge_file(&f_, context);
    }

private:
    FILE* f_ = nullptr;
    std::vector<char> file_buffer_;
    DataFileHeader header_ = {};
};

class MergeOutputWriter {
public:
    MergeOutputWriter(const MergeOutputWriter&) = delete;
    MergeOutputWriter& operator=(const MergeOutputWriter&) = delete;
    MergeOutputWriter(MergeOutputWriter&&) = delete;
    MergeOutputWriter& operator=(MergeOutputWriter&&) = delete;

    // Borrows the destination file handle and error-context string; both must
    // outlive this helper. In practice the FILE* comes from a live MergeFile
    // owned by the caller, and the context is a string literal used to build
    // stable error messages across the whole merge flow.
    MergeOutputWriter(FILE* f, const DataFileHeader& header, const char* context)
        : f_(f),
          type_(data_type_from_int(static_cast<int>(header.type))),
          dim_(header.dim),
          vector_size_(compute_vector_size(type_, dim_)),
          vector_stride_(header.vector_stride),
          context_(context),
          cosine_enabled_((header.flags & kDataFileHasCosineInvNorms) != 0u),
          cosine_inv_norms_((header.flags & kDataFileHasCosineInvNorms) != 0u) {}

    // The merge emits at most source.count() + updater.count() live rows.
    // Reserving once keeps the hot merge loop simple and avoids reallocations.
    // The text scratch buffer is only needed for direct-input text merges.
    void reserve(size_t count, bool needs_text_buffer) {
        output_ids_.reserve(count);
        cosine_inv_norms_.reserve(count);
        if (needs_text_buffer) {
            parsed_text_buffer_.resize(vector_size_);
        }
    }

    // Appends one surviving row to the output in the exact order required by
    // the file format: first vector records, later the optional cosine array and
    // id array. The ids are buffered here until those trailing sections are
    // written after all vectors.
    Ret write_binary_record(uint64_t id, const uint8_t* data, float cosine_inv_norm) {
        output_ids_.push_back(id);
        CHECK(write_vector_record(f_, data, vector_size_, vector_stride_, context_));
        cosine_inv_norms_.push(cosine_inv_norm);
        return Ret(0);
    }

    Ret write_text_record(uint64_t id, const char* start, const char* end, bool comma_delimited) {
        if (start == nullptr || end == nullptr) {
            return Ret("MergeOutputWriter: missing text vector range");
        }
        if (end < start) {
            return Ret("MergeOutputWriter: invalid text vector range");
        }
        if (parsed_text_buffer_.size() != vector_size_) {
            parsed_text_buffer_.resize(vector_size_);
        }

        const Ret parse_ret = comma_delimited
            ? parse_vector(parsed_text_buffer_.data(), parsed_text_buffer_.size(), type_, dim_, start, end)
            : parse_vector_spaces(parsed_text_buffer_.data(), parsed_text_buffer_.size(), type_, dim_, start, end);
        CHECK(parse_ret);

        const float cosine_inv_norm = cosine_inv_norms_enabled()
            ? compute_cosine_inverse_norm(parsed_text_buffer_.data(), type_, dim_)
            : 0.0f;
        return write_binary_record(id, parsed_text_buffer_.data(), cosine_inv_norm);
    }

    Ret write_updater_record(const MergeItem& item, MergeItemStorage storage, bool text_is_comma_delimited) {
        if (storage == MergeItemStorage::Binary) {
            return write_binary_record(item.id, item.binary.data, item.binary.cosine_inv_norm);
        }
        return write_text_record(item.id, item.text.start, item.text.end, text_is_comma_delimited);
    }

    // Writes the trailer that follows the vector-record area in every merged
    // file: optional cosine values, alignment padding before ids, and then the
    // sorted live-id array itself.
    Ret write_ids_section(const DataFileHeader& header,
            const char* ids_padding_message,
            const char* ids_message) const {
        cosine_inv_norms_.assert_matches(output_ids_.size());
        const IdsLayout ids_layout = compute_ids_layout(header, output_ids_.size());
        CHECK(cosine_inv_norms_.write(f_, context_));
        CHECK(write_zero_padding(f_, ids_layout.ids_padding, ids_padding_message));
        CHECK(write_u64_array(f_, output_ids_, ids_message));
        return Ret(0);
    }

    const std::vector<uint64_t>& output_ids() const { return output_ids_; }
    bool cosine_inv_norms_enabled() const { return cosine_enabled_; }

private:
    FILE* f_ = nullptr;
    DataType type_ = DataType::f32;
    uint16_t dim_ = 0;
    size_t vector_size_ = 0;
    size_t vector_stride_ = 0;
    const char* context_ = "";
    bool cosine_enabled_ = false;
    CosineInvNormOutput cosine_inv_norms_;
    std::vector<uint64_t> output_ids_;
    std::vector<uint8_t> parsed_text_buffer_;
};

void set_merge_file_buffer(FILE* f, std::vector<char>* file_buffer) {
    file_buffer->resize(kFileBufferSize);
    (void)setvbuf(f, file_buffer->data(), _IOFBF, file_buffer->size());
}

// Merges are implemented as "write everything to a fresh file, then rename by
// the caller". This helper makes the file durable before we report success.
Ret flush_and_close_merge_file(FILE** f, const char* context) {
    const int flush_ret = fflush(*f);
    int fsync_ret = 0;
    if (flush_ret == 0) {
        fsync_ret = fsync(fileno(*f));
    }
    const int close_ret = fclose(*f);
    *f = nullptr;
    if (flush_ret != 0 || fsync_ret != 0 || close_ret != 0) {
        return Ret(std::string(context) + ": failed to flush and close merge file");
    }
    return Ret(0);
}

// Materializes persisted updater rows into a sorted merge stream that points
// directly at mmap-backed binary records and their cosine metadata.
std::vector<MergeItem> load_update_records(const DataReader& updater) {
    std::vector<MergeItem> updater_items;
    updater_items.reserve(updater.count());
    for (auto iter = updater.begin(); !iter.eof(); iter.next()) {
        assert(iter.data() != nullptr);
        updater_items.push_back(make_binary_merge_item(iter.id(), iter.data(), iter.cosine_inv_norm()));
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

// Materializes a range view from raw input into the normalized updater shape
// used by the merge path: sorted live rows plus a sorted delete list. Binary
// input items borrow the mapped payload directly; text input items carry just
// the source slice and are parsed on demand when a surviving row is written.
//
// Important lifetime contract: text slices in `items` point into the mmap
// owned by the InputReader behind `updater`. Callers must keep that
// InputReader/InputReaderView alive until the merge fully completes.
Ret materialize_input_updater(const InputReaderView& updater,
        bool compute_cosine_inv_norms,
        MaterializedInputUpdater* materialized) {
    // This function keeps the merge core agnostic to the original source of
    // updates while avoiding a second large buffer for text input vectors.
    materialized->items.clear();
    materialized->deletes.clear();
    materialized->storage = updater.is_binary() ? MergeItemStorage::Binary : MergeItemStorage::Text;
    materialized->text_is_comma_delimited = updater.is_comma_delimited();
    materialized->compute_cosine_inv_norms = compute_cosine_inv_norms;

    // First count and split ids into "live rows" vs "deletes". The input view
    // is already sorted by id, so both output arrays preserve sorted order.
    size_t live_count = 0;
    for (size_t i = 0; i < updater.count(); ++i) {
        if (updater.is_no_data(i)) {
            materialized->deletes.push_back(updater.id(i));
        } else {
            ++live_count;
        }
    }

    materialized->items.reserve(live_count);
    for (size_t i = 0; i < updater.count(); ++i) {
        if (updater.is_no_data(i)) {
            continue;
        }

        if (updater.is_binary()) {
            const uint8_t* raw_data = nullptr;
            CHECK(updater.raw_data(i, &raw_data));
            materialized->items.push_back(make_binary_merge_item(
                updater.id(i),
                raw_data,
                compute_cosine_inv_norms ? compute_cosine_inverse_norm(raw_data, updater.type(), updater.dim()) : 0.0f));
        } else {
            const char* start = nullptr;
            const char* end = nullptr;
            CHECK(updater.text_data_range(i, &start, &end));
            materialized->items.push_back(make_text_merge_item(updater.id(i), start, end));
        }
    }

#ifndef NDEBUG
    for (size_t i = 1; i < materialized->items.size(); ++i) {
        assert(materialized->items[i - 1].id < materialized->items[i].id);
    }
    for (size_t i = 1; i < materialized->deletes.size(); ++i) {
        assert(materialized->deletes[i - 1] < materialized->deletes[i]);
    }
#endif
    return Ret(0);
}

// Computes the delete set for a merged delta file. Existing deletes survive
// unless the updater reintroduces that id, and new deletes are then unioned in.
std::vector<uint64_t> build_delta_deletes(const DataReader& source,
        const std::vector<MergeItem>& updater_items,
        const std::vector<uint64_t>& updater_deletes) {
    std::vector<uint64_t> deletes;
    const std::vector<uint64_t> source_deletes = load_deleted_ids(source);

    // A delta merge is not a plain union of tombstones. If the source delta had
    // a delete for some id, but the updater now reintroduces that id as a live
    // row, the old tombstone must disappear. The two-pointer walk below keeps
    // exactly those source deletes that are not resurrected by updater_items.
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

// Merges the sorted source stream and sorted updater stream into one ordered
// output. Deletes suppress matching ids, updater rows replace same-id source
// rows, and surviving records are streamed into the output writer.
Ret merge_records(const DataReader& source,
        const std::vector<MergeItem>& updater_items,
        MergeItemStorage updater_storage,
        bool updater_text_is_comma_delimited,
        const std::vector<uint64_t>& deletes,
        const std::string& conflict_message,
        MergeOutputWriter* output) {
    // i  -> current live row in the persisted source file
    // j  -> current live row in the updater stream
    // di -> current delete id checked against source ids
    // dj -> current delete id checked against updater ids
    //
    // The arrays are all sorted, so each index only moves forward. That keeps
    // the merge linear in the total number of source rows, update rows, and
    // delete ids.
    for (size_t i = 0, j = 0, di = 0, dj = 0; i < source.count() || j < updater_items.size(); ) {
        const bool has_source = i < source.count();
        const bool has_update = j < updater_items.size();

        if (has_source) {
            const uint64_t source_id = source.id(i);
            // A delete means "this id must not appear as a live row". When the
            // current source row is deleted, we simply skip it and keep merging.
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
            // Updater rows are already filtered to visible/live rows. If a live
            // updater id also appears in the delete set, the inputs are
            // contradictory and we fail the merge instead of silently choosing
            // one interpretation.
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
                // Source id comes first and is not shadowed by an update.
                CHECK(output->write_binary_record(source_id, source.at(i), source.cosine_inv_norm(i)));
                ++i;
            } else if (source_id > update_id) {
                // Updater inserted a new id before the current source id.
                CHECK(output->write_updater_record(
                    updater_items[j], updater_storage, updater_text_is_comma_delimited));
                ++j;
            } else {
                // Same id in both streams means "replace source with updater".
                CHECK(output->write_updater_record(
                    updater_items[j], updater_storage, updater_text_is_comma_delimited));
                ++i;
                ++j;
            }
            continue;
        }

        if (has_source) {
            const uint64_t source_id = source.id(i);
            // No updater rows remain, so every remaining source row survives.
            CHECK(output->write_binary_record(source_id, source.at(i), source.cosine_inv_norm(i)));
            ++i;
        } else {
            // No source rows remain, so every remaining updater row is appended.
            CHECK(output->write_updater_record(
                updater_items[j], updater_storage, updater_text_is_comma_delimited));
            ++j;
        }
    }

#ifndef NDEBUG
    const std::vector<uint64_t>& output_ids = output->output_ids();
    assert(std::is_sorted(output_ids.begin(), output_ids.end()));
    assert(std::adjacent_find(output_ids.begin(), output_ids.end()) == output_ids.end());
#endif
    return Ret(0);
}

// Header min/max/count are derived from the final live-id stream, so they can
// only be filled in after merge_records has finished.
void set_output_id_range(const std::vector<uint64_t>& output_ids, DataFileHeader* header) {
    header->min_id = output_ids.empty() ? 0 : output_ids.front();
    header->max_id = output_ids.empty() ? 0 : output_ids.back();
    header->count = static_cast<uint32_t>(output_ids.size());
}

} // namespace

Ret DataMerger::merge_data_file(const DataReader& source, const DataReader& updater, const std::string& path) {
    if (source.dim() != updater.dim() || source.type() != updater.type()) {
        return Ret("DataMerger::merge_data_file: incompatible source and updater");
    }

    // The public wrapper is intentionally defensive: convert unexpected
    // exceptions into Ret, and remove any partially-written destination file on
    // failure so callers never observe a half-formed merge artifact.
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

    Timer timer("merge_data_file");
    // For a data-file merge, deletes come directly from the updater because the
    // destination is a compact base file with no persisted tombstone section.
    const std::vector<MergeItem> updater_items = load_update_records(updater);
    const std::vector<uint64_t> deletes = load_deleted_ids(updater);
    MergeFile merge_file;
    CHECK(merge_file.open(source, path, "DataMerger::merge_data_files"));

    MergeOutputWriter output(merge_file.file(), *merge_file.header(), "DataMerger::merge_data_files");
    output.reserve(source.count() + updater_items.size(), false);
    CHECK(merge_records(
        source,
        updater_items,
        MergeItemStorage::Binary,
        true,
        deletes,
        "DataMerger::merge_data_files: updated id is also deleted",
        &output));
    // After all vector records are streamed, write the trailing metadata needed
    // to reopen the new file as a normal compact `.data` file.
    CHECK(output.write_ids_section(
        *merge_file.header(),
        "DataMerger::merge_data_files: failed to write id alignment padding",
        "DataMerger::merge_data_files: failed to write ids to merge file"));

    set_output_id_range(output.output_ids(), merge_file.header());
    CHECK(rewrite_header(merge_file.file(), *merge_file.header(), "DataMerger::merge_data_files"));
    Ret ret = merge_file.flush_and_close("DataMerger::merge_data_files");

    LOG_INFO << "Merged data file " << source.path() << " in " << timer.elapsed_ms() << " ms";
    return ret;
}

Ret DataMerger::merge_delta_file(const DataReader& source, const DataReader& updater, const std::string& path) {
    if (source.dim() != updater.dim() || source.type() != updater.type()) {
        return Ret("DataMerger::merge_delta_file: incompatible source and updater");
    }

    // Same failure contract as merge_data_file(): catch exceptions and remove
    // partial outputs so the caller either gets a complete file or no file.
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

Ret DataMerger::merge_data_file(const DataReader& source, const InputReaderView& updater, const std::string& path) {
    if (source.dim() != updater.dim() || source.type() != updater.type()) {
        return Ret("DataMerger::merge_data_file: incompatible source and updater");
    }

    // Public overload for direct-input merges. The failure contract intentionally
    // matches the DataReader overload so DatasetWriter can switch between the
    // two without having to special-case cleanup behavior.
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

Ret DataMerger::merge_delta_file(const DataReader& source, const InputReaderView& updater, const std::string& path) {
    if (source.dim() != updater.dim() || source.type() != updater.type()) {
        return Ret("DataMerger::merge_delta_file: incompatible source and updater");
    }

    // Same wrapper contract as the persisted-file overload: exceptions become
    // Ret and partial outputs are removed before the caller sees a result.
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

// Rewrites a delta file while preserving delta semantics: live updates stay in
// the record stream and the merged tombstone set is carried forward separately.
Ret DataMerger::merge_delta_file_(const DataReader& source, const DataReader& updater, const std::string& path) {
    if (source.has_cosine_inv_norms() != updater.has_cosine_inv_norms()) {
        return Ret("DataMerger::merge_delta_file: incompatible cosine inverse-norm layout");
    }
    if (source.has_delta() || updater.has_delta()) {
        return Ret("DataMerger::merge_delta_file: source and updater must not have deltas");
    }

    Timer timer("merge_delta_file");
    const std::vector<MergeItem> updater_items = load_update_records(updater);
    const std::vector<uint64_t> updater_deletes = load_deleted_ids(updater);
    // Delta-to-delta merge keeps a tombstone section, but it must first remove
    // any old deletes that the updater resurrected as live rows.
    const std::vector<uint64_t> deletes = build_delta_deletes(source, updater_items, updater_deletes);
    MergeFile merge_file;
    CHECK(merge_file.open(source, path, "DataMerger::merge_delta_file"));

    MergeOutputWriter output(merge_file.file(), *merge_file.header(), "DataMerger::merge_delta_file");
    output.reserve(source.count() + updater_items.size(), false);
    CHECK(merge_records(
        source,
        updater_items,
        MergeItemStorage::Binary,
        true,
        deletes,
        "DataMerger::merge_delta_file: updated id is also deleted",
        &output));
    // Delta files have the same live-row trailer as data files...
    CHECK(output.write_ids_section(
        *merge_file.header(),
        "DataMerger::merge_delta_file: failed to write id alignment padding",
        "DataMerger::merge_delta_file: failed to write ids to merge file"));
    // ...followed by the persisted tombstone array that future readers must
    // honor when overlaying this delta onto a base file.
    CHECK(write_u64_array(
        merge_file.file(),
        deletes,
        "DataMerger::merge_delta_file: failed to write deletes_array to merge file"));

    merge_file.header()->deleted_count = static_cast<uint32_t>(deletes.size());
    set_output_id_range(output.output_ids(), merge_file.header());
    CHECK(rewrite_header(merge_file.file(), *merge_file.header(), "DataMerger::merge_delta_file"));
    Ret ret = merge_file.flush_and_close("DataMerger::merge_delta_file");

    LOG_INFO << "Merged delta file " << source.path() << " in " << timer.elapsed_ms() << " ms";
    return ret;
}

Ret DataMerger::merge_data_file_(const DataReader& source, const InputReaderView& updater, const std::string& path) {
    if (source.dim() != updater.dim() || source.type() != updater.type()) {
        return Ret("DataMerger::merge_data_file: incompatible source and updater");
    }
    if (source.has_delta()) {
        return Ret("DataMerger::merge_data_file: source and updater must not have deltas");
    }

    Timer timer("merge_data_file");
    // The direct-input path still reuses the normal record merge loop and file
    // writer. The normalization step now only captures updater ids, deletes,
    // and either borrowed binary pointers or borrowed text slices; text vectors
    // are parsed later only if the row survives to output.
    MaterializedInputUpdater materialized;
    CHECK(materialize_input_updater(updater, source.has_cosine_inv_norms(), &materialized));
    MergeFile merge_file;
    CHECK(merge_file.open(source, path, "DataMerger::merge_data_files"));

    MergeOutputWriter output(merge_file.file(), *merge_file.header(), "DataMerger::merge_data_files");
    output.reserve(source.count() + materialized.items.size(), materialized.storage == MergeItemStorage::Text);
#ifndef NDEBUG
    assert(materialized.compute_cosine_inv_norms == output.cosine_inv_norms_enabled());
#endif
    CHECK(merge_records(
        source,
        materialized.items,
        materialized.storage,
        materialized.text_is_comma_delimited,
        materialized.deletes,
        "DataMerger::merge_data_files: updated id is also deleted",
        &output));
    CHECK(output.write_ids_section(
        *merge_file.header(),
        "DataMerger::merge_data_files: failed to write id alignment padding",
        "DataMerger::merge_data_files: failed to write ids to merge file"));

    set_output_id_range(output.output_ids(), merge_file.header());
    CHECK(rewrite_header(merge_file.file(), *merge_file.header(), "DataMerger::merge_data_files"));
    Ret ret = merge_file.flush_and_close("DataMerger::merge_data_files");

    LOG_INFO << "Merged data file " << source.path() << " in " << timer.elapsed_ms() << " ms";
    return ret;
}

Ret DataMerger::merge_delta_file_(const DataReader& source, const InputReaderView& updater, const std::string& path) {
    if (source.dim() != updater.dim() || source.type() != updater.type()) {
        return Ret("DataMerger::merge_delta_file: incompatible source and updater");
    }
    if (source.has_delta()) {
        return Ret("DataMerger::merge_delta_file: source and updater must not have deltas");
    }

    Timer timer("merge_delta_file");
    // Delta merges use the same lightweight adapter, then run the delta-
    // specific delete reconciliation step so old tombstones survive unless the
    // new input resurrects those ids as live rows.
    MaterializedInputUpdater materialized;
    CHECK(materialize_input_updater(updater, source.has_cosine_inv_norms(), &materialized));
    const std::vector<uint64_t> deletes = build_delta_deletes(source, materialized.items, materialized.deletes);
    MergeFile merge_file;
    CHECK(merge_file.open(source, path, "DataMerger::merge_delta_file"));

    MergeOutputWriter output(merge_file.file(), *merge_file.header(), "DataMerger::merge_delta_file");
    output.reserve(source.count() + materialized.items.size(), materialized.storage == MergeItemStorage::Text);
#ifndef NDEBUG
    assert(materialized.compute_cosine_inv_norms == output.cosine_inv_norms_enabled());
#endif
    CHECK(merge_records(
        source,
        materialized.items,
        materialized.storage,
        materialized.text_is_comma_delimited,
        deletes,
        "DataMerger::merge_delta_file: updated id is also deleted",
        &output));
    CHECK(output.write_ids_section(
        *merge_file.header(),
        "DataMerger::merge_delta_file: failed to write id alignment padding",
        "DataMerger::merge_delta_file: failed to write ids to merge file"));
    CHECK(write_u64_array(
        merge_file.file(),
        deletes,
        "DataMerger::merge_delta_file: failed to write deletes_array to merge file"));

    merge_file.header()->deleted_count = static_cast<uint32_t>(deletes.size());
    set_output_id_range(output.output_ids(), merge_file.header());
    CHECK(rewrite_header(merge_file.file(), *merge_file.header(), "DataMerger::merge_delta_file"));
    Ret ret = merge_file.flush_and_close("DataMerger::merge_delta_file");

    LOG_INFO << "Merged delta file " << source.path() << " in " << timer.elapsed_ms() << " ms";
    return ret;
}

} // namespace sketch2
