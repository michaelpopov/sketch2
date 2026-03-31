// Implements merge operations for base data files and delta files.

#include "data_merger.h"
#include "core/compute/compute.h"
#include "core/storage/data_file_layout.h"
#include "core/storage/input_reader.h"
#include "core/utils/compact_ids.h"
#include "core/utils/log.h"
#include "core/utils/shared_consts.h"
#include "core/utils/string_utils.h"
#include "core/utils/timer.h"
#include <algorithm>
#include <cassert>
#include <filesystem>
#include <limits>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <utility>

namespace sketch2 {

namespace {

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
        const Ret append_ret = output_ids_.append(id);
        if (append_ret.code() != 0) {
            return Ret(std::string(context_) + ": active ids: " + append_ret.message());
        }
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

    // Writes the trailer that follows the vector-record area in every merged
    // file: optional cosine values, alignment padding before ids, and then the
    // compact active/deleted id sections.
    Ret write_ids_section(const DataFileHeader& header,
            const CompactIdsBuilder& deleted_ids,
            const char* ids_padding_message,
            const char* ids_message,
            const char* deleted_ids_message) const {
        cosine_inv_norms_.assert_matches(output_ids_.count());
        const DataMetadataLayout metadata_layout = compute_data_metadata_layout(header, output_ids_.count());
        CHECK(cosine_inv_norms_.write(f_, context_));
        CHECK(write_zero_padding(f_, metadata_layout.ids_trailer_padding, ids_padding_message));
        CHECK(output_ids_.write(f_, ids_message));
        CHECK(deleted_ids.write(f_, deleted_ids_message));
        return Ret(0);
    }

    const CompactIdsBuilder& output_ids() const { return output_ids_; }
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
    CompactIdsBuilder output_ids_;
    std::vector<uint8_t> parsed_text_buffer_;
};

class DataReaderUpdaterCursor {
public:
    explicit DataReaderUpdaterCursor(const DataReader& reader) : iter_(reader.base_begin()) {}

    bool eof() const { return iter_.eof(); }
    uint64_t id() const { return iter_.id(); }
    void next() { iter_.next(); }

    Ret write_current(MergeOutputWriter* output) const {
        return output->write_binary_record(iter_.id(), iter_.data(), iter_.cosine_inv_norm());
    }

private:
    DataReader::OrderedIterator iter_;
};

class DataReaderDeletedCursor {
public:
    explicit DataReaderDeletedCursor(const DataReader& reader) : reader_(reader) {}

    bool eof() const { return index_ >= reader_.deleted_count(); }
    uint64_t id() const { return reader_.deleted_id(index_); }
    void next() {
        if (!eof()) {
            ++index_;
        }
    }

private:
    const DataReader& reader_;
    size_t index_ = 0;
};

class InputReaderUpdaterCursor {
public:
    InputReaderUpdaterCursor(const InputReaderView& reader, bool compute_cosine_inv_norms)
        : reader_(reader), compute_cosine_inv_norms_(compute_cosine_inv_norms) {
        advance_to_live_row();
    }

    bool eof() const { return index_ >= reader_.count(); }
    uint64_t id() const { return reader_.id(index_); }
    void next() {
        if (!eof()) {
            ++index_;
            advance_to_live_row();
        }
    }

    Ret write_current(MergeOutputWriter* output) const {
        if (reader_.is_binary()) {
            const uint8_t* raw_data = nullptr;
            CHECK(reader_.raw_data(index_, &raw_data));
            const float cosine_inv_norm = compute_cosine_inv_norms_
                ? compute_cosine_inverse_norm(raw_data, reader_.type(), reader_.dim())
                : 0.0f;
            return output->write_binary_record(id(), raw_data, cosine_inv_norm);
        }

        const char* start = nullptr;
        const char* end = nullptr;
        CHECK(reader_.text_data_range(index_, &start, &end));
        return output->write_text_record(id(), start, end, reader_.is_comma_delimited());
    }

private:
    void advance_to_live_row() {
        while (index_ < reader_.count() && reader_.is_no_data(index_)) {
            ++index_;
        }
    }

    const InputReaderView& reader_;
    size_t index_ = 0;
    bool compute_cosine_inv_norms_ = false;
};

class InputReaderDeletedCursor {
public:
    explicit InputReaderDeletedCursor(const InputReaderView& reader) : reader_(reader) {
        advance_to_deleted_row();
    }

    bool eof() const { return index_ >= reader_.count(); }
    uint64_t id() const { return reader_.id(index_); }
    void next() {
        if (!eof()) {
            ++index_;
            advance_to_deleted_row();
        }
    }

private:
    void advance_to_deleted_row() {
        while (index_ < reader_.count() && !reader_.is_no_data(index_)) {
            ++index_;
        }
    }

    const InputReaderView& reader_;
    size_t index_ = 0;
};

struct InputReaderViewCounts {
    size_t live_count = 0;
    size_t delete_count = 0;
};

InputReaderViewCounts count_input_reader_view_rows(const InputReaderView& reader) {
    InputReaderViewCounts counts;
    for (size_t i = 0; i < reader.count(); ++i) {
        if (reader.is_no_data(i)) {
            ++counts.delete_count;
        } else {
            ++counts.live_count;
        }
    }
    return counts;
}

template <typename SourceDeletedCursor, typename UpdaterCursor, typename UpdaterDeletedCursor>
class DeltaDeleteCursor {
public:
    DeltaDeleteCursor(SourceDeletedCursor source_deleted,
            UpdaterCursor updater,
            UpdaterDeletedCursor updater_deleted)
        : source_deleted_(std::move(source_deleted)),
          updater_(std::move(updater)),
          updater_deleted_(std::move(updater_deleted)) {
        select_current();
    }

    bool eof() const { return !has_current_; }
    uint64_t id() const { return current_id_; }

    void next() {
        if (!has_current_) {
            return;
        }

        switch (current_source_) {
        case CurrentSource::SourceOnly:
            source_deleted_.next();
            break;
        case CurrentSource::UpdaterOnly:
            updater_deleted_.next();
            break;
        case CurrentSource::Both:
            source_deleted_.next();
            updater_deleted_.next();
            break;
        }
        select_current();
    }

private:
    enum class CurrentSource {
        SourceOnly,
        UpdaterOnly,
        Both,
    };

    void skip_resurrected_source_deletes() {
        while (!source_deleted_.eof()) {
            const uint64_t source_id = source_deleted_.id();
            while (!updater_.eof() && updater_.id() < source_id) {
                updater_.next();
            }
            if (!updater_.eof() && updater_.id() == source_id) {
                source_deleted_.next();
                continue;
            }
            break;
        }
    }

    void select_current() {
        skip_resurrected_source_deletes();
        if (source_deleted_.eof()) {
            if (updater_deleted_.eof()) {
                has_current_ = false;
                return;
            }
            has_current_ = true;
            current_source_ = CurrentSource::UpdaterOnly;
            current_id_ = updater_deleted_.id();
            return;
        }
        if (updater_deleted_.eof()) {
            has_current_ = true;
            current_source_ = CurrentSource::SourceOnly;
            current_id_ = source_deleted_.id();
            return;
        }

        const uint64_t source_id = source_deleted_.id();
        const uint64_t updater_deleted_id = updater_deleted_.id();
        has_current_ = true;
        if (source_id < updater_deleted_id) {
            current_source_ = CurrentSource::SourceOnly;
            current_id_ = source_id;
        } else if (updater_deleted_id < source_id) {
            current_source_ = CurrentSource::UpdaterOnly;
            current_id_ = updater_deleted_id;
        } else {
            current_source_ = CurrentSource::Both;
            current_id_ = source_id;
        }
    }

    SourceDeletedCursor source_deleted_;
    UpdaterCursor updater_;
    UpdaterDeletedCursor updater_deleted_;
    uint64_t current_id_ = 0;
    bool has_current_ = false;
    CurrentSource current_source_ = CurrentSource::SourceOnly;
};

auto make_data_reader_delta_delete_cursor(const DataReader& source, const DataReader& updater) {
    return DeltaDeleteCursor(
        DataReaderDeletedCursor(source),
        DataReaderUpdaterCursor(updater),
        DataReaderDeletedCursor(updater));
}

auto make_input_reader_delta_delete_cursor(
        const DataReader& source,
        const InputReaderView& updater,
        bool compute_cosine_inv_norms) {
    return DeltaDeleteCursor(
        DataReaderDeletedCursor(source),
        InputReaderUpdaterCursor(updater, compute_cosine_inv_norms),
        InputReaderDeletedCursor(updater));
}

template <typename Cursor>
Ret build_compact_ids_builder(Cursor cursor, const char* context, size_t reserve_count, CompactIdsBuilder* out) {
    out->clear();
    out->reserve(reserve_count);
    for (; !cursor.eof(); cursor.next()) {
        const Ret append_ret = out->append(cursor.id());
        if (append_ret.code() != 0) {
            return Ret(std::string(context) + ": " + append_ret.message());
        }
    }
    return Ret(0);
}

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

// Merges the sorted source stream and sorted updater stream into one ordered
// output. Deletes suppress matching ids, updater rows replace same-id source
// rows, and surviving records are streamed into the output writer.
template <typename UpdaterCursor, typename SourceDeleteCursor, typename UpdaterDeleteCursor>
Ret merge_records(const DataReader& source,
        UpdaterCursor updater,
        SourceDeleteCursor source_deletes,
        UpdaterDeleteCursor updater_deletes,
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
    for (size_t i = 0; i < source.count() || !updater.eof(); ) {
        const bool has_source = i < source.count();
        const bool has_update = !updater.eof();

        if (has_source) {
            const uint64_t source_id = source.id(i);
            // A delete means "this id must not appear as a live row". When the
            // current source row is deleted, we simply skip it and keep merging.
            while (!source_deletes.eof() && source_deletes.id() < source_id) {
                source_deletes.next();
            }
            if (!source_deletes.eof() && source_deletes.id() == source_id) {
                ++i;
                continue;
            }
        }

        if (has_update) {
            const uint64_t update_id = updater.id();
            // Updater rows are already filtered to visible/live rows. If a live
            // updater id also appears in the delete set, the inputs are
            // contradictory and we fail the merge instead of silently choosing
            // one interpretation.
            while (!updater_deletes.eof() && updater_deletes.id() < update_id) {
                updater_deletes.next();
            }
            if (!updater_deletes.eof() && updater_deletes.id() == update_id) {
                return Ret(conflict_message);
            }
        }

        if (has_source && has_update) {
            const uint64_t source_id = source.id(i);
            const uint64_t update_id = updater.id();
            if (source_id < update_id) {
                // Source id comes first and is not shadowed by an update.
                CHECK(output->write_binary_record(source_id, source.at(i), source.cosine_inv_norm(i)));
                ++i;
            } else if (source_id > update_id) {
                // Updater inserted a new id before the current source id.
                CHECK(updater.write_current(output));
                updater.next();
            } else {
                // Same id in both streams means "replace source with updater".
                CHECK(updater.write_current(output));
                ++i;
                updater.next();
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
            CHECK(updater.write_current(output));
            updater.next();
        }
    }

#ifndef NDEBUG
    const CompactIdsBuilder& output_ids = output->output_ids();
    if (output_ids.count() > 1) {
        assert(output_ids.min_id() < output_ids.max_id());
    }
#endif
    return Ret(0);
}

// Header min/max/count are derived from the final live-id stream, so they can
// only be filled in after merge_records has finished.
void set_output_id_range(const CompactIdsBuilder& output_ids, DataFileHeader* header) {
    header->min_id = output_ids.empty() ? 0 : output_ids.min_id();
    header->max_id = output_ids.empty() ? 0 : output_ids.max_id();
    header->count = static_cast<uint32_t>(output_ids.count());
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
    MergeFile merge_file;
    CHECK(merge_file.open(source, path, "DataMerger::merge_data_files"));

    MergeOutputWriter output(merge_file.file(), *merge_file.header(), "DataMerger::merge_data_files");
    output.reserve(source.count() + updater.count(), false);
    CHECK(merge_records(
        source,
        DataReaderUpdaterCursor(updater),
        DataReaderDeletedCursor(updater),
        DataReaderDeletedCursor(updater),
        "DataMerger::merge_data_files: updated id is also deleted",
        &output));
    // After all vector records are streamed, write the trailing metadata needed
    // to reopen the new file as a normal compact `.data` file.
    CHECK(output.write_ids_section(
        *merge_file.header(),
        {},
        "DataMerger::merge_data_files: failed to write id alignment padding",
        "DataMerger::merge_data_files: failed to write ids to merge file",
        "DataMerger::merge_data_files: failed to write deleted_ids to merge file"));

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
    // Delta-to-delta merge keeps a tombstone section, but it must first remove
    // any old deletes that the updater resurrected as live rows.
    CompactIdsBuilder compact_deleted_ids;
    CHECK(build_compact_ids_builder(
        make_data_reader_delta_delete_cursor(source, updater),
        "DataMerger::merge_delta_file: deleted ids",
        source.deleted_count() + updater.deleted_count(),
        &compact_deleted_ids));
    MergeFile merge_file;
    CHECK(merge_file.open(source, path, "DataMerger::merge_delta_file"));

    MergeOutputWriter output(merge_file.file(), *merge_file.header(), "DataMerger::merge_delta_file");
    output.reserve(source.count() + updater.count(), false);
    CHECK(merge_records(
        source,
        DataReaderUpdaterCursor(updater),
        make_data_reader_delta_delete_cursor(source, updater),
        make_data_reader_delta_delete_cursor(source, updater),
        "DataMerger::merge_delta_file: updated id is also deleted",
        &output));
    // Delta files have the same live-row trailer as data files...
    CHECK(output.write_ids_section(
        *merge_file.header(),
        compact_deleted_ids,
        "DataMerger::merge_delta_file: failed to write id alignment padding",
        "DataMerger::merge_delta_file: failed to write ids to merge file",
        "DataMerger::merge_delta_file: failed to write deleted_ids to merge file"));

    merge_file.header()->deleted_count = static_cast<uint32_t>(compact_deleted_ids.count());
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
    const InputReaderViewCounts updater_counts = count_input_reader_view_rows(updater);
    const CompactIdsBuilder empty_deleted_ids;
    MergeFile merge_file;
    CHECK(merge_file.open(source, path, "DataMerger::merge_data_files"));

    MergeOutputWriter output(merge_file.file(), *merge_file.header(), "DataMerger::merge_data_files");
    output.reserve(source.count() + updater_counts.live_count, !updater.is_binary());
#ifndef NDEBUG
    assert(source.has_cosine_inv_norms() == output.cosine_inv_norms_enabled());
#endif
    CHECK(merge_records(
        source,
        InputReaderUpdaterCursor(updater, source.has_cosine_inv_norms()),
        InputReaderDeletedCursor(updater),
        InputReaderDeletedCursor(updater),
        "DataMerger::merge_data_files: updated id is also deleted",
        &output));
    CHECK(output.write_ids_section(
        *merge_file.header(),
        empty_deleted_ids,
        "DataMerger::merge_data_files: failed to write id alignment padding",
        "DataMerger::merge_data_files: failed to write ids to merge file",
        "DataMerger::merge_data_files: failed to write deleted_ids to merge file"));

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
    const InputReaderViewCounts updater_counts = count_input_reader_view_rows(updater);
    const bool compute_cosine_inv_norms = source.has_cosine_inv_norms();
    CompactIdsBuilder compact_deleted_ids;
    CHECK(build_compact_ids_builder(
        make_input_reader_delta_delete_cursor(source, updater, compute_cosine_inv_norms),
        "DataMerger::merge_delta_file: deleted ids",
        source.deleted_count() + updater_counts.delete_count,
        &compact_deleted_ids));
    MergeFile merge_file;
    CHECK(merge_file.open(source, path, "DataMerger::merge_delta_file"));

    MergeOutputWriter output(merge_file.file(), *merge_file.header(), "DataMerger::merge_delta_file");
    output.reserve(source.count() + updater_counts.live_count, !updater.is_binary());
#ifndef NDEBUG
    assert(source.has_cosine_inv_norms() == output.cosine_inv_norms_enabled());
#endif
    CHECK(merge_records(
        source,
        InputReaderUpdaterCursor(updater, compute_cosine_inv_norms),
        make_input_reader_delta_delete_cursor(source, updater, compute_cosine_inv_norms),
        make_input_reader_delta_delete_cursor(source, updater, compute_cosine_inv_norms),
        "DataMerger::merge_delta_file: updated id is also deleted",
        &output));
    CHECK(output.write_ids_section(
        *merge_file.header(),
        compact_deleted_ids,
        "DataMerger::merge_delta_file: failed to write id alignment padding",
        "DataMerger::merge_delta_file: failed to write ids to merge file",
        "DataMerger::merge_delta_file: failed to write deleted_ids to merge file"));

    merge_file.header()->deleted_count = static_cast<uint32_t>(compact_deleted_ids.count());
    set_output_id_range(output.output_ids(), merge_file.header());
    CHECK(rewrite_header(merge_file.file(), *merge_file.header(), "DataMerger::merge_delta_file"));
    Ret ret = merge_file.flush_and_close("DataMerger::merge_delta_file");

    LOG_INFO << "Merged delta file " << source.path() << " in " << timer.elapsed_ms() << " ms";
    return ret;
}

} // namespace sketch2
