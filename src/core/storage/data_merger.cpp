// Implements merge operations for base data files and delta files.

#include "data_merger.h"
#include "core/compute/norm_utils.h"
#include "core/storage/data_file_layout.h"
#include "core/storage/input_reader.h"
#include "core/storage/compact_ids.h"
#include "core/storage/compact_ids_shared.h"
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

// Small forward declarations so the RAII helpers below can call the low-level
// file utilities that are defined later in this anonymous namespace.
void set_merge_file_buffer(FILE* f, std::vector<char>* file_buffer);
Ret flush_and_close_merge_file(FILE** f, const char* context);

float compute_stored_norm_for_dist(const uint8_t* data, DataType type, size_t dim, DistFunc dist_func) {
    switch (dist_func) {
        case DistFunc::COS:
            return static_cast<float>(compute_cosine_inverse_norm(data, type, dim));
        case DistFunc::L2:
            return static_cast<float>(compute_squared_norm(data, type, dim));
        case DistFunc::DOT:
            throw std::runtime_error("compute_stored_norm_for_dist: DOT does not use stored norms");
        default:
            throw std::runtime_error("compute_stored_norm_for_dist: unsupported distance function");
    }
}

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
    Ret open(const DataReader& source, uint32_t norm_flags, const std::string& path, const char* context) {
        f_ = fopen(path.c_str(), "wb");
        if (!f_) {
            return Ret(strerror(errno));
        }

        set_merge_file_buffer(f_, &file_buffer_);
        header_ = make_data_header(
            0, 0, source.min_range_id(), 0, 0, source.type(), static_cast<uint16_t>(source.dim()), norm_flags);
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
          min_range_id_(header.min_range_id),
          record_layout_(compute_data_record_layout(type_, dim_, data_file_has_norms(header))),
          context_(context),
          norms_enabled_(data_file_has_norms(header)) {}

    // The merge emits at most source.count() + updater.count() live rows.
    // Reserving once keeps the hot merge loop simple and avoids reallocations.
    // The text scratch buffer is only needed for direct-input text merges.
    void reserve(size_t count, bool needs_text_buffer) {
        output_ids_capacity_ = count;
        if (needs_text_buffer) {
            parsed_text_buffer_.resize(record_layout_.vector_size);
        }
    }

    // Appends one surviving row to the output in the exact order required by
    // the file format: a full inline record first, then the ids trailer later.
    // The ids are buffered here until those trailing sections are written
    // after all rows.
    Ret write_binary_record(uint64_t id, const uint8_t* data, float norm) {
        if (id < min_range_id_) {
            return Ret(std::string(context_) + ": active ids: id is below min_range_id");
        }
        if (id - min_range_id_ > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            return Ret(std::string(context_) + ": active ids: data file range exceeds uint32_t");
        }
        if (!output_ids_initialized_) {
            output_ids_.init(id, output_ids_capacity_);
            output_ids_min_ = id;
            output_ids_max_ = id;
            output_ids_initialized_ = true;
        } else {
            if (id <= output_ids_max_) {
                return Ret(std::string(context_) + ": active ids: ids must be strictly increasing");
            }
            output_ids_max_ = id;
        }
        output_ids_.add(id);
        CHECK(write_data_record(
            f_, data, record_layout_, norms_enabled_ ? &norm : nullptr, context_));
        return Ret(0);
    }

    Ret write_text_record(uint64_t id, const char* start, const char* end, bool comma_delimited,
            DistFunc dist_func) {
        if (start == nullptr || end == nullptr) {
            return Ret("MergeOutputWriter: missing text vector range");
        }
        if (end < start) {
            return Ret("MergeOutputWriter: invalid text vector range");
        }
        if (parsed_text_buffer_.size() != record_layout_.vector_size) {
            parsed_text_buffer_.resize(record_layout_.vector_size);
        }

        const Ret parse_ret = comma_delimited
            ? parse_vector(parsed_text_buffer_.data(), parsed_text_buffer_.size(), type_, dim_, start, end)
            : parse_vector_spaces(parsed_text_buffer_.data(), parsed_text_buffer_.size(), type_, dim_, start, end);
        CHECK(parse_ret);

        const float norm = norms_enabled()
            ? compute_stored_norm_for_dist(parsed_text_buffer_.data(), type_, dim_, dist_func)
            : 0.0f;
        return write_binary_record(id, parsed_text_buffer_.data(), norm);
    }

    // Writes the trailer that follows the inline record area in every merged
    // file: alignment padding before ids, then the compact active/deleted id
    // sections. Norms live inside each record, so there is no separate norms
    // section between vectors and ids.
    Ret write_ids_section(const DataFileHeader& header,
            const CompactIds& deleted_ids,
            const char* ids_message,
            const char* deleted_ids_message) {
        CompactIds output_ids_ext;
        output_ids_.complete_adding();
        CHECK(output_ids_ext.init(output_ids_));
        output_ids_bytes_ = output_ids_ext.serialized_size_bytes();
        const DataMetadataLayout metadata_layout = compute_data_metadata_layout(header, output_ids_.size());
        CHECK(write_zero_padding(f_, metadata_layout.vectors_padding,
            std::string(context_) + ": failed to write ids alignment padding"));
        CHECK(output_ids_ext.write(f_, ids_message));
        CHECK(write_zero_padding(
            f_,
            compute_deleted_ids_padding(metadata_layout.ids_trailer_offset, output_ids_ext.serialized_size_bytes()),
            std::string(context_) + ": failed to write deleted_ids alignment padding"));
        CHECK(deleted_ids.write(f_, deleted_ids_message));
        return Ret(0);
    }

    size_t output_count() const { return output_ids_.size(); }
    bool output_empty() const { return output_ids_.size() == 0; }
    uint64_t output_min_id() const { return output_ids_min_; }
    uint64_t output_max_id() const { return output_ids_max_; }
    bool norms_enabled() const { return norms_enabled_; }
    size_t output_ids_bytes() const { return output_ids_bytes_; }

private:
    FILE* f_ = nullptr;
    DataType type_ = DataType::f32;
    uint16_t dim_ = 0;
    uint64_t min_range_id_ = 0;
    DataRecordLayout record_layout_{};
    const char* context_ = "";
    bool norms_enabled_ = false;
    CompactIdsAccumulator output_ids_;
    size_t output_ids_capacity_ = 0;
    size_t output_ids_bytes_ = 0;
    bool output_ids_initialized_ = false;
    uint64_t output_ids_min_ = 0;
    uint64_t output_ids_max_ = 0;
    std::vector<uint8_t> parsed_text_buffer_;
};

class DataReaderUpdaterCursor {
public:
    // We intentionally iterate updater base rows only. Merge entry points reject
    // updater files with attached deltas, so base_begin() is equivalent to
    // begin() here and avoids pulling delta semantics into this cursor.
    explicit DataReaderUpdaterCursor(const DataReader& reader, uint32_t target_norm_flags)
        : reader_(&reader), iter_(reader.base_begin()), target_norm_flags_(target_norm_flags) {}

    bool eof() const { return iter_.eof(); }
    uint64_t id() const { return iter_.id(); }
    void next() {
        if (iter_.eof()) {
            return;
        }
#ifndef NDEBUG
        const uint64_t prev_id = iter_.id();
#endif
        iter_.next();
#ifndef NDEBUG
        if (!iter_.eof()) {
            assert(prev_id < iter_.id());
        }
#endif
    }

    Ret write_current(MergeOutputWriter* output) const {
        float norm = 0.0f;
        if (output->norms_enabled()) {
            if (reader_->norm_flags() == target_norm_flags_) {
                norm = iter_.get_norm();
            } else {
                norm = compute_stored_norm_for_dist(
                    iter_.data(),
                    reader_->type(),
                    reader_->dim(),
                    dist_func_for_data_file_norm_flags(target_norm_flags_));
            }
        }
        return output->write_binary_record(iter_.id(), iter_.data(), norm);
    }

private:
    const DataReader* reader_ = nullptr;
    DataReader::OrderedIterator iter_;
    uint32_t target_norm_flags_ = 0;
};

class DataReaderDeletedCursor {
public:
    explicit DataReaderDeletedCursor(const DataReader& reader) : reader_(reader) {}

    bool eof() const { return index_ >= reader_.deleted_count(); }
    uint64_t id() const { return reader_.deleted_id(index_); }
    void next() {
        if (!eof()) {
#ifndef NDEBUG
            const uint64_t prev_id = reader_.deleted_id(index_);
#endif
            ++index_;
#ifndef NDEBUG
            if (!eof()) {
                assert(prev_id < reader_.deleted_id(index_));
            }
#endif
        }
    }

private:
    const DataReader& reader_;
    size_t index_ = 0;
};

class InputReaderUpdaterCursor {
public:
    InputReaderUpdaterCursor(const InputReaderView& reader, DistFunc dist_func, bool compute_norms)
        : reader_(reader), dist_func_(dist_func), compute_norms_(compute_norms) {
        advance_to_live_row();
    }

    bool eof() const { return index_ >= reader_.count(); }
    uint64_t id() const { return reader_.id(index_); }
    void next() {
        if (!eof()) {
#ifndef NDEBUG
            const uint64_t prev_id = reader_.id(index_);
#endif
            ++index_;
            advance_to_live_row();
#ifndef NDEBUG
            if (!eof()) {
                assert(prev_id < reader_.id(index_));
            }
#endif
        }
    }

    Ret write_current(MergeOutputWriter* output) const {
        if (reader_.is_binary()) {
            const uint8_t* raw_data = nullptr;
            CHECK(reader_.raw_data(index_, &raw_data));
            const float norm = compute_norms_
                ? compute_stored_norm_for_dist(raw_data, reader_.type(), reader_.dim(), dist_func_)
                : 0.0f;
            return output->write_binary_record(id(), raw_data, norm);
        }

        const char* start = nullptr;
        const char* end = nullptr;
        CHECK(reader_.text_data_range(index_, &start, &end));
        return output->write_text_record(id(), start, end, reader_.is_comma_delimited(), dist_func_);
    }

private:
    void advance_to_live_row() {
        while (index_ < reader_.count() && reader_.is_no_data(index_)) {
            ++index_;
        }
    }

    const InputReaderView& reader_;
    DistFunc dist_func_ = DistFunc::DOT;
    size_t index_ = 0;
    bool compute_norms_ = false;
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
#ifndef NDEBUG
            const uint64_t prev_id = reader_.id(index_);
#endif
            ++index_;
            advance_to_deleted_row();
#ifndef NDEBUG
            if (!eof()) {
                assert(prev_id < reader_.id(index_));
            }
#endif
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

size_t count_input_reader_view_deleted_rows(const InputReaderView& reader) {
    size_t deleted_count = 0;
    for (size_t i = 0; i < reader.count(); ++i) {
        if (reader.is_no_data(i)) {
            ++deleted_count;
        }
    }
    return deleted_count;
}

template <typename UpdaterCursor>
std::vector<uint64_t> collect_updater_live_ids(UpdaterCursor updater, size_t reserve_count) {
    std::vector<uint64_t> live_ids;
    live_ids.reserve(reserve_count);
    for (; !updater.eof(); updater.next()) {
        live_ids.push_back(updater.id());
    }
    return live_ids;
}

template <typename SourceDeletedCursor, typename UpdaterDeletedCursor>
class DeltaDeleteCursor {
public:
    DeltaDeleteCursor(SourceDeletedCursor source_deleted,
            UpdaterDeletedCursor updater_deleted,
            const std::vector<uint64_t>* updater_live_ids)
        : source_deleted_(std::move(source_deleted)),
          updater_deleted_(std::move(updater_deleted)),
          updater_live_ids_(updater_live_ids) {
        assert(updater_live_ids_ != nullptr);
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
            while (updater_live_index_ < updater_live_ids_->size() &&
                    (*updater_live_ids_)[updater_live_index_] < source_id) {
                ++updater_live_index_;
            }
            if (updater_live_index_ < updater_live_ids_->size() &&
                    (*updater_live_ids_)[updater_live_index_] == source_id) {
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
    UpdaterDeletedCursor updater_deleted_;
    const std::vector<uint64_t>* updater_live_ids_ = nullptr;
    size_t updater_live_index_ = 0;
    uint64_t current_id_ = 0;
    bool has_current_ = false;
    CurrentSource current_source_ = CurrentSource::SourceOnly;
};

auto make_data_reader_delta_delete_cursor(
        const DataReader& source,
        const DataReader& updater,
        const std::vector<uint64_t>* updater_live_ids) {
    return DeltaDeleteCursor(
        DataReaderDeletedCursor(source),
        DataReaderDeletedCursor(updater),
        updater_live_ids);
}

auto make_input_reader_delta_delete_cursor(
        const DataReader& source,
        const InputReaderView& updater,
        const std::vector<uint64_t>* updater_live_ids) {
    return DeltaDeleteCursor(
        DataReaderDeletedCursor(source),
        InputReaderDeletedCursor(updater),
        updater_live_ids);
}

template <typename Cursor>
Ret build_compact_accumulator(Cursor cursor, const char* context, size_t reserve_count, CompactIdsAccumulator* out) {
    bool initialized = false;
    uint64_t base = 0;
    for (; !cursor.eof(); cursor.next()) {
        const uint64_t id = cursor.id();
        if (!initialized) {
            out->init(id, reserve_count);
            base = id;
            initialized = true;
        } else if (id - base > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            return Ret(std::string(context) + ": id range exceeds uint32_t");
        }
        out->add(id);
    }
    return Ret(0);
}

float compute_source_output_norm(const DataReader& source, size_t index, uint32_t target_norm_flags) {
    if (target_norm_flags == 0u) {
        return 0.0f;
    }
    if (source.norm_flags() == target_norm_flags) {
        return source.get_norm(index);
    }

    return compute_stored_norm_for_dist(
        source.at(index),
        source.type(),
        source.dim(),
        dist_func_for_data_file_norm_flags(target_norm_flags));
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
        uint32_t target_norm_flags,
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
                const float norm = compute_source_output_norm(source, i, target_norm_flags);
                CHECK(output->write_binary_record(source_id, source.at(i), norm));
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
            const float norm = compute_source_output_norm(source, i, target_norm_flags);
            CHECK(output->write_binary_record(source_id, source.at(i), norm));
            ++i;
        } else {
            // No source rows remain, so every remaining updater row is appended.
            CHECK(updater.write_current(output));
            updater.next();
        }
    }

#ifndef NDEBUG
    if (output->output_count() > 1) {
        assert(output->output_min_id() < output->output_max_id());
    }
#endif
    return Ret(0);
}

// Header min/max/count are derived from the final live-id stream, so they can
// only be filled in after merge_records has finished.
void set_output_id_range(const MergeOutputWriter& output, DataFileHeader* header) {
    header->min_id = output.output_empty() ? 0 : output.output_min_id();
    header->max_id = output.output_empty() ? 0 : output.output_max_id();
    header->count = static_cast<uint32_t>(output.output_count());
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
    if (source.norm_flags() != updater.norm_flags()) {
        return Ret("DataMerger::merge_data_file: incompatible norm layout");
    }
    if (source.has_delta() || updater.has_delta()) {
        return Ret("DataMerger::merge_data_file: source and updater must not have deltas");
    }

    Timer timer("merge_data_file");
    // For a data-file merge, deletes come directly from the updater because the
    // destination is a compact base file with no persisted tombstone section.
    MergeFile merge_file;
    CHECK(merge_file.open(source, source.norm_flags(), path, "DataMerger::merge_data_files"));

    MergeOutputWriter output(merge_file.file(), *merge_file.header(), "DataMerger::merge_data_files");
    output.reserve(source.count() + updater.count(), false);
    CHECK(merge_records(
        source,
        DataReaderUpdaterCursor(updater, source.norm_flags()),
        DataReaderDeletedCursor(updater),
        DataReaderDeletedCursor(updater),
        source.norm_flags(),
        "DataMerger::merge_data_files: updated id is also deleted",
        &output));
    // After all vector records are streamed, write the trailing metadata needed
    // to reopen the new file as a normal compact `.data` file.
    CHECK(output.write_ids_section(
        *merge_file.header(),
        {},
        "DataMerger::merge_data_files: failed to write ids to merge file",
        "DataMerger::merge_data_files: failed to write deleted_ids to merge file"));

    const CompactIds empty_deleted_ids;
    set_output_id_range(output, merge_file.header());
    CHECK(set_data_header_layout(
        merge_file.header(), output.output_ids_bytes(), empty_deleted_ids.serialized_size_bytes()));
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

Ret DataMerger::merge_data_file(const DataReader& source, const InputReaderView& updater,
        const std::string& path, DistFunc dist_func) {
    if (source.dim() != updater.dim() || source.type() != updater.type()) {
        return Ret("DataMerger::merge_data_file: incompatible source and updater");
    }

    // Public overload for direct-input merges. The failure contract intentionally
    // matches the DataReader overload so DatasetWriter can switch between the
    // two without having to special-case cleanup behavior.
    Ret ret(0);
    try {
        ret = merge_data_file_(source, updater, path, dist_func);
    } catch (const std::exception& ex) {
        ret = Ret(ex.what());
    }

    if (ret.code() != 0 && std::filesystem::exists(path)) {
        std::filesystem::remove(path);
    }

    return ret;
}

Ret DataMerger::merge_delta_file(const DataReader& source, const InputReaderView& updater,
        const std::string& path, DistFunc dist_func) {
    if (source.dim() != updater.dim() || source.type() != updater.type()) {
        return Ret("DataMerger::merge_delta_file: incompatible source and updater");
    }

    // Same wrapper contract as the persisted-file overload: exceptions become
    // Ret and partial outputs are removed before the caller sees a result.
    Ret ret(0);
    try {
        ret = merge_delta_file_(source, updater, path, dist_func);
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
    if (source.norm_flags() != updater.norm_flags()) {
        return Ret("DataMerger::merge_delta_file: incompatible norm layout");
    }
    if (source.has_delta() || updater.has_delta()) {
        return Ret("DataMerger::merge_delta_file: source and updater must not have deltas");
    }

    Timer timer("merge_delta_file");
    // Delta-to-delta merge keeps a tombstone section, but it must first remove
    // any old deletes that the updater resurrected as live rows.
    const std::vector<uint64_t> updater_live_ids =
        collect_updater_live_ids(DataReaderUpdaterCursor(updater, source.norm_flags()), updater.count());
    CompactIdsAccumulator compact_deleted_ids_accum;
    CHECK(build_compact_accumulator(
        make_data_reader_delta_delete_cursor(source, updater, &updater_live_ids),
        "DataMerger::merge_delta_file: deleted ids",
        source.deleted_count() + updater.deleted_count(),
        &compact_deleted_ids_accum));
    CompactIds compact_deleted_ids;
    compact_deleted_ids_accum.complete_adding();
    CHECK(compact_deleted_ids.init(compact_deleted_ids_accum));
    MergeFile merge_file;
    CHECK(merge_file.open(source, source.norm_flags(), path, "DataMerger::merge_delta_file"));

    MergeOutputWriter output(merge_file.file(), *merge_file.header(), "DataMerger::merge_delta_file");
    output.reserve(source.count() + updater.count(), false);
    const auto merge_deleted_cursor =
        make_data_reader_delta_delete_cursor(source, updater, &updater_live_ids);
    CHECK(merge_records(
        source,
        DataReaderUpdaterCursor(updater, source.norm_flags()),
        merge_deleted_cursor,
        merge_deleted_cursor,
        source.norm_flags(),
        "DataMerger::merge_delta_file: updated id is also deleted",
        &output));
    // Delta files have the same live-row trailer as data files...
    CHECK(output.write_ids_section(
        *merge_file.header(),
        compact_deleted_ids,
        "DataMerger::merge_delta_file: failed to write ids to merge file",
        "DataMerger::merge_delta_file: failed to write deleted_ids to merge file"));

    merge_file.header()->deleted_count = static_cast<uint32_t>(compact_deleted_ids.count());
    set_output_id_range(output, merge_file.header());
    CHECK(set_data_header_layout(
        merge_file.header(), output.output_ids_bytes(), compact_deleted_ids.serialized_size_bytes()));
    CHECK(rewrite_header(merge_file.file(), *merge_file.header(), "DataMerger::merge_delta_file"));
    Ret ret = merge_file.flush_and_close("DataMerger::merge_delta_file");

    LOG_INFO << "Merged delta file " << source.path() << " in " << timer.elapsed_ms() << " ms";
    return ret;
}

Ret DataMerger::merge_data_file_(const DataReader& source, const InputReaderView& updater,
        const std::string& path, DistFunc dist_func) {
    if (source.dim() != updater.dim() || source.type() != updater.type()) {
        return Ret("DataMerger::merge_data_file: incompatible source and updater");
    }
    if (source.has_delta()) {
        return Ret("DataMerger::merge_data_file: source and updater must not have deltas");
    }

    Timer timer("merge_data_file");
    const CompactIds empty_deleted_ids;
    const uint32_t target_norm_flags = data_file_norm_flags_for_dist(dist_func);
    MergeFile merge_file;
    CHECK(merge_file.open(source, target_norm_flags, path, "DataMerger::merge_data_files"));

    MergeOutputWriter output(merge_file.file(), *merge_file.header(), "DataMerger::merge_data_files");
    output.reserve(source.count() + updater.count(), !updater.is_binary());
#ifndef NDEBUG
    assert((target_norm_flags != 0u) == output.norms_enabled());
#endif
    CHECK(merge_records(
        source,
        InputReaderUpdaterCursor(updater, dist_func, target_norm_flags != 0u),
        InputReaderDeletedCursor(updater),
        InputReaderDeletedCursor(updater),
        target_norm_flags,
        "DataMerger::merge_data_files: updated id is also deleted",
        &output));
    CHECK(output.write_ids_section(
        *merge_file.header(),
        empty_deleted_ids,
        "DataMerger::merge_data_files: failed to write ids to merge file",
        "DataMerger::merge_data_files: failed to write deleted_ids to merge file"));

    set_output_id_range(output, merge_file.header());
    CHECK(set_data_header_layout(
        merge_file.header(), output.output_ids_bytes(), empty_deleted_ids.serialized_size_bytes()));
    CHECK(rewrite_header(merge_file.file(), *merge_file.header(), "DataMerger::merge_data_files"));
    Ret ret = merge_file.flush_and_close("DataMerger::merge_data_files");

    LOG_INFO << "Merged data file " << source.path() << " in " << timer.elapsed_ms() << " ms";
    return ret;
}

Ret DataMerger::merge_delta_file_(const DataReader& source, const InputReaderView& updater,
        const std::string& path, DistFunc dist_func) {
    if (source.dim() != updater.dim() || source.type() != updater.type()) {
        return Ret("DataMerger::merge_delta_file: incompatible source and updater");
    }
    if (source.has_delta()) {
        return Ret("DataMerger::merge_delta_file: source and updater must not have deltas");
    }

    Timer timer("merge_delta_file");
    const uint32_t target_norm_flags = data_file_norm_flags_for_dist(dist_func);
    const bool compute_norms = target_norm_flags != 0u;
    const std::vector<uint64_t> updater_live_ids = collect_updater_live_ids(
        InputReaderUpdaterCursor(updater, dist_func, compute_norms),
        updater.count());
    const size_t updater_deleted_count = count_input_reader_view_deleted_rows(updater);
    CompactIdsAccumulator compact_deleted_ids_accum;
    CHECK(build_compact_accumulator(
        make_input_reader_delta_delete_cursor(source, updater, &updater_live_ids),
        "DataMerger::merge_delta_file: deleted ids",
        source.deleted_count() + updater_deleted_count,
        &compact_deleted_ids_accum));
    CompactIds compact_deleted_ids;
    compact_deleted_ids_accum.complete_adding();
    CHECK(compact_deleted_ids.init(compact_deleted_ids_accum));
    MergeFile merge_file;
    CHECK(merge_file.open(source, target_norm_flags, path, "DataMerger::merge_delta_file"));

    MergeOutputWriter output(merge_file.file(), *merge_file.header(), "DataMerger::merge_delta_file");
    output.reserve(source.count() + updater.count(), !updater.is_binary());
#ifndef NDEBUG
    assert(compute_norms == output.norms_enabled());
#endif
    const auto merge_deleted_cursor =
        make_input_reader_delta_delete_cursor(source, updater, &updater_live_ids);
    CHECK(merge_records(
        source,
        InputReaderUpdaterCursor(updater, dist_func, compute_norms),
        merge_deleted_cursor,
        merge_deleted_cursor,
        target_norm_flags,
        "DataMerger::merge_delta_file: updated id is also deleted",
        &output));
    CHECK(output.write_ids_section(
        *merge_file.header(),
        compact_deleted_ids,
        "DataMerger::merge_delta_file: failed to write ids to merge file",
        "DataMerger::merge_delta_file: failed to write deleted_ids to merge file"));

    merge_file.header()->deleted_count = static_cast<uint32_t>(compact_deleted_ids.count());
    set_output_id_range(output, merge_file.header());
    CHECK(set_data_header_layout(
        merge_file.header(), output.output_ids_bytes(), compact_deleted_ids.serialized_size_bytes()));
    CHECK(rewrite_header(merge_file.file(), *merge_file.header(), "DataMerger::merge_delta_file"));
    Ret ret = merge_file.flush_and_close("DataMerger::merge_delta_file");

    LOG_INFO << "Merged delta file " << source.path() << " in " << timer.elapsed_ms() << " ms";
    return ret;
}

} // namespace sketch2
