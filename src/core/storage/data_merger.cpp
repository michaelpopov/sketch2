// Implements merge operations for base data files and delta files.

#include "data_merger.h"
#include "core/compute/norm_utils.h"
#include "core/storage/count_utils.h"
#include "core/storage/data_file_layout.h"
#include "core/storage/input_reader.h"
#include "core/utils/log.h"
#include "core/bitset/roaring_ids.h"
#include "core/utils/shared_consts.h"
#include "core/utils/string_utils.h"
#include "core/utils/timer.h"
#include <algorithm>
#include <cassert>
#include <filesystem>
#include <limits>
#include <utility>

namespace sketch2 {

namespace {

// Scratch buffer size for the streaming output ids builder. Big enough to let
// load_unbuffered see long runs of consecutive ids across batch boundaries
// (and emit them as roaring_bitmap_add_range_closed), small enough to stay
// cache-friendly at 512 KB.
constexpr size_t kOutputIdsBufferSize = 64 * 1024;

float compute_stored_norm_for_dist(const uint8_t* data, DataType type, size_t dim, DistFunc dist_func) {
    switch (dist_func) {
        case DistFunc::COS:
            return static_cast<float>(inverse_norm(data, type, dim));
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
        CHECK(out_.open(path, context));
        header_ = make_data_header(
            0, 0, source.min_range_id(), 0, 0, source.type(), static_cast<uint16_t>(source.dim()), norm_flags);
        return write_header_and_data_padding(out_.file(), header_, context);
    }

    FILE* file() const { return out_.file(); }
    DataFileHeader* header() { return &header_; }

    Ret flush_and_close(const char* context) {
        return out_.flush_and_close(context);
    }

private:
    OutputFile out_;
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
          norms_enabled_(data_file_has_norms(header)) {
        // Buffered ingest with needs_sorting=false: write_binary_record's
        // strictly-increasing check below runs before each add(), so the
        // builder never sees an out-of-order batch.
        const Ret ret = output_ids_builder_.init_buffered(
            min_range_id_, kOutputIdsBufferSize, /*needs_sorting=*/false);
        if (ret.code() != 0) {
            throw std::runtime_error(ret.message());
        }
    }

    // The text scratch buffer is only needed for direct-input text merges.
    void reserve(bool needs_text_buffer) {
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
        if (output_count_ == 0) {
            output_ids_min_ = id;
            output_ids_max_ = id;
        } else {
            if (id <= output_ids_max_) {
                return Ret(std::string(context_) + ": active ids: ids must be strictly increasing");
            }
            output_ids_max_ = id;
        }
        CHECK(output_ids_builder_.add(id));
        ++output_count_;
        CHECK(write_data_record(
            f_, data, record_layout_, norms_enabled_ ? &norm : nullptr, context_, &payload_crc32_));
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
    // file: alignment padding before ids, then the Roaring active/deleted id
    // sections. Norms live inside each record, so there is no separate norms
    // section between vectors and ids.
    Ret write_ids_section(const DataFileHeader& header,
            const RoaringIds& deleted_ids) {
        output_ids_ = std::move(output_ids_builder_).build();
        const DataMetadataLayout metadata_layout = compute_data_metadata_layout(header, output_count_);
        const RoaringIdsTrailerLayout trailer_layout = compute_roaring_ids_trailer_layout(
            metadata_layout.ids_trailer_offset, output_ids_, deleted_ids);
        output_ids_bytes_ = trailer_layout.ids_bytes;
        deleted_ids_bytes_ = trailer_layout.deleted_ids_bytes;
        CHECK(write_zero_padding(f_, metadata_layout.vectors_padding,
            std::string(context_) + ": failed to write ids alignment padding", &payload_crc32_));
        CHECK(write_roaring_ids_trailer_mmap(
            f_,
            output_ids_,
            deleted_ids,
            trailer_layout,
            context_,
            &payload_crc32_));
        return Ret(0);
    }

    size_t output_count() const { return output_count_; }
    bool output_empty() const { return output_count_ == 0; }
    uint64_t output_min_id() const { return output_ids_min_; }
    uint64_t output_max_id() const { return output_ids_max_; }
    bool norms_enabled() const { return norms_enabled_; }
    size_t output_ids_bytes() const { return output_ids_bytes_; }
    size_t deleted_ids_bytes() const { return deleted_ids_bytes_; }
    uint32_t payload_crc32() const { return payload_crc32_; }

private:
    FILE* f_ = nullptr;
    DataType type_ = DataType::f32;
    uint16_t dim_ = 0;
    uint64_t min_range_id_ = 0;
    DataRecordLayout record_layout_{};
    const char* context_ = "";
    bool norms_enabled_ = false;
    RoaringIdsBuilder output_ids_builder_;
    RoaringIds output_ids_;
    size_t output_ids_bytes_ = 0;
    size_t deleted_ids_bytes_ = 0;
    size_t output_count_ = 0;
    uint64_t output_ids_min_ = 0;
    uint64_t output_ids_max_ = 0;
    uint32_t payload_crc32_ = 0;
    std::vector<uint8_t> parsed_text_buffer_;
};

class DataReaderLiveRowCursor {
public:
    // Merge entry points reject readers with attached deltas, so base_begin()
    // is equivalent to a full live-row scan here and keeps id access sequential.
    explicit DataReaderLiveRowCursor(const DataReader& reader, uint32_t target_norm_flags)
        : reader_(&reader), iter_(reader.base_begin()), target_norm_flags_(target_norm_flags) {}

    bool eof() const { return iter_.eof(); }
    uint64_t id() const { return iter_.id(); }
    void next() { iter_.next(); }

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
            ++index_;
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
            ++index_;
            advance_to_live_row();
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

template <typename SourceDeletedCursor, typename UpdaterDeletedCursor, typename UpdaterLiveCursor>
class DeltaDeleteCursor {
public:
    DeltaDeleteCursor(SourceDeletedCursor source_deleted,
            UpdaterDeletedCursor updater_deleted,
            UpdaterLiveCursor updater_live)
        : source_deleted_(std::move(source_deleted)),
          updater_deleted_(std::move(updater_deleted)),
          updater_live_(std::move(updater_live)) {
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
            while (!updater_live_.eof() && updater_live_.id() < source_id) {
                updater_live_.next();
            }
            if (!updater_live_.eof() && updater_live_.id() == source_id) {
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
    UpdaterLiveCursor updater_live_;
    uint64_t current_id_ = 0;
    bool has_current_ = false;
    CurrentSource current_source_ = CurrentSource::SourceOnly;
};

auto make_input_reader_delta_delete_cursor(
        const DataReader& source,
        const InputReaderView& updater,
        DistFunc dist_func,
        bool compute_norms) {
    return DeltaDeleteCursor(
        DataReaderDeletedCursor(source),
        InputReaderDeletedCursor(updater),
        InputReaderUpdaterCursor(updater, dist_func, compute_norms));
}

class RoaringIdsCursor {
public:
    explicit RoaringIdsCursor(const RoaringIds& ids) : iter_(ids.begin()) {}

    bool eof() const { return iter_.eof(); }
    uint64_t id() const { return iter_.id(); }
    void next() {
        if (!iter_.eof()) {
            iter_.next();
        }
    }

private:
    RoaringIds::Iterator iter_;
};

// Bitmap-algebra fast path for the delta-to-delta delete merge when both
// source and updater are DataReaders. The DeltaDeleteCursor logic
//   out = (source.deleted - {ids live in updater}) ∪ updater.deleted
// becomes one andnot + one or, both inplace and operating on whole bitmaps.
//
// Precondition: source.min_range_id() == updater.min_range_id() == base
// (callers verify this; merge_delta_file_ rejects mismatched ranges before
// reaching this code).
Ret build_delta_delete_roaring_ids(
        const DataReader& source,
        const DataReader& updater,
        uint64_t base,
        const char* context,
        RoaringIds* out) {
    auto wrap = [context](const Ret& ret) -> Ret {
        if (ret.code() == 0) {
            return ret;
        }
        return Ret(std::string(context) + ": " + ret.message());
    };
    RoaringIdsBuilder builder;
    CHECK(wrap(builder.init_copy(source.deleted_ids(), base)));
    CHECK(wrap(builder.andnot_in_place(updater.ids())));
    CHECK(wrap(builder.union_in_place(updater.deleted_ids())));
    *out = std::move(builder).build();
    return Ret(0);
}

template <typename Cursor>
Ret build_roaring_ids(Cursor cursor, uint64_t base, const char* context, RoaringIds* out) {
    RoaringIdsBuilder builder;
    CHECK(builder.init(base));
    bool have_prev = false;
    uint64_t prev_id = 0;
    for (; !cursor.eof(); cursor.next()) {
        const uint64_t id = cursor.id();
        if (have_prev && id <= prev_id) {
            return Ret(std::string(context) + ": ids must be strictly increasing");
        }
        if (id < base) {
            return Ret(std::string(context) + ": id is below min_range_id");
        }
        if (id - base > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            return Ret(std::string(context) + ": id range exceeds uint32_t");
        }
        CHECK(builder.add(id));
        prev_id = id;
        have_prev = true;
    }
    *out = std::move(builder).build();
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
    // source_rows -> current live row in the persisted source file
    // updater     -> current live row in the updater stream
    // source_deletes  -> current delete id checked against source ids
    // updater_deletes -> current delete id checked against updater ids
    //
    // The streams are all sorted, so each cursor only moves forward. That keeps
    // the merge linear in the total number of source rows, update rows, and
    // delete ids.
    DataReaderLiveRowCursor source_rows(source, target_norm_flags);
    for (; !source_rows.eof() || !updater.eof(); ) {
        const bool has_source = !source_rows.eof();
        const bool has_update = !updater.eof();

        if (has_source) {
            const uint64_t source_id = source_rows.id();
            // A delete means "this id must not appear as a live row". When the
            // current source row is deleted, we simply skip it and keep merging.
            while (!source_deletes.eof() && source_deletes.id() < source_id) {
                source_deletes.next();
            }
            if (!source_deletes.eof() && source_deletes.id() == source_id) {
                source_rows.next();
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
            const uint64_t source_id = source_rows.id();
            const uint64_t update_id = updater.id();
            if (source_id < update_id) {
                // Source id comes first and is not shadowed by an update.
                CHECK(source_rows.write_current(output));
                source_rows.next();
            } else if (source_id > update_id) {
                // Updater inserted a new id before the current source id.
                CHECK(updater.write_current(output));
                updater.next();
            } else {
                // Same id in both streams means "replace source with updater".
                CHECK(updater.write_current(output));
                source_rows.next();
                updater.next();
            }
            continue;
        }

        if (has_source) {
            // No updater rows remain, so every remaining source row survives.
            CHECK(source_rows.write_current(output));
            source_rows.next();
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
Ret set_output_id_range(const MergeOutputWriter& output, DataFileHeader* header) {
    header->min_id = output.output_empty() ? 0 : output.output_min_id();
    header->max_id = output.output_empty() ? 0 : output.output_max_id();
    return checked_size_to_uint32(
        output.output_count(),
        &header->count,
        "DataMerger: active id count exceeds uint32_t");
}

// Writes the trailer + final header for a merged file and flushes it durably.
// Used by all four merge variants; for compact data merges, deleted_ids is
// empty and deleted_count ends up zero.
Ret finalize_merge_file(MergeFile* merge_file,
        MergeOutputWriter* output,
        const RoaringIds& deleted_ids,
        const char* context) {
    CHECK(checked_size_to_uint32(
        deleted_ids.count(),
        &merge_file->header()->deleted_count,
        "DataMerger: deleted id count exceeds uint32_t"));
    CHECK(set_output_id_range(*output, merge_file->header()));
    CHECK(output->write_ids_section(*merge_file->header(), deleted_ids));
    CHECK(set_data_header_layout(
        merge_file->header(), output->output_ids_bytes(), output->deleted_ids_bytes()));
    merge_file->header()->flags |= kDataFileHasPayloadCrc32;
    merge_file->header()->payload_crc32 = output->payload_crc32();
    CHECK(rewrite_header(merge_file->file(), *merge_file->header(), context));
    return merge_file->flush_and_close(context);
}

// Wraps a merge body with the standard failure contract: convert exceptions to
// Ret, and remove any partial output file so callers either get a complete file
// or no file.
template <typename Fn>
Ret run_merge(const std::string& path, Fn&& fn) {
    Ret ret(0);
    try {
        ret = fn();
    } catch (const std::exception& ex) {
        ret = Ret(ex.what());
    }
    if (ret.code() != 0 && std::filesystem::exists(path)) {
        std::filesystem::remove(path);
    }
    return ret;
}

} // namespace

Ret DataMerger::merge_data_file(const DataReader& source, const DataReader& updater, const std::string& path) {
    return run_merge(path, [&]() { return merge_data_file_(source, updater, path); });
}

// Rewrites a full data file by merging persisted rows with another sorted file
// of updates/deletes, producing a compact output with no tombstone section.
Ret DataMerger::merge_data_file_(const DataReader& source, const DataReader& updater, const std::string& path) {
    if (source.dim() != updater.dim() || source.type() != updater.type()) {
        return Ret("DataMerger::merge_data_file: incompatible source and updater");
    }
    if (source.min_range_id() != updater.min_range_id()) {
        return Ret("DataMerger::merge_data_file: incompatible source and updater range");
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
    output.reserve(false);
    CHECK(merge_records(
        source,
        DataReaderLiveRowCursor(updater, source.norm_flags()),
        DataReaderDeletedCursor(updater),
        DataReaderDeletedCursor(updater),
        source.norm_flags(),
        "DataMerger::merge_data_files: updated id is also deleted",
        &output));
    // After all vector records are streamed, write the trailing metadata needed
    // to reopen the new file as a normal compact `.data` file.
    const RoaringIds empty_deleted_ids;
    const Ret ret = finalize_merge_file(&merge_file, &output, empty_deleted_ids, "DataMerger::merge_data_files");

    LOG_INFO << "Merged data file " << source.path() << " in " << timer.elapsed_ms() << " ms";
    return ret;
}

Ret DataMerger::merge_delta_file(const DataReader& source, const DataReader& updater, const std::string& path) {
    return run_merge(path, [&]() { return merge_delta_file_(source, updater, path); });
}

Ret DataMerger::merge_data_file(const DataReader& source, const InputReaderView& updater,
        const std::string& path, DistFunc dist_func) {
    return run_merge(path, [&]() { return merge_data_file_(source, updater, path, dist_func); });
}

Ret DataMerger::merge_delta_file(const DataReader& source, const InputReaderView& updater,
        const std::string& path, DistFunc dist_func) {
    return run_merge(path, [&]() { return merge_delta_file_(source, updater, path, dist_func); });
}

// Rewrites a delta file while preserving delta semantics: live updates stay in
// the record stream and the merged tombstone set is carried forward separately.
Ret DataMerger::merge_delta_file_(const DataReader& source, const DataReader& updater, const std::string& path) {
    if (source.dim() != updater.dim() || source.type() != updater.type()) {
        return Ret("DataMerger::merge_delta_file: incompatible source and updater");
    }
    if (source.min_range_id() != updater.min_range_id()) {
        return Ret("DataMerger::merge_delta_file: incompatible source and updater range");
    }
    if (source.norm_flags() != updater.norm_flags()) {
        return Ret("DataMerger::merge_delta_file: incompatible norm layout");
    }
    if (source.has_delta() || updater.has_delta()) {
        return Ret("DataMerger::merge_delta_file: source and updater must not have deltas");
    }

    Timer timer("merge_delta_file");
    // Delta-to-delta merge keeps a tombstone section, but it must first remove
    // any old deletes that the updater resurrected as live rows. With both
    // sides backed by RoaringIds, this is pure set algebra on whole bitmaps.
    RoaringIds roaring_deleted_ids;
    CHECK(build_delta_delete_roaring_ids(
        source, updater,
        source.min_range_id(),
        "DataMerger::merge_delta_file: deleted ids",
        &roaring_deleted_ids));
    MergeFile merge_file;
    CHECK(merge_file.open(source, source.norm_flags(), path, "DataMerger::merge_delta_file"));

    MergeOutputWriter output(merge_file.file(), *merge_file.header(), "DataMerger::merge_delta_file");
    output.reserve(false);
    CHECK(merge_records(
        source,
        DataReaderLiveRowCursor(updater, source.norm_flags()),
        RoaringIdsCursor(roaring_deleted_ids),
        RoaringIdsCursor(roaring_deleted_ids),
        source.norm_flags(),
        "DataMerger::merge_delta_file: updated id is also deleted",
        &output));
    // Delta files have the same live-row trailer as data files...
    const Ret ret = finalize_merge_file(
        &merge_file, &output, roaring_deleted_ids, "DataMerger::merge_delta_file");

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
    const RoaringIds empty_deleted_ids;
    const uint32_t target_norm_flags = data_file_norm_flags_for_dist(dist_func);
    MergeFile merge_file;
    CHECK(merge_file.open(source, target_norm_flags, path, "DataMerger::merge_data_files"));

    MergeOutputWriter output(merge_file.file(), *merge_file.header(), "DataMerger::merge_data_files");
    output.reserve(!updater.is_binary());
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
    const Ret ret = finalize_merge_file(
        &merge_file, &output, empty_deleted_ids, "DataMerger::merge_data_files");

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
    RoaringIds roaring_deleted_ids;
    CHECK(build_roaring_ids(
        make_input_reader_delta_delete_cursor(source, updater, dist_func, compute_norms),
        source.min_range_id(),
        "DataMerger::merge_delta_file: deleted ids",
        &roaring_deleted_ids));
    MergeFile merge_file;
    CHECK(merge_file.open(source, target_norm_flags, path, "DataMerger::merge_delta_file"));

    MergeOutputWriter output(merge_file.file(), *merge_file.header(), "DataMerger::merge_delta_file");
    output.reserve(!updater.is_binary());
#ifndef NDEBUG
    assert(compute_norms == output.norms_enabled());
#endif
    CHECK(merge_records(
        source,
        InputReaderUpdaterCursor(updater, dist_func, compute_norms),
        RoaringIdsCursor(roaring_deleted_ids),
        RoaringIdsCursor(roaring_deleted_ids),
        target_norm_flags,
        "DataMerger::merge_delta_file: updated id is also deleted",
        &output));
    const Ret ret = finalize_merge_file(
        &merge_file, &output, roaring_deleted_ids, "DataMerger::merge_delta_file");

    LOG_INFO << "Merged delta file " << source.path() << " in " << timer.elapsed_ms() << " ms";
    return ret;
}

} // namespace sketch2
