// Implements conversion from text or binary input records into the binary data-file format.

#include "data_writer.h"
#include "core/compute/compute.h"
#include "core/storage/input_reader.h"
#include "core/storage/data_file_layout.h"
#include "core/utils/compact_ids_ext.h"
#include "core/utils/compact_ids_shared.h"
#include "core/utils/log.h"
#include "core/utils/shared_consts.h"
#include "core/utils/timer.h"
#include <algorithm>
#include <experimental/scope>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <cassert>
#include <limits>
#include <unistd.h>

namespace sketch2 {

namespace {

struct IdStats {
    uint64_t min_id = std::numeric_limits<uint64_t>::max();
    uint64_t max_id = 0;
    uint32_t active_count = 0;
    uint32_t deleted_count = 0;
};

Ret scan_ids(const InputReaderView& reader, IdStats* stats) {
    const size_t count = reader.count();
    uint64_t prev_id = 0;

    for (size_t i = 0; i < count; ++i) {
        const uint64_t id = reader.id(i);
        if (i > 0 && prev_id >= id) {
            return Ret("Invalid order of ids in input data.");
        }
        prev_id = id;

        if (reader.is_no_data(i)) {
            ++stats->deleted_count;
            continue;
        }

        stats->min_id = std::min(stats->min_id, id);
        stats->max_id = std::max(stats->max_id, id);
        ++stats->active_count;
    }

    if (stats->deleted_count == count) {
        stats->min_id = 0;
        stats->max_id = 0;
    }

    return Ret(0);
}

Ret write_vector_section(
        FILE* f,
        const InputReaderView& reader,
        const DataFileHeader& hdr,
        bool write_cosine_inv_norms,
        std::vector<float>* cosine_inv_norms) {
    const size_t count = reader.count();
    const size_t vec_size = reader.size();
    const bool binary_input = reader.is_binary();
    std::vector<uint8_t> buf = binary_input ? std::vector<uint8_t>() : std::vector<uint8_t>(vec_size);

    if (binary_input) {
        for (size_t i = 0; i < count; ++i) {
            if (reader.is_no_data(i)) {
                continue;
            }

            const uint8_t* vector_data = nullptr;
            CHECK(reader.raw_data(i, &vector_data));
            CHECK(write_vector_record(
                f,
                vector_data,
                vec_size,
                hdr.vector_stride,
                "DataWriter: failed to write vector data at index " + std::to_string(i)));
            if (write_cosine_inv_norms) {
                cosine_inv_norms->push_back(
                    compute_cosine_inverse_norm(vector_data, reader.type(), reader.dim()));
            }
        }
        return Ret(0);
    }

    for (size_t i = 0; i < count; ++i) {
        if (reader.is_no_data(i)) {
            continue;
        }

        CHECK(reader.data(i, buf.data(), buf.size()));
        const uint8_t* vector_data = buf.data();
        CHECK(write_vector_record(
            f,
            vector_data,
            vec_size,
            hdr.vector_stride,
            "DataWriter: failed to write vector data at index " + std::to_string(i)));
        if (write_cosine_inv_norms) {
            cosine_inv_norms->push_back(
                compute_cosine_inverse_norm(vector_data, reader.type(), reader.dim()));
        }
    }

    return Ret(0);
}

Ret build_compact_accum(
        const InputReaderView& reader,
        uint32_t active_count,
        uint32_t deleted_count,
        CompactIdsAccumulator* active_accum,
        CompactIdsAccumulator* deleted_accum) {
    bool active_initialized = false;
    bool deleted_initialized = false;
    uint64_t active_base = 0;
    uint64_t deleted_base = 0;

    for (size_t i = 0; i < reader.count(); ++i) {
        const uint64_t id = reader.id(i);
        if (reader.is_no_data(i)) {
            if (!deleted_initialized) {
                deleted_accum->init(id, deleted_count);
                deleted_base = id;
                deleted_initialized = true;
            } else if (id - deleted_base > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
                return Ret("DataWriter: deleted ids: id range exceeds uint32_t");
            }
            deleted_accum->add(id);
            continue;
        }

        if (!active_initialized) {
            active_accum->init(id, active_count);
            active_base = id;
            active_initialized = true;
        } else if (id - active_base > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            return Ret("DataWriter: active ids: id range exceeds uint32_t");
        }
        active_accum->add(id);
    }

    return Ret(0);
}

Ret finalize_output_file(FILE* f) {
    const int flush_ret = fflush(f);
    const int sync_ret = fsync(fileno(f));
    const int close_ret = fclose(f);
    if (flush_ret != 0 || sync_ret != 0 || close_ret != 0) {
        return Ret("DataWriter: failed to flush and close file");
    }
    return Ret(0);
}

} // namespace

Ret DataWriter::init(const std::string& input_path, const std::string& output_path,
    uint64_t start, uint64_t end, bool write_cosine_inv_norms) {

    input_path_  = input_path;
    output_path_ = output_path;
    start_ = start;
    end_ = end;
    write_cosine_inv_norms_ = write_cosine_inv_norms;
    return Ret(0);
}

Ret DataWriter::exec_for_testing() {
    if (input_path_.empty()) {
        return Ret("Input path is not set.");
    }
    if (output_path_.empty()) {
        return Ret("Output path is not set.");
    }

    Timer timer("data_writer::exec");

    // Create and init InputReader from input_path
    InputReader source;
    CHECK(source.init(input_path_));

    InputReaderView reader(source, start_, end_);
    Ret ret = write(reader, output_path_, write_cosine_inv_norms_);

    LOG_INFO << "DataWriter completed exec_for_testing for " << output_path_ << " in " << timer.elapsed_ms() << " ms";
    return ret;
}

// Converts a sorted text-or-binary input view into the binary on-disk data-file format.
// It separates live ids from deletions, streams vectors into the aligned data
// section, optionally persists cosine inverse norms, and then appends both id tables.
Ret DataWriter::write(const InputReaderView& reader, const std::string& output_path, bool write_cosine_inv_norms) {
    const size_t count = reader.count();
    if (count == 0) {
        return Ret("Invalid count of vectors in reader.");
    }

    IdStats stats;
    CHECK(scan_ids(reader, &stats));

    CompactIdsExt active_ids_ext;
    CompactIdsExt deleted_ids_ext;

    {
        CompactIdsAccumulator active_accum;
        CompactIdsAccumulator deleted_accum;
        CHECK(build_compact_accum(
            reader, stats.active_count, stats.deleted_count, &active_accum, &deleted_accum));

        CHECK(active_ids_ext.init(active_accum));
        CHECK(deleted_ids_ext.init(deleted_accum));
    }

    // Build DataFileHeader
    DataFileHeader hdr = make_data_header(
        stats.min_id,
        stats.max_id,
        stats.active_count,
        stats.deleted_count,
        reader.type(),
        static_cast<uint16_t>(reader.dim()),
        write_cosine_inv_norms);
    CHECK(set_data_header_layout(
        &hdr,
        active_ids_ext.serialized_size_bytes(),
        deleted_ids_ext.serialized_size_bytes()));

    // Write output file
    FILE *f = fopen(output_path.c_str(), "wb");
    if (!f) {
        return Ret("DataWriter: failed to open output file: " + output_path);
    }

    // Use a larger stdio buffer to reduce write-related syscalls for large datasets.
    std::vector<char> file_buffer(kFileBufferSize);
    (void)setvbuf(f, file_buffer.data(), _IOFBF, file_buffer.size());

    std::experimental::scope_exit file_guard([&f]() {
        if (f) fclose(f);
    });

    // Write header
    static_assert(sizeof(hdr) % 8 == 0);
    CHECK(write_header_and_data_padding(f, hdr, "DataWriter"));

    const DataMetadataLayout metadata_layout = compute_data_metadata_layout(hdr, stats.active_count);
    std::vector<float> cosine_inv_norms;
    if (write_cosine_inv_norms) {
        cosine_inv_norms.reserve(stats.active_count);
    }
    CHECK(write_vector_section(f, reader, hdr, write_cosine_inv_norms, &cosine_inv_norms));
    CHECK(write_zero_padding(f, metadata_layout.vectors_padding,
        "DataWriter: failed to write cosine alignment padding"));

#ifndef NDEBUG
    assert(!write_cosine_inv_norms || cosine_inv_norms.size() == stats.active_count);
#endif
    CHECK(write_f32_array(f, cosine_inv_norms,
        "DataWriter: failed to write cosine inverse norms"));
    CHECK(write_zero_padding(f, metadata_layout.ids_trailer_padding,
        "DataWriter: failed to write id alignment padding"));

    // Write compact id sections (active ids then deleted ids).
    CHECK(active_ids_ext.write(f, "DataWriter: failed to write ids"));
    CHECK(write_zero_padding(f,
        compute_deleted_ids_padding(metadata_layout.ids_trailer_offset, active_ids_ext.serialized_size_bytes()),
        "DataWriter: failed to write deleted_ids alignment padding"));
    CHECK(deleted_ids_ext.write(f, "DataWriter: failed to write deleted_ids"));

#ifndef NDEBUG
    const size_t ids_trailer_size =
        active_ids_ext.serialized_size_bytes()
        + compute_deleted_ids_padding(metadata_layout.ids_trailer_offset, active_ids_ext.serialized_size_bytes())
        + deleted_ids_ext.serialized_size_bytes();
    const long file_pos_after_ids = ftell(f);
    const long expected_file_pos_after_ids =
        static_cast<long>(metadata_layout.ids_trailer_offset + ids_trailer_size);
    assert(file_pos_after_ids == expected_file_pos_after_ids);
#endif

    const Ret finalize_ret = finalize_output_file(f);
    f = nullptr;
    if (finalize_ret.code() != 0) {
        return finalize_ret;
    }

    return Ret(0);
}

} // namespace sketch2
