// Implements conversion from text or binary input records into the binary data-file format.

#include "data_writer.h"
#include "core/compute/compute.h"
#include "core/storage/input_reader.h"
#include "core/storage/data_file_layout.h"
#include "core/utils/compact_ids.h"
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

Ret DataWriter::init(const std::string& input_path, const std::string& output_path,
    uint64_t start, uint64_t end, bool write_cosine_inv_norms) {

    input_path_  = input_path;
    output_path_ = output_path;
    start_ = start;
    end_ = end;
    write_cosine_inv_norms_ = write_cosine_inv_norms;
    return Ret(0);
}

Ret DataWriter::exec() {
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
    Ret ret = load(reader, output_path_, write_cosine_inv_norms_);

    LOG_INFO << "DataWriter completed exec for " << output_path_ << " in " << timer.elapsed_ms() << " ms";
    return ret;
}

// Converts a sorted text-or-binary input view into the binary on-disk data-file format.
// It separates live ids from deletions, streams vectors into the aligned data
// section, optionally persists cosine inverse norms, and then appends both id tables.
Ret DataWriter::load(const InputReaderView& reader, const std::string& output_path, bool write_cosine_inv_norms) {
    const size_t count = reader.count();
    if (count == 0) {
        return Ret("Invalid count of vectors in reader.");
    }

    CompactIdsBuilder active_ids;
    CompactIdsBuilder deleted_ids;
    active_ids.reserve(count);
    deleted_ids.reserve(count);

    uint64_t prev_id = 0;
    uint64_t min_id = std::numeric_limits<uint64_t>::max();
    uint64_t max_id = 0;
    for (size_t i = 0; i < count; ++i) {
        const uint64_t id = reader.id(i);
        if (i > 0 && prev_id >= id) {
            return Ret("Invalid order of ids in input data.");
        }
        prev_id = id;
        if (reader.is_no_data(i)) {
            const Ret append_ret = deleted_ids.append(id);
            if (append_ret.code() != 0) {
                return Ret("DataWriter: deleted ids: " + append_ret.message());
            }
        } else {
            if (id < min_id) {
                min_id = id;
            }
            if (id > max_id) {
                max_id = id;
            }
            const Ret append_ret = active_ids.append(id);
            if (append_ret.code() != 0) {
                return Ret("DataWriter: active ids: " + append_ret.message());
            }
        }
    }

    if (deleted_ids.count() == count) {
        min_id = max_id = 0;
    }

    // Build DataFileHeader
    DataFileHeader hdr = make_data_header(
        min_id,
        max_id,
        static_cast<uint32_t>(active_ids.count()),
        static_cast<uint32_t>(deleted_ids.count()),
        reader.type(),
        static_cast<uint16_t>(reader.dim()),
        write_cosine_inv_norms);

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

    // Write vector data
    const size_t vec_size = reader.size();
    const DataMetadataLayout metadata_layout = compute_data_metadata_layout(hdr, active_ids.count());
    const bool binary_input = reader.is_binary();
    std::vector<uint8_t> buf = binary_input ? std::vector<uint8_t>() : std::vector<uint8_t>(vec_size);
    std::vector<float> cosine_inv_norms;
    if (write_cosine_inv_norms) {
        cosine_inv_norms.reserve(active_ids.count());
    }

    if (binary_input) {
        for (size_t i = 0; i < count; ++i) {
            if (reader.is_no_data(i)) {
                continue;
            }

            const uint8_t* vector_data = nullptr;
            CHECK(reader.raw_data(i, &vector_data));
            CHECK(write_vector_record(f, vector_data, vec_size, hdr.vector_stride,
                "DataWriter: failed to write vector data at index " + std::to_string(i)));
            if (write_cosine_inv_norms) {
                cosine_inv_norms.push_back(compute_cosine_inverse_norm(vector_data, reader.type(), reader.dim()));
            }
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            if (!reader.is_no_data(i)) {
                const uint8_t* vector_data = nullptr;
                CHECK(reader.data(i, buf.data(), buf.size()));
                vector_data = buf.data();
                CHECK(write_vector_record(f, vector_data, vec_size, hdr.vector_stride,
                    "DataWriter: failed to write vector data at index " + std::to_string(i)));
                if (write_cosine_inv_norms) {
                    cosine_inv_norms.push_back(compute_cosine_inverse_norm(vector_data, reader.type(), reader.dim()));
                }
            }
        }
    }

#ifndef NDEBUG
    assert(!write_cosine_inv_norms || cosine_inv_norms.size() == active_ids.count());
#endif
    CHECK(write_f32_array(f, cosine_inv_norms,
        "DataWriter: failed to write cosine inverse norms"));
    CHECK(write_zero_padding(f, metadata_layout.ids_trailer_padding,
        "DataWriter: failed to write id alignment padding"));

    // Write compact id sections (active ids then deleted ids).
    CHECK(active_ids.write(f, "DataWriter: failed to write ids"));
    CHECK(deleted_ids.write(f, "DataWriter: failed to write deleted_ids"));

#ifndef NDEBUG
    const size_t ids_trailer_size =
        active_ids.serialized_size_bytes() + deleted_ids.serialized_size_bytes();
    const long file_pos_after_ids = ftell(f);
    const long expected_file_pos_after_ids =
        static_cast<long>(metadata_layout.ids_trailer_offset + ids_trailer_size);
    assert(file_pos_after_ids == expected_file_pos_after_ids);
#endif

    int n1 = fflush(f);
    int n2 = fsync(fileno(f));
    int n3 = fclose(f);
    f = nullptr;
    if (n1 != 0 || n2 != 0 || n3 != 0) {
        return Ret("DataWriter: failed to flush and close file");
    }

    return Ret(0);
}

} // namespace sketch2
