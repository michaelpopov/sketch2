// Implements conversion from text or binary input records into the binary data-file format.

#include "data_writer.h"
#include "core/compute/norm_utils.h"
#include "core/storage/input_reader.h"
#include "core/storage/data_file_layout.h"
#include "core/utils/log.h"
#include "core/utils/roaring_ids.h"
#include "core/utils/shared_consts.h"
#include "core/utils/timer.h"
#include <algorithm>
#include <cstdint>
#include <vector>
#include <cassert>
#include <limits>

namespace sketch2 {

namespace {

struct IdStats {
    uint64_t min_id = std::numeric_limits<uint64_t>::max();
    uint64_t max_id = 0;
    uint32_t active_count = 0;
    uint32_t deleted_count = 0;
};

Ret validate_active_range_u32(const IdStats& stats, uint64_t min_range_id) {
    if (stats.active_count == 0) {
        return Ret(0);
    }

    if (stats.min_id < min_range_id) {
        return Ret("DataWriter: active ids: min id is below min_range_id");
    }
    if (stats.max_id - min_range_id > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return Ret("DataWriter: data file range exceeds uint32_t");
    }

    return Ret(0);
}

Ret write_vector_section(
        FILE* f,
        const InputReaderView& reader,
        const DataFileHeader& hdr,
        DistFunc dist_func) {
    const size_t count = reader.count();
    const bool binary_input = reader.is_binary();
    const DataRecordLayout record_layout =
        compute_data_record_layout(reader.type(), static_cast<uint16_t>(reader.dim()), data_file_has_norms(hdr));
    std::vector<uint8_t> buf = binary_input ? std::vector<uint8_t>() : std::vector<uint8_t>(record_layout.vector_size);
    const auto compute_norm = [&](const uint8_t* vector_data, float* norm_value) {
        switch (dist_func) {
            case DistFunc::COS:
                *norm_value = static_cast<float>(
                    inverse_norm(vector_data, reader.type(), reader.dim()));
                return;
            case DistFunc::L2:
                *norm_value = static_cast<float>(
                    compute_squared_norm(vector_data, reader.type(), reader.dim()));
                return;
            case DistFunc::DOT:
                return;
            default:
                throw std::runtime_error("write_vector_section: unsupported distance function");
        }
    };

    for (size_t i = 0; i < count; ++i) {
        if (reader.is_no_data(i)) {
            continue;
        }

        const uint8_t* vector_data = nullptr;
        if (binary_input) {
            CHECK(reader.raw_data(i, &vector_data));
        } else {
            CHECK(reader.data(i, buf.data(), buf.size()));
            vector_data = buf.data();
        }
        float norm_value = 0.0f;
        float* norm_ptr = nullptr;
        if (data_file_has_norms(hdr)) {
            compute_norm(vector_data, &norm_value);
            norm_ptr = &norm_value;
        }
        CHECK(write_data_record(
            f,
            vector_data,
            record_layout,
            norm_ptr,
            "DataWriter: failed to write vector data at index " + std::to_string(i)));
    }

    return Ret(0);
}

Ret increment_count(uint32_t* count, const char* error_message) {
    if (*count == std::numeric_limits<uint32_t>::max()) {
        return Ret(error_message);
    }
    ++(*count);
    return Ret(0);
}

Ret build_roaring_ids_and_stats(
        const InputReaderView& reader,
        uint64_t min_range_id,
        IdStats* stats,
        RoaringIds* active_ids,
        RoaringIds* deleted_ids) {
    if (stats == nullptr) {
        return Ret("DataWriter: missing id stats output");
    }
    CHECK(active_ids->init_writable(min_range_id));
    CHECK(deleted_ids->init_writable(min_range_id));

    uint64_t prev_id = 0;

    for (size_t i = 0; i < reader.count(); ++i) {
        const uint64_t id = reader.id(i);
        if (i > 0 && prev_id >= id) {
            return Ret("Invalid order of ids in input data.");
        }
        prev_id = id;

        if (reader.is_no_data(i)) {
            if (id < min_range_id) {
                return Ret("DataWriter: deleted ids: id is below min_range_id");
            }
            if (id - min_range_id > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
                return Ret("DataWriter: deleted ids: id range exceeds uint32_t");
            }
            CHECK(increment_count(
                &stats->deleted_count,
                "DataWriter: deleted id count exceeds uint32_t"));
            CHECK(deleted_ids->add(id));
            continue;
        }

        if (id < min_range_id) {
            return Ret("DataWriter: active ids: min id is below min_range_id");
        }
        if (id - min_range_id > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            return Ret("DataWriter: data file range exceeds uint32_t");
        }
        CHECK(increment_count(
            &stats->active_count,
            "DataWriter: active id count exceeds uint32_t"));
        stats->min_id = std::min(stats->min_id, id);
        stats->max_id = std::max(stats->max_id, id);
        CHECK(active_ids->add(id));
    }

    if (stats->active_count == 0) {
        stats->min_id = 0;
        stats->max_id = 0;
    }

    active_ids->compact();
    deleted_ids->compact();
    return Ret(0);
}

} // namespace

Ret DataWriter::exec_for_testing(const std::string& input_path, const std::string& output_path,
    uint64_t min_range_id, uint64_t start, uint64_t end, DistFunc dist_func) {
    if (input_path.empty()) {
        return Ret("Input path is not set.");
    }
    if (output_path.empty()) {
        return Ret("Output path is not set.");
    }

    Timer timer("data_writer::exec");

    // Create and init InputReader from input_path
    InputReader source;
    CHECK(source.init(input_path));

    InputReaderView reader(source, start, end);
    Ret ret = write(reader, output_path, dist_func, min_range_id);

    LOG_INFO << "DataWriter completed exec_for_testing for " << output_path
             << " in " << timer.elapsed_ms() << " ms";
    return ret;
}

Ret DataWriter::write(const InputReaderView& reader, const std::string& output_path, DistFunc dist_func,
        uint64_t min_range_id) {
    const size_t count = reader.count();
    if (count == 0) {
        return Ret("Invalid count of vectors in reader.");
    }

    IdStats stats;
    RoaringIds active_ids;
    RoaringIds deleted_ids;
    CHECK(build_roaring_ids_and_stats(reader, min_range_id, &stats, &active_ids, &deleted_ids));
    CHECK(validate_active_range_u32(stats, min_range_id));
    const size_t active_ids_bytes = serialized_bytes_or_zero(active_ids);
    const size_t deleted_ids_bytes = serialized_bytes_or_zero(deleted_ids);

    const uint32_t norm_flags = data_file_norm_flags_for_dist(dist_func);
    // Build DataFileHeader
    DataFileHeader hdr = make_data_header(
        stats.min_id,
        stats.max_id,
        min_range_id,
        stats.active_count,
        stats.deleted_count,
        reader.type(),
        static_cast<uint16_t>(reader.dim()),
        norm_flags);
    CHECK(set_data_header_layout(
        &hdr,
        active_ids_bytes,
        deleted_ids_bytes));

    OutputFile out;
    CHECK(out.open(output_path, "DataWriter"));

    static_assert(sizeof(hdr) % 8 == 0);
    CHECK(write_header_and_data_padding(out.file(), hdr, "DataWriter"));

    const DataMetadataLayout metadata_layout = compute_data_metadata_layout(hdr, stats.active_count);
    const RoaringIdsTrailerLayout trailer_layout = compute_roaring_ids_trailer_layout(
        metadata_layout.ids_trailer_offset, active_ids_bytes, deleted_ids_bytes);
    CHECK(write_vector_section(out.file(), reader, hdr, dist_func));
    CHECK(write_zero_padding(out.file(), metadata_layout.vectors_padding,
        "DataWriter: failed to write ids alignment padding"));

    CHECK(write_roaring_ids_trailer_mmap(
        out.file(),
        active_ids,
        deleted_ids,
        trailer_layout,
        "DataWriter"));

    assert(ftell(out.file()) == static_cast<long>(trailer_layout.file_size));

    return out.flush_and_close("DataWriter");
}

} // namespace sketch2
