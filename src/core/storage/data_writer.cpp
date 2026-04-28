// Implements conversion from text or binary input records into the binary data-file format.

#include "data_writer.h"
#include "core/compute/norm_utils.h"
#include "core/storage/input_reader.h"
#include "core/storage/data_file_layout.h"
#include "core/utils/log.h"
#include "core/bitset/roaring_ids.h"
#include "core/utils/shared_consts.h"
#include "core/utils/timer.h"
#include <algorithm>
#include <cstdint>
#include <vector>
#include <cassert>
#include <limits>
#include <utility>

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

Ret add_to_count(uint32_t* count, size_t increment, const char* error_message) {
    if (increment > std::numeric_limits<uint32_t>::max() - *count) {
        return Ret(error_message);
    }
    *count += static_cast<uint32_t>(increment);
    return Ret(0);
}

Ret build_roaring_ids_and_stats(
        const InputReaderView& reader,
        uint64_t min_range_id,
        IdStats* stats,
        RoaringIdsBuilder* active_ids,
        RoaringIdsBuilder* deleted_ids) {
    if (stats == nullptr) {
        return Ret("DataWriter: missing id stats output");
    }
    CHECK(active_ids->init(min_range_id));
    CHECK(deleted_ids->init(min_range_id));

    const size_t count = reader.count();
    if (count == 0) {
        return Ret(0);
    }
    const uint64_t* ids = reader.ids_data();

    // Verify the slice is strictly increasing. Sortedness is the load()
    // precondition; rejecting duplicates here keeps the original error.
    for (size_t i = 1; i < count; ++i) {
        if (ids[i - 1] >= ids[i]) {
            return Ret("Invalid order of ids in input data.");
        }
    }

    // Walk runs of same-category rows and hand each run's contiguous slice
    // straight to the matching bitmap. No copying.
    size_t i = 0;
    while (i < count) {
        const bool deleted = reader.is_no_data(i);
        size_t j = i + 1;
        while (j < count && reader.is_no_data(j) == deleted) {
            ++j;
        }
        const size_t len = j - i;

        const uint64_t run_min = ids[i];
        const uint64_t run_max = ids[j - 1];

        if (deleted) {
            if (run_min < min_range_id) {
                return Ret("DataWriter: deleted ids: id is below min_range_id");
            }
            if (run_max - min_range_id > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
                return Ret("DataWriter: deleted ids: id range exceeds uint32_t");
            }
            CHECK(add_to_count(
                &stats->deleted_count, len,
                "DataWriter: deleted id count exceeds uint32_t"));
            CHECK(deleted_ids->load(ids + i, len));
        } else {
            if (run_min < min_range_id) {
                return Ret("DataWriter: active ids: min id is below min_range_id");
            }
            if (run_max - min_range_id > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
                return Ret("DataWriter: data file range exceeds uint32_t");
            }
            CHECK(add_to_count(
                &stats->active_count, len,
                "DataWriter: active id count exceeds uint32_t"));
            stats->min_id = std::min(stats->min_id, run_min);
            stats->max_id = std::max(stats->max_id, run_max);
            CHECK(active_ids->load(ids + i, len));
        }
        i = j;
    }

    if (stats->active_count == 0) {
        stats->min_id = 0;
        stats->max_id = 0;
    }

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
    RoaringIdsBuilder active_ids_builder;
    RoaringIdsBuilder deleted_ids_builder;
    CHECK(build_roaring_ids_and_stats(
        reader, min_range_id, &stats, &active_ids_builder, &deleted_ids_builder));
    RoaringIds active_ids = std::move(active_ids_builder).build();
    RoaringIds deleted_ids = std::move(deleted_ids_builder).build();
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
