#include "data_writer.h"
#include "core/storage/input_reader.h"
#include "core/storage/data_file_layout.h"
#include "core/utils/shared_consts.h"
#include <experimental/scope>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <cassert>
#include <limits>
#include <unistd.h>

namespace sketch2 {

Ret DataWriter::init(const std::string& input_path, const std::string& output_path,
    uint64_t start, uint64_t end) {

    input_path_  = input_path;
    output_path_ = output_path;
    start_ = start;
    end_ = end;
    return Ret(0);
}

Ret DataWriter::exec() {
    if (input_path_.empty()) {
        return Ret("Input path is not set.");
    }
    if (output_path_.empty()) {
        return Ret("Output path is not set.");
    }

    // Create and init InputReader from input_path
    InputReader source;
    CHECK(source.init(input_path_));

    InputReaderView reader(source, start_, end_);
    return load(reader, output_path_);
}

Ret DataWriter::load(const InputReaderView& reader, const std::string& output_path) {
    const size_t count = reader.count();
    if (count == 0) {
        return Ret("Invalid count of vectors in reader.");
    }

    std::vector<uint64_t> ids;
    std::vector<uint64_t> deleted_ids;
    ids.reserve(count);
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
            deleted_ids.push_back(id);
        } else {
            if (id < min_id) {
                min_id = id;
            }
            if (id > max_id) {
                max_id = id;
            }
            ids.push_back(id);
        }
    }

    if (deleted_ids.size() == count) {
        min_id = max_id = 0;
    }

    // Build DataFileHeader
    DataFileHeader hdr = make_data_header(
        min_id,
        max_id,
        static_cast<uint32_t>(ids.size()),
        static_cast<uint32_t>(deleted_ids.size()),
        reader.type(),
        static_cast<uint16_t>(reader.dim()));

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
    const IdsLayout ids_layout = compute_ids_layout(hdr, ids.size());
    std::vector<uint8_t> buf(vec_size);
    for (size_t i = 0; i < count; ++i) {
        if (!reader.is_no_data(i)) {
            CHECK(reader.data(i, buf.data(), buf.size()));
            CHECK(write_vector_record(f, buf.data(), vec_size, hdr.vector_stride,
                "DataWriter: failed to write vector data at index " + std::to_string(i)));
        }
    }

    CHECK(write_zero_padding(f, ids_layout.ids_padding, "DataWriter: failed to write id alignment padding"));

    // Write ids
    CHECK(write_u64_array(f, ids, "DataWriter: failed to write ids"));

    // Write deleted ids
    CHECK(write_u64_array(f, deleted_ids, "DataWriter: failed to write deleted_ids"));

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
