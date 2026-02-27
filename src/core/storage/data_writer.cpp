#include "data_writer.h"
#include "core/storage/input_reader.h"
#include "core/storage/data_file.h"
#include <experimental/scope>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <cassert>
#include <iostream>
#include <limits>

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
    DataFileHeader hdr;
    hdr.magic         = kMagic;
    hdr.kind          = static_cast<uint16_t>(FileType::Data);
    hdr.version       = kVersion;
    hdr.min_id        = min_id;
    hdr.max_id        = max_id;
    hdr.count         = static_cast<uint32_t>(ids.size());
    hdr.deleted_count = static_cast<uint32_t>(deleted_ids.size());
    hdr.type          = static_cast<uint16_t>(data_type_to_int(reader.type()));
    hdr.dim           = static_cast<uint16_t>(reader.dim());
    hdr.padding       = 0;

    // Write output file
    FILE *f = fopen(output_path.c_str(), "wb");
    if (!f) {
        return Ret("DataWriter: failed to open output file: " + output_path);
    }
    std::experimental::scope_exit file_guard([f]() { fclose(f); });

    // Write header
    assert(sizeof(hdr) % 8 == 0);
    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        return Ret("DataWriter: failed to write header");
    }

    // Write vector data
    const size_t vec_size = reader.size();
    std::vector<uint8_t> buf(vec_size);
    for (size_t i = 0; i < count; ++i) {
        if (!reader.is_no_data(i)) {
            CHECK(reader.data(i, buf.data(), buf.size()));
            if (fwrite(buf.data(), vec_size, 1, f) != 1) {
                return Ret("DataWriter: failed to write vector data at index " + std::to_string(i));
            }
        }
    }

    // Write ids
    if (fwrite(ids.data(), sizeof(uint64_t), ids.size(), f) != ids.size()) {
        return Ret("DataWriter: failed to write ids");
    }

    // Write deleted ids
    if (!deleted_ids.empty()) {
        if (fwrite(deleted_ids.data(), sizeof(uint64_t), deleted_ids.size(), f) != deleted_ids.size()) {
            return Ret("DataWriter: failed to write deleted_ids");
        }
    }

    return Ret(0);
}

} // namespace sketch2
