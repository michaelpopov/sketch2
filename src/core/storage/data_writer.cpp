#include "data_writer.h"
#include "core/storage/input_reader.h"
#include "core/storage/data_file.h"
#include <experimental/scope>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace sketch2 {

static uint16_t to_file_data_type(DataType type) {
    switch (type) {
        case DataType::f32: return 0;
        case DataType::f16: return 1;
        case DataType::i32: return 2;
        default:            return 0;
    }
}

Ret DataWriter::init(const std::string& input_path, const std::string& output_path) {
    input_path_  = input_path;
    output_path_ = output_path;
    return Ret(0);
}

Ret DataWriter::exec() {
    // Create and init InputReader from input_path
    InputReader reader;
    CHECK(reader.init(input_path_));

    const size_t count = reader.count();

    // Collect ids
    std::vector<uint64_t> ids(count);
    for (size_t i = 0; i < count; ++i) {
        ids[i] = reader.id(i);
    }

    // Build DataFileHeader
    DataFileHeader hdr{};
    hdr.magic   = kMagic;
    hdr.kind    = static_cast<uint16_t>(FileType::Data);
    hdr.version = kVersion;
    hdr.min_id  = count > 0 ? ids.front() : 0;
    hdr.max_id  = count > 0 ? ids.back()  : 0;
    hdr.count   = static_cast<uint32_t>(count);
    hdr.type    = to_file_data_type(reader.type());
    hdr.dim     = static_cast<uint16_t>(reader.dim());

    // Write output file
    FILE* f = fopen(output_path_.c_str(), "wb");
    if (!f) {
        return Ret("DataWriter: failed to open output file: " + output_path_);
    }
    std::experimental::scope_exit file_guard([f]() { fclose(f); });

    // Write header
    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        return Ret("DataWriter: failed to write header");
    }

    // Write vector data
    const size_t vec_size = reader.size();
    for (size_t i = 0; i < count; ++i) {
        if (fwrite(reader.data(i), vec_size, 1, f) != 1) {
            return Ret("DataWriter: failed to write vector data at index " + std::to_string(i));
        }
    }

    // Write ids
    if (fwrite(ids.data(), sizeof(uint64_t), count, f) != count) {
        return Ret("DataWriter: failed to write ids");
    }

    return Ret(0);
}

} // namespace sketch2
