#include "data_writer.h"
#include "core/storage/input_reader.h"
#include "core/storage/data_file.h"
#include <experimental/scope>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <cassert>
#include <iostream>

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

    // Collect ids
    uint64_t prev_id = 0;
    std::vector<uint64_t> ids(count);
    for (size_t i = 0; i < count; ++i) {
        ids[i] = reader.id(i);
        if (i > 0) {
            if (prev_id >= ids[i]) {
                return Ret("Invalid order of ids");
            }
        }
        prev_id = ids[i];
    }

    // Build DataFileHeader
    DataFileHeader hdr{};
    hdr.magic   = kMagic;
    hdr.kind    = static_cast<uint16_t>(FileType::Data);
    hdr.version = kVersion;
    hdr.min_id  = count > 0 ? ids.front() : 0;
    hdr.max_id  = count > 0 ? ids.back()  : 0;
    hdr.count   = static_cast<uint32_t>(count);
    hdr.type    = static_cast<uint16_t>(data_type_to_int(reader.type()));
    hdr.dim     = static_cast<uint16_t>(reader.dim());

    /**************************************************
    std::cout << "\n\n\n";
    std::cout << "min_id=" << hdr.min_id << "\n";
    std::cout << "max_id=" << hdr.max_id << "\n";
    std::cout << "count= " << hdr.count << "\n";
    std::cout << "type=  " << hdr.type << "\n";
    std::cout << "dim=   " << hdr.dim << "\n";
    std::cout << "\n\n\n";
    ***************************************************/

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
