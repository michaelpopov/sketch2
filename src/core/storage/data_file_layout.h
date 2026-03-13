#pragma once
#include "core/storage/data_file.h"
#include "utils/shared_types.h"
#include <cstdio>
#include <string>
#include <vector>

namespace sketch2 {

struct IdsLayout {
    size_t vectors_bytes = 0;
    size_t ids_offset = 0;
    size_t ids_padding = 0;
};

inline size_t compute_vector_size(DataType type, uint16_t dim) {
    return static_cast<size_t>(dim) * data_type_size(type);
}

inline uint32_t compute_vector_stride(size_t vec_size) {
    return static_cast<uint32_t>(align_up<size_t>(vec_size, static_cast<size_t>(kDataAlignment)));
}

inline DataFileHeader make_data_header(uint64_t min_id, uint64_t max_id,
                                       uint32_t count, uint32_t deleted_count,
                                       DataType type, uint16_t dim) {
    DataFileHeader hdr{};
    hdr.base.magic = kMagic;
    hdr.base.kind = static_cast<uint16_t>(FileType::Data);
    hdr.base.version = kVersion;
    hdr.min_id = min_id;
    hdr.max_id = max_id;
    hdr.count = count;
    hdr.deleted_count = deleted_count;
    hdr.type = static_cast<uint16_t>(data_type_to_int(type));
    hdr.dim = dim;
    hdr.data_offset = static_cast<uint32_t>(align_up<size_t>(sizeof(DataFileHeader), kDataAlignment));
    hdr.vector_stride = compute_vector_stride(compute_vector_size(type, dim));
    hdr.reserved = 0;
    return hdr;
}

inline IdsLayout compute_ids_layout(const DataFileHeader& hdr, size_t count) {
    IdsLayout layout{};
    layout.vectors_bytes = count * static_cast<size_t>(hdr.vector_stride);
    layout.ids_offset = align_up<size_t>(static_cast<size_t>(hdr.data_offset) + layout.vectors_bytes, kIdsAlignment);
    layout.ids_padding = layout.ids_offset - (static_cast<size_t>(hdr.data_offset) + layout.vectors_bytes);
    return layout;
}

inline Ret write_zero_padding(FILE* f, size_t size, const std::string& error_message) {
    if (size == 0) {
        return Ret(0);
    }
    std::vector<uint8_t> pad(size, 0);
    if (fwrite(pad.data(), 1, pad.size(), f) != pad.size()) {
        return Ret(error_message);
    }
    return Ret(0);
}

inline Ret write_header_and_data_padding(FILE* f, const DataFileHeader& hdr, const std::string& context) {
    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        return Ret(context + ": failed to write header");
    }

    const size_t pad_size = static_cast<size_t>(hdr.data_offset) - sizeof(DataFileHeader);
    return write_zero_padding(f, pad_size, context + ": failed to write alignment padding");
}

inline Ret rewrite_header(FILE* f, const DataFileHeader& hdr, const std::string& context) {
    if (0 != fseek(f, 0, SEEK_SET)) {
        return Ret(context + ": failed to rewind to header");
    }
    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        return Ret(context + ": failed to write header");
    }
    return Ret(0);
}

inline Ret write_u64_array(FILE* f, const std::vector<uint64_t>& values, const std::string& error_message) {
    if (values.empty()) {
        return Ret(0);
    }
    if (fwrite(values.data(), sizeof(uint64_t), values.size(), f) != values.size()) {
        return Ret(error_message);
    }
    return Ret(0);
}

inline Ret write_vector_record(FILE* f, const uint8_t* data, size_t vec_size, size_t vector_stride,
        const std::string& context) {
    if (data == nullptr) {
        return Ret(context + ": missing vector data");
    }
    if (vec_size == 0 || vector_stride < vec_size) {
        return Ret(context + ": invalid vector stride");
    }
    if (fwrite(data, vec_size, 1, f) != 1) {
        return Ret(context + ": failed to write vector data");
    }
    if (vector_stride == vec_size) {
        return Ret(0);
    }

    constexpr uint8_t kZeroPadding[kDataAlignment] = {};
    const size_t padding_size = vector_stride - vec_size;
    if (padding_size > sizeof(kZeroPadding)) {
        return Ret(context + ": invalid vector padding size");
    }
    if (fwrite(kZeroPadding, 1, padding_size, f) != padding_size) {
        return Ret(context + ": failed to write vector padding");
    }
    return Ret(0);
}

} // namespace sketch2
