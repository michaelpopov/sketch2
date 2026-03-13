// Provides helpers for computing and writing binary data-file layout sections.

#pragma once
#include "core/storage/data_file.h"
#include "utils/shared_types.h"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace sketch2 {

struct IdsLayout {
    size_t vectors_bytes = 0;
    size_t cosine_inv_norms_offset = 0;
    size_t cosine_inv_norms_bytes = 0;
    size_t ids_offset = 0;
    size_t ids_padding = 0;
};

inline bool data_file_has_cosine_inv_norms(const DataFileHeader& hdr) {
    return (hdr.flags & kDataFileHasCosineInvNorms) != 0u;
}

inline size_t compute_vector_size(DataType type, uint16_t dim) {
    return static_cast<size_t>(dim) * data_type_size(type);
}

inline uint32_t compute_vector_stride(size_t vec_size) {
    return static_cast<uint32_t>(align_up<size_t>(vec_size, static_cast<size_t>(kDataAlignment)));
}

inline DataFileHeader make_data_header(uint64_t min_id, uint64_t max_id,
                                       uint32_t count, uint32_t deleted_count,
                                       DataType type, uint16_t dim,
                                       bool has_cosine_inv_norms = false) {
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
    hdr.flags = has_cosine_inv_norms ? kDataFileHasCosineInvNorms : 0u;
    return hdr;
}

inline IdsLayout compute_ids_layout(const DataFileHeader& hdr, size_t count) {
    IdsLayout layout{};
    layout.vectors_bytes = count * static_cast<size_t>(hdr.vector_stride);
    const size_t after_vectors = static_cast<size_t>(hdr.data_offset) + layout.vectors_bytes;
    layout.cosine_inv_norms_offset = after_vectors;
    layout.cosine_inv_norms_bytes = data_file_has_cosine_inv_norms(hdr) ? count * sizeof(float) : 0;
    const size_t after_cosine = after_vectors + layout.cosine_inv_norms_bytes;
    layout.ids_offset = align_up<size_t>(after_cosine, kIdsAlignment);
    layout.ids_padding = layout.ids_offset - after_cosine;
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

inline Ret write_f32_array(FILE* f, const std::vector<float>& values, const std::string& error_message) {
    if (values.empty()) {
        return Ret(0);
    }
    if (fwrite(values.data(), sizeof(float), values.size(), f) != values.size()) {
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

inline float compute_cosine_inverse_norm(const uint8_t* data, DataType type, size_t dim) {
    double norm_sq = 0.0;
    switch (type) {
        case DataType::f32: {
            const auto* values = reinterpret_cast<const float*>(data);
            for (size_t i = 0; i < dim; ++i) {
                const double value = static_cast<double>(values[i]);
                norm_sq += value * value;
            }
            break;
        }
        case DataType::f16: {
            const auto* values = reinterpret_cast<const float16*>(data);
            for (size_t i = 0; i < dim; ++i) {
                const double value = static_cast<double>(values[i]);
                norm_sq += value * value;
            }
            break;
        }
        case DataType::i16: {
            const auto* values = reinterpret_cast<const int16_t*>(data);
            for (size_t i = 0; i < dim; ++i) {
                const double value = static_cast<double>(values[i]);
                norm_sq += value * value;
            }
            break;
        }
        default:
            throw std::runtime_error("compute_cosine_inverse_norm: unsupported data type");
    }

    if (norm_sq == 0.0) {
        return 0.0f;
    }
    return static_cast<float>(1.0 / std::sqrt(norm_sq));
}

} // namespace sketch2
