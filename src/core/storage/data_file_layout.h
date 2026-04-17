// Provides helpers for computing and writing binary data-file layout sections.

#pragma once
#include "core/storage/data_file.h"
#include "utils/shared_types.h"
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace sketch2 {

struct DataMetadataLayout {
    size_t vectors_bytes = 0;
    size_t vectors_padding = 0;
    size_t norms_offset = 0;
    size_t norms_bytes = 0;
    size_t ids_trailer_offset = 0;
    size_t ids_trailer_padding = 0;
    size_t deleted_ids_offset = 0;
    size_t deleted_ids_bytes = 0;
    size_t deleted_ids_padding = 0;
};

inline uint32_t data_file_norm_flags(const DataFileHeader& hdr) {
    return hdr.flags & kDataFileNormKindMask;
}

inline bool data_file_has_norms(const DataFileHeader& hdr) {
    return data_file_norm_flags(hdr) != 0u;
}

inline bool data_file_has_cosine_inv_norms(const DataFileHeader& hdr) {
    return data_file_norm_flags(hdr) == kDataFileHasCosineInvNorms;
}

inline bool data_file_has_squared_norms(const DataFileHeader& hdr) {
    return data_file_norm_flags(hdr) == kDataFileHasSquaredNorms;
}

inline uint32_t data_file_norm_flags_for_dist(DistFunc dist_func) {
    switch (dist_func) {
        case DistFunc::COS:
            return kDataFileHasCosineInvNorms;
        case DistFunc::L2:
            return kDataFileHasSquaredNorms;
        case DistFunc::DOT:
            return 0u;
        default:
            throw std::runtime_error("data_file_norm_flags_for_dist: unsupported distance function");
    }
}

inline bool data_file_has_valid_norm_flags(const DataFileHeader& hdr) {
    const uint32_t norm_flags = data_file_norm_flags(hdr);
    return norm_flags == 0u
        || norm_flags == kDataFileHasCosineInvNorms
        || norm_flags == kDataFileHasSquaredNorms;
}

inline bool data_file_matches_stored_norms_dist(const DataFileHeader& hdr, DistFunc dist_func) {
    return data_file_norm_flags(hdr) == data_file_norm_flags_for_dist(dist_func);
}

inline DistFunc dist_func_for_data_file_norm_flags(uint32_t norm_flags) {
    switch (norm_flags) {
        case 0u:
            return DistFunc::DOT;
        case kDataFileHasCosineInvNorms:
            return DistFunc::COS;
        case kDataFileHasSquaredNorms:
            return DistFunc::L2;
        default:
            throw std::runtime_error("dist_func_for_data_file_norm_flags: invalid norm flags");
    }
}

inline bool dataset_requires_stored_norms(DistFunc dist_func) {
    return dist_func == DistFunc::COS || dist_func == DistFunc::L2;
}

inline size_t compute_vector_size(DataType type, uint16_t dim) {
    return static_cast<size_t>(dim) * data_type_size(type);
}

inline uint32_t compute_vector_stride(size_t vec_size) {
    return static_cast<uint32_t>(align_up<size_t>(vec_size, static_cast<size_t>(kDataAlignment)));
}

inline size_t compute_data_region_offset(size_t offset) {
    return align_up<size_t>(offset, static_cast<size_t>(kDataRegionAlignment));
}

inline size_t compute_deleted_ids_offset(size_t ids_offset, size_t active_ids_bytes) {
    return compute_data_region_offset(ids_offset + active_ids_bytes);
}

inline size_t compute_deleted_ids_padding(size_t ids_offset, size_t active_ids_bytes) {
    const size_t deleted_ids_offset = compute_deleted_ids_offset(ids_offset, active_ids_bytes);
    return deleted_ids_offset - (ids_offset + active_ids_bytes);
}

inline DataFileHeader make_data_header(uint64_t min_id, uint64_t max_id,
                                       uint32_t count, uint32_t deleted_count,
                                       DataType type, uint16_t dim,
                                       uint32_t norm_flags = 0u) {
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
    hdr.data_offset = static_cast<uint64_t>(compute_data_region_offset(sizeof(DataFileHeader)));
    hdr.vector_stride = compute_vector_stride(compute_vector_size(type, dim));
    hdr.flags = norm_flags;
    return hdr;
}

inline DataMetadataLayout compute_data_metadata_layout(const DataFileHeader& hdr, size_t count) {
    DataMetadataLayout layout{};
    layout.vectors_bytes = count * static_cast<size_t>(hdr.vector_stride);
    const size_t after_vectors = static_cast<size_t>(hdr.data_offset) + layout.vectors_bytes;
    layout.norms_offset = compute_data_region_offset(after_vectors);
    layout.norms_bytes = data_file_has_norms(hdr) ? count * sizeof(float) : 0;
    layout.vectors_padding = data_file_has_norms(hdr)
        ? layout.norms_offset - after_vectors
        : 0;
    const size_t after_norms = data_file_has_norms(hdr)
        ? layout.norms_offset + layout.norms_bytes
        : after_vectors;
    layout.ids_trailer_offset = compute_data_region_offset(after_norms);
    layout.ids_trailer_padding = layout.ids_trailer_offset - after_norms;
    return layout;
}

inline Ret set_data_header_layout(DataFileHeader* hdr, size_t ids_bytes, size_t deleted_ids_bytes) {
    if (hdr == nullptr) {
        return Ret("set_data_header_layout: missing header");
    }

    const DataMetadataLayout layout = compute_data_metadata_layout(*hdr, hdr->count);
    const size_t deleted_ids_offset = compute_deleted_ids_offset(layout.ids_trailer_offset, ids_bytes);

    hdr->vectors_bytes = static_cast<uint64_t>(layout.vectors_bytes);
    hdr->norms_offset = static_cast<uint64_t>(layout.norms_offset);
    hdr->norms_bytes = static_cast<uint64_t>(layout.norms_bytes);
    hdr->ids_offset = static_cast<uint64_t>(layout.ids_trailer_offset);
    hdr->ids_bytes = static_cast<uint64_t>(ids_bytes);
    hdr->deleted_ids_offset = static_cast<uint64_t>(deleted_ids_offset);
    hdr->deleted_ids_bytes = static_cast<uint64_t>(deleted_ids_bytes);
    return Ret(0);
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

} // namespace sketch2
