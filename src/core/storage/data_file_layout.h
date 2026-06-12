// Provides helpers for computing and writing binary data-file layout sections.

#pragma once
#include "core/storage/data_file.h"
#include "core/storage/data_file_norms.h"
#include "core/bitset/roaring_ids.h"
#include "utils/shared_types.h"
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace sketch2 {

// RAII wrapper for a buffered output file. Both DataWriter and DataMerger
// produce data files with the same open/setvbuf/flush+fsync+close lifecycle;
// this class encapsulates it so callers cannot leak handles on early returns.
class OutputFile {
public:
    OutputFile() = default;
    OutputFile(const OutputFile&) = delete;
    OutputFile& operator=(const OutputFile&) = delete;
    OutputFile(OutputFile&&) = delete;
    OutputFile& operator=(OutputFile&&) = delete;
    ~OutputFile();

    Ret open(const std::string& path, const std::string& context);
    FILE* file() const { return f_; }
    Ret flush_and_close(const std::string& context);

private:
    FILE* f_ = nullptr;
    std::vector<char> file_buffer_;
};

struct DataRecordLayout {
    size_t vector_size = 0;
    size_t norm_offset = 0;
    size_t stride = 0;
};

struct DataMetadataLayout {
    size_t vectors_bytes = 0;
    size_t vectors_padding = 0;
    size_t ids_trailer_offset = 0;
    size_t ids_trailer_padding = 0;
    size_t deleted_ids_offset = 0;
    size_t deleted_ids_bytes = 0;
    size_t deleted_ids_padding = 0;
};

struct RoaringIdsTrailerLayout {
    size_t ids_offset = 0;
    size_t ids_bytes = 0;
    size_t deleted_ids_bytes = 0;
    size_t deleted_ids_offset = 0;
    size_t file_size = 0;
};

inline size_t compute_vector_size(DataType type, uint16_t dim) {
    return static_cast<size_t>(dim) * data_type_size(type);
}

inline DataRecordLayout compute_data_record_layout(DataType type, uint16_t dim, bool has_norms) {
    const size_t vector_size = compute_vector_size(type, dim);
    const size_t norm_offset = align_up<size_t>(vector_size, alignof(float));
    const size_t norm_bytes = has_norms ? sizeof(float) : 0;

    DataRecordLayout layout{};
    layout.vector_size = vector_size;
    layout.norm_offset = norm_offset;
    layout.stride = align_up<size_t>(norm_offset + norm_bytes, static_cast<size_t>(kDataAlignment));
    return layout;
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

inline size_t serialized_bytes_or_zero(const RoaringIds& ids) {
    return ids.empty() ? 0 : ids.serialized_size_bytes();
}

inline RoaringIdsTrailerLayout compute_roaring_ids_trailer_layout(
        size_t ids_offset,
        size_t ids_bytes,
        size_t deleted_ids_bytes) {
    RoaringIdsTrailerLayout layout{};
    layout.ids_offset = ids_offset;
    layout.ids_bytes = ids_bytes;
    layout.deleted_ids_bytes = deleted_ids_bytes;
    layout.deleted_ids_offset = compute_deleted_ids_offset(ids_offset, ids_bytes);
    layout.file_size = layout.deleted_ids_offset + deleted_ids_bytes;
    return layout;
}

inline RoaringIdsTrailerLayout compute_roaring_ids_trailer_layout(
        size_t ids_offset,
        const RoaringIds& ids,
        const RoaringIds& deleted_ids) {
    return compute_roaring_ids_trailer_layout(
        ids_offset,
        serialized_bytes_or_zero(ids),
        serialized_bytes_or_zero(deleted_ids));
}

inline DataFileHeader make_data_header(uint64_t min_id, uint64_t max_id,
                                       uint64_t min_range_id,
                                       uint32_t count, uint32_t deleted_count,
                                       DataType type, uint16_t dim,
                                       uint32_t norm_flags = 0u) {
    DataFileHeader hdr{};
    hdr.base.magic = kMagic;
    hdr.base.kind = static_cast<uint16_t>(FileType::Data);
    hdr.base.version = kVersion;
    hdr.min_id = min_id;
    hdr.max_id = max_id;
    hdr.min_range_id = min_range_id;
    hdr.count = count;
    hdr.deleted_count = deleted_count;
    hdr.type = static_cast<uint16_t>(data_type_to_int(type));
    hdr.dim = dim;
    hdr.data_offset = static_cast<uint64_t>(compute_data_region_offset(sizeof(DataFileHeader)));
    hdr.flags = norm_flags;
    hdr.vector_stride =
        static_cast<uint32_t>(compute_data_record_layout(type, dim, data_file_has_norms(hdr)).stride);
    return hdr;
}

inline DataMetadataLayout compute_data_metadata_layout(const DataFileHeader& hdr, size_t count) {
    DataMetadataLayout layout{};
    layout.vectors_bytes = count * static_cast<size_t>(hdr.vector_stride);
    const size_t after_vectors = static_cast<size_t>(hdr.data_offset) + layout.vectors_bytes;
    layout.ids_trailer_offset = compute_data_region_offset(after_vectors);
    layout.vectors_padding = layout.ids_trailer_offset - after_vectors;
    layout.ids_trailer_padding = 0;
    return layout;
}

inline Ret set_data_header_layout(DataFileHeader* hdr, size_t ids_bytes, size_t deleted_ids_bytes) {
    if (hdr == nullptr) {
        return Ret("set_data_header_layout: missing header");
    }

    const DataMetadataLayout layout = compute_data_metadata_layout(*hdr, hdr->count);
    const size_t deleted_ids_offset = compute_deleted_ids_offset(layout.ids_trailer_offset, ids_bytes);

    hdr->vectors_bytes = static_cast<uint64_t>(layout.vectors_bytes);
    hdr->ids_offset = static_cast<uint64_t>(layout.ids_trailer_offset);
    hdr->ids_bytes = static_cast<uint64_t>(ids_bytes);
    hdr->deleted_ids_offset = static_cast<uint64_t>(deleted_ids_offset);
    hdr->deleted_ids_bytes = static_cast<uint64_t>(deleted_ids_bytes);
    return Ret(0);
}

Ret write_roaring_ids_trailer_mmap(FILE* f,
        const RoaringIds& ids,
        const RoaringIds& deleted_ids,
        const RoaringIdsTrailerLayout& trailer_layout,
        const std::string& context,
        uint32_t* payload_crc32 = nullptr);

Ret write_zero_padding(FILE* f, size_t size, const std::string& error_message,
        uint32_t* payload_crc32 = nullptr);

Ret read_data_file_header(const std::string& path, DataFileHeader* hdr);

Ret write_header_and_data_padding(FILE* f, const DataFileHeader& hdr, const std::string& context);

Ret rewrite_header(FILE* f, const DataFileHeader& hdr, const std::string& context);

Ret write_vector_record(FILE* f, const uint8_t* data, size_t vec_size, size_t vector_stride,
        const std::string& context,
        uint32_t* payload_crc32 = nullptr);

Ret write_data_record(FILE* f,
        const uint8_t* data,
        const DataRecordLayout& layout,
        const float* norm,
        const std::string& context,
        uint32_t* payload_crc32 = nullptr);

Ret verify_data_file_payload_crc32(int fd, const DataFileHeader& hdr, size_t file_size,
        const std::string& context);

} // namespace sketch2
