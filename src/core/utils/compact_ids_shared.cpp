// Shared helpers for compact sorted-id containers.

#include "compact_ids_shared.h"

#include <cstring>

namespace sketch2 {

Ret parse_compact_ids_mapped_prefix(
    const uint8_t* data,
    size_t size,
    CompactIdsExtEncoding expected_encoding,
    const char* context,
    size_t* bytes_consumed,
    CompactIdsMappedPrefix* prefix) {
    if (bytes_consumed != nullptr) {
        *bytes_consumed = 0;
    }
    if (data == nullptr) {
        return Ret(std::string(context) + ": data pointer is null");
    }
    if (size < sizeof(CompactIdsHeader)) {
        return Ret(std::string(context) + ": buffer too small to contain header");
    }

    CompactIdsHeader header{};
    std::memcpy(&header, data, sizeof(CompactIdsHeader));
    if (header.encoding != static_cast<uint8_t>(expected_encoding)) {
        return Ret(std::string(context) + ": unexpected encoding");
    }

    const size_t payload_size = static_cast<size_t>(header.payload_size);
    if (payload_size > size - sizeof(CompactIdsHeader)) {
        return Ret(std::string(context) + ": truncated payload");
    }

    prefix->header = header;
    prefix->payload_size = payload_size;
    prefix->consumed = sizeof(CompactIdsHeader) + payload_size;
    return Ret(0);
}

Ret validate_compact_ids_empty_header(
    const CompactIdsHeader& header,
    const char* error_message) {
    if (header.aux_data != 0 || header.payload_size != 0 || header.base != 0) {
        return Ret(error_message);
    }
    return Ret(0);
}

Ret write_compact_ids_header(
    FILE* f,
    CompactIdsExtEncoding encoding,
    uint32_t count,
    uint32_t aux_data,
    uint32_t payload_size,
    uint64_t base,
    const std::string& base_message) {
    if (f == nullptr) {
        return Ret(base_message + ": file handle is null");
    }

    CompactIdsHeader header{};
    header.encoding = static_cast<uint8_t>(encoding);
    header.count = count;
    header.aux_data = aux_data;
    header.payload_size = payload_size;
    header.base = base;

    if (fwrite(&header, sizeof(header), 1, f) != 1) {
        return Ret(base_message);
    }
    return Ret(0);
}

} // namespace sketch2
