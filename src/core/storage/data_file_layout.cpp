// Implements non-trivial data-file layout write helpers.

#include "core/storage/data_file_layout.h"
#include "utils/mapped_region.h"
#include "utils/shared_consts.h"
#include "utils/string_utils.h"
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace sketch2 {

namespace {

constexpr uint8_t kZeroPadding[kDataAlignment] = {};

} // namespace

OutputFile::~OutputFile() {
    if (f_ != nullptr) {
        fclose(f_);
    }
}

Ret OutputFile::open(const std::string& path, const std::string& context) {
    if (f_ != nullptr) {
        return Ret(context + ": output file is already open");
    }
    f_ = fopen(path.c_str(), "w+b");
    if (f_ == nullptr) {
        return Ret(context + ": failed to open output file: " + path);
    }
    file_buffer_.resize(kFileBufferSize);
    (void)setvbuf(f_, file_buffer_.data(), _IOFBF, file_buffer_.size());
    return Ret(0);
}

Ret OutputFile::flush_and_close(const std::string& context) {
    if (f_ == nullptr) {
        return Ret(context + ": output file is not open");
    }
    const int flush_ret = fflush(f_);
    int fsync_ret = 0;
    if (flush_ret == 0) {
        fsync_ret = fsync(fileno(f_));
    }
    const int close_ret = fclose(f_);
    f_ = nullptr;
    if (flush_ret != 0 || fsync_ret != 0 || close_ret != 0) {
        return Ret(context + ": failed to flush and close file");
    }
    return Ret(0);
}

Ret read_data_file_header(const std::string& path, DataFileHeader* hdr) {
    if (hdr == nullptr) {
        return Ret("read_data_file_header: missing header output");
    }
    *hdr = {};

    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return Ret("read_data_file_header: failed to open file " + path + ": " + std::strerror(errno));
    }

    auto fail = [&](const std::string& message) {
        (void)::close(fd);
        *hdr = {};
        return Ret(message);
    };

    struct stat st;
    if (::fstat(fd, &st) < 0) {
        return fail("read_data_file_header: failed to stat file " + path + ": " + std::strerror(errno));
    }
    if (st.st_size < static_cast<off_t>(sizeof(DataFileHeader))) {
        return fail("read_data_file_header: file too small to contain a valid header");
    }

    const ssize_t header_bytes = ::pread(fd, hdr, sizeof(*hdr), 0);
    if (header_bytes < 0) {
        return fail("read_data_file_header: failed to read file header from " + path + ": " + std::strerror(errno));
    }
    if (header_bytes != static_cast<ssize_t>(sizeof(*hdr))) {
        return fail("read_data_file_header: short read while reading file header");
    }

    if (hdr->base.magic != kMagic) {
        return fail("read_data_file_header: invalid magic number");
    }
    if (hdr->base.kind != static_cast<uint16_t>(FileType::Data)) {
        return fail("read_data_file_header: not a data file");
    }
    if (hdr->base.version != kVersion) {
        return fail("read_data_file_header: unsupported file version");
    }

    const int close_rc = ::close(fd);
    if (close_rc != 0) {
        *hdr = {};
        return Ret("read_data_file_header: failed to close file " + path + ": " + std::strerror(errno));
    }
    return Ret(0);
}

// The FILE* must be opened read/write because MAP_SHARED writable mappings
// require an O_RDWR descriptor on Linux.
Ret write_roaring_ids_trailer_mmap(FILE* f,
        const RoaringIds& ids,
        const RoaringIds& deleted_ids,
        const RoaringIdsTrailerLayout& trailer_layout,
        const std::string& context,
        uint32_t* payload_crc32) {
    if (f == nullptr) {
        return Ret(context + ": file handle is null");
    }

    const long pos = ftell(f);
    if (pos < 0) {
        return Ret(context + ": failed to get ids trailer file position");
    }
    if (static_cast<size_t>(pos) != trailer_layout.ids_offset) {
        return Ret(context + ": unexpected ids trailer file position");
    }

    if (fflush(f) != 0) {
        return Ret(context + ": failed to flush before ids trailer mmap");
    }

    const int fd = fileno(f);
    if (fd < 0) {
        return Ret(context + ": failed to get output file descriptor");
    }
    if (ftruncate(fd, static_cast<off_t>(trailer_layout.file_size)) != 0) {
        return Ret(context + ": failed to resize output file for ids trailer");
    }

    const size_t map_size = trailer_layout.file_size - trailer_layout.ids_offset;
    if (map_size > 0) {
        MappedRegion trailer_region;
        CHECK(trailer_region.init(
            fd,
            trailer_layout.ids_offset,
            map_size,
            MappedRegionAccess::Writable,
            context + ": failed to mmap ids trailer"));
        uint8_t* data = trailer_region.mutable_data();
        if (!ids.empty()) {
            const Ret ret = ids.serialize(reinterpret_cast<char*>(data));
            if (ret.code() != 0) {
                return Ret(context + ": failed to write ids: " + ret.message());
            }
        }
        if (!deleted_ids.empty()) {
            char* deleted_ids_data =
                reinterpret_cast<char*>(
                    data + (trailer_layout.deleted_ids_offset - trailer_layout.ids_offset));
            const Ret ret = deleted_ids.serialize(deleted_ids_data);
            if (ret.code() != 0) {
                return Ret(context + ": failed to write deleted_ids: " + ret.message());
            }
        }
        if (payload_crc32 != nullptr) {
            *payload_crc32 = crc32_update(*payload_crc32, data, map_size);
        }
        CHECK(trailer_region.sync(context + ": failed to sync ids trailer mmap"));
        trailer_region.reset();
    }

    if (fseek(f, static_cast<long>(trailer_layout.file_size), SEEK_SET) != 0) {
        return Ret(context + ": failed to seek past ids trailer");
    }
    return Ret(0);
}

Ret write_zero_padding(FILE* f, size_t size, const std::string& error_message,
        uint32_t* payload_crc32) {
    if (size == 0) {
        return Ret(0);
    }
    std::vector<uint8_t> pad(size, 0);
    if (fwrite(pad.data(), 1, pad.size(), f) != pad.size()) {
        return Ret(error_message);
    }
    if (payload_crc32 != nullptr) {
        *payload_crc32 = crc32_update(*payload_crc32, pad.data(), pad.size());
    }
    return Ret(0);
}

Ret rewrite_header(FILE* f, const DataFileHeader& hdr, const std::string& context) {
    if (0 != fseek(f, 0, SEEK_SET)) {
        return Ret(context + ": failed to rewind to header");
    }
    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        return Ret(context + ": failed to write header");
    }
    return Ret(0);
}

Ret write_header_and_data_padding(FILE* f, const DataFileHeader& hdr, const std::string& context) {
    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        return Ret(context + ": failed to write header");
    }

    const size_t pad_size = static_cast<size_t>(hdr.data_offset) - sizeof(DataFileHeader);
    return write_zero_padding(f, pad_size, context + ": failed to write alignment padding");
}

Ret write_vector_record(FILE* f, const uint8_t* data, size_t vec_size, size_t vector_stride,
        const std::string& context,
        uint32_t* payload_crc32) {
    if (data == nullptr) {
        return Ret(context + ": missing vector data");
    }
    if (vec_size == 0 || vector_stride < vec_size) {
        return Ret(context + ": invalid vector stride");
    }
    if (fwrite(data, vec_size, 1, f) != 1) {
        return Ret(context + ": failed to write vector data");
    }
    if (payload_crc32 != nullptr) {
        *payload_crc32 = crc32_update(*payload_crc32, data, vec_size);
    }
    if (vector_stride == vec_size) {
        return Ret(0);
    }

    const size_t padding_size = vector_stride - vec_size;
    if (padding_size > sizeof(kZeroPadding)) {
        return Ret(context + ": invalid vector padding size");
    }
    if (fwrite(kZeroPadding, 1, padding_size, f) != padding_size) {
        return Ret(context + ": failed to write vector padding");
    }
    if (payload_crc32 != nullptr) {
        *payload_crc32 = crc32_update(*payload_crc32, kZeroPadding, padding_size);
    }
    return Ret(0);
}

Ret write_data_record(FILE* f,
        const uint8_t* data,
        const DataRecordLayout& layout,
        const float* norm,
        const std::string& context,
        uint32_t* payload_crc32) {
    if (data == nullptr) {
        return Ret(context + ": missing vector data");
    }
    if (layout.vector_size == 0 || layout.stride < layout.vector_size) {
        return Ret(context + ": invalid vector stride");
    }
    if (norm != nullptr && layout.norm_offset + sizeof(float) > layout.stride) {
        return Ret(context + ": invalid inline norm layout");
    }

    if (fwrite(data, layout.vector_size, 1, f) != 1) {
        return Ret(context + ": failed to write vector data");
    }
    if (payload_crc32 != nullptr) {
        *payload_crc32 = crc32_update(*payload_crc32, data, layout.vector_size);
    }

    const size_t bytes_before_norm = norm != nullptr
        ? layout.norm_offset - layout.vector_size
        : layout.stride - layout.vector_size;
    // Valid layouts keep padding within one alignment block; retain the runtime guard so
    // release builds still fail cleanly if a caller constructs a malformed layout by hand.
    assert(bytes_before_norm <= sizeof(kZeroPadding));
    if (bytes_before_norm > sizeof(kZeroPadding)) {
        return Ret(context + ": invalid vector padding size");
    }
    if (bytes_before_norm > 0 && fwrite(kZeroPadding, 1, bytes_before_norm, f) != bytes_before_norm) {
        return Ret(context + ": failed to write vector padding");
    }
    if (payload_crc32 != nullptr && bytes_before_norm > 0) {
        *payload_crc32 = crc32_update(*payload_crc32, kZeroPadding, bytes_before_norm);
    }

    if (norm != nullptr) {
        if (fwrite(norm, sizeof(float), 1, f) != 1) {
            return Ret(context + ": failed to write inline norm");
        }
        if (payload_crc32 != nullptr) {
            *payload_crc32 = crc32_update(
                *payload_crc32, reinterpret_cast<const uint8_t*>(norm), sizeof(float));
        }
        const size_t bytes_after_norm = layout.stride - (layout.norm_offset + sizeof(float));
        assert(bytes_after_norm <= sizeof(kZeroPadding));
        if (bytes_after_norm > sizeof(kZeroPadding)) {
            return Ret(context + ": invalid inline norm padding size");
        }
        if (bytes_after_norm > 0 && fwrite(kZeroPadding, 1, bytes_after_norm, f) != bytes_after_norm) {
            return Ret(context + ": failed to write inline norm padding");
        }
        if (payload_crc32 != nullptr && bytes_after_norm > 0) {
            *payload_crc32 = crc32_update(*payload_crc32, kZeroPadding, bytes_after_norm);
        }
    }

    return Ret(0);
}

Ret verify_data_file_payload_crc32(int fd, const DataFileHeader& hdr, size_t file_size,
        const std::string& context) {
    if (fd < 0) {
        return Ret(context + ": invalid file descriptor for payload checksum");
    }
    if (!data_file_has_payload_crc32(hdr)) {
        return Ret(context + ": payload checksum is absent");
    }
    if (hdr.data_offset > file_size) {
        return Ret(context + ": payload checksum range is invalid");
    }

    uint32_t crc = 0;
    uint64_t offset = hdr.data_offset;
    std::vector<uint8_t> buf(kFileBufferSize);
    while (offset < file_size) {
        const size_t remaining = static_cast<size_t>(file_size - offset);
        const size_t chunk = std::min(buf.size(), remaining);
        const ssize_t bytes = ::pread(fd, buf.data(), chunk, static_cast<off_t>(offset));
        if (bytes < 0) {
            return Ret(context + ": failed to read payload for checksum");
        }
        if (bytes == 0) {
            return Ret(context + ": short read while verifying payload checksum");
        }
        crc = crc32_update(crc, buf.data(), static_cast<size_t>(bytes));
        offset += static_cast<uint64_t>(bytes);
    }

    if (crc != hdr.payload_crc32) {
        return Ret(context + ": payload checksum mismatch");
    }
    return Ret(0);
}

} // namespace sketch2
