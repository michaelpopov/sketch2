// Implements non-trivial data-file layout write helpers.

#include "core/storage/data_file_layout.h"
#include "utils/mapped_region.h"
#include <unistd.h>
#include <vector>

namespace sketch2 {

// The FILE* must be opened read/write because MAP_SHARED writable mappings
// require an O_RDWR descriptor on Linux.
Ret write_roaring_ids_trailer_mmap(FILE* f,
        const RoaringIds& ids,
        const RoaringIds& deleted_ids,
        const RoaringIdsTrailerLayout& trailer_layout,
        const std::string& context) {
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
            false,
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
        CHECK(trailer_region.sync(context + ": failed to sync ids trailer mmap"));
        trailer_region.reset();
    }

    if (fseek(f, static_cast<long>(trailer_layout.file_size), SEEK_SET) != 0) {
        return Ret(context + ": failed to seek past ids trailer");
    }
    return Ret(0);
}

Ret write_zero_padding(FILE* f, size_t size, const std::string& error_message) {
    if (size == 0) {
        return Ret(0);
    }
    std::vector<uint8_t> pad(size, 0);
    if (fwrite(pad.data(), 1, pad.size(), f) != pad.size()) {
        return Ret(error_message);
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

} // namespace sketch2
