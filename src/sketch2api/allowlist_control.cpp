#include "allowlist_control.h"

#include <fcntl.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

namespace sketch2api::detail {

namespace {

bool align_up_size(size_t value, size_t alignment, size_t* out) {
    const size_t mask = alignment - 1u;
    if (value > std::numeric_limits<size_t>::max() - mask) {
        return false;
    }
    *out = (value + mask) & ~mask;
    return true;
}

sketch2::Ret allocate_file_size(int fd, size_t size) {
    if (size > static_cast<size_t>(std::numeric_limits<off_t>::max())) {
        return sketch2::Ret("allowlist builder: serialized blob is too large");
    }
#if defined(__linux__)
    const int rc = posix_fallocate(fd, 0, static_cast<off_t>(size));
    if (rc != 0) {
        return sketch2::Ret("allowlist builder: failed to allocate spill file");
    }
    return sketch2::Ret(0);
#else
    if (ftruncate(fd, static_cast<off_t>(size)) != 0) {
        return sketch2::Ret("allowlist builder: failed to resize spill file");
    }
    return sketch2::Ret(0);
#endif
}

class TempFileGuard {
public:
    TempFileGuard(int fd, const char* path) noexcept
        : fd_(fd), path_(path) {}

    TempFileGuard(const TempFileGuard&) = delete;
    TempFileGuard& operator=(const TempFileGuard&) = delete;

    ~TempFileGuard() {
        close_fd();
        if (remove_file_) {
            unlink(path_);
        }
    }

    void close_fd() noexcept {
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
    }

    void release_file() noexcept {
        remove_file_ = false;
    }

private:
    int fd_ = -1;
    const char* path_ = nullptr;
    bool remove_file_ = true;
};

} // namespace

AllowlistStorage::~AllowlistStorage() {
    reset();
}

void AllowlistStorage::reset() {
    if (data != nullptr && kind == AllowlistStorageKind::Heap) {
        std::free(data);
    }
    if (kind == AllowlistStorageKind::MappedFile) {
        region.reset();
        if (!path.empty()) {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    }
    kind = AllowlistStorageKind::Heap;
    data = nullptr;
    size = 0;
    path.clear();
}

void AllowlistControl::reset() {
    view = sketch2::ChunkedBitsView();
    storage.reset();
}

sketch2::Ret init_heap_allowlist(
        const sketch2::ChunkedBits& bits, size_t blob_size, AllowlistControl* control) {
    if (control == nullptr) {
        return sketch2::Ret("allowlist builder: invalid control");
    }

    size_t allocation_size = 0;
    if (!align_up_size(blob_size, sketch2::kChunkedBitsBlobAlignment, &allocation_size)) {
        return sketch2::Ret("allowlist builder: serialized blob is too large");
    }

    void* blob = std::aligned_alloc(sketch2::kChunkedBitsBlobAlignment, allocation_size);
    if (blob == nullptr) {
        return sketch2::Ret("sketch2: out of memory");
    }

    control->reset();
    control->storage.kind = AllowlistStorageKind::Heap;
    control->storage.data = blob;
    control->storage.size = blob_size;

    sketch2::Ret ret = bits.serialize(control->storage.data, blob_size);
    if (ret.code() != 0) {
        control->reset();
        return ret;
    }

    ret = control->view.init_blob(control->storage.data, blob_size);
    if (ret.code() != 0) {
        control->reset();
        return ret;
    }
    return sketch2::Ret(0);
}

sketch2::Ret init_mapped_allowlist(
        const sketch2::ChunkedBits& bits, size_t blob_size,
        const std::filesystem::path& spill_dir, AllowlistControl* control) {
    if (control == nullptr) {
        return sketch2::Ret("allowlist builder: invalid control");
    }

    std::string pattern = (spill_dir / "sketch2_allowlist_XXXXXX").string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');

    const int fd = mkstemp(writable.data());
    if (fd < 0) {
        return sketch2::Ret("allowlist builder: failed to create spill file");
    }
    TempFileGuard temp_file(fd, writable.data());

    const std::filesystem::path path(writable.data());

    sketch2::Ret ret = allocate_file_size(fd, blob_size);
    if (ret.code() != 0) {
        return ret;
    }

    control->reset();
    control->storage.kind = AllowlistStorageKind::MappedFile;
    control->storage.size = blob_size;
    control->storage.path = path;
    temp_file.release_file();

    ret = control->storage.region.init(
        fd, 0, blob_size, false, sketch2::MappedRegionAccess::Writable,
        "allowlist builder: mmap spill file");
    if (ret.code() != 0) {
        control->reset();
        return ret;
    }
    temp_file.close_fd();

    control->storage.data = control->storage.region.mutable_data();
    if (control->storage.data == nullptr) {
        control->reset();
        return sketch2::Ret("allowlist builder: mapped spill file is not writable");
    }

    ret = bits.serialize(control->storage.data, blob_size);
    if (ret.code() != 0) {
        control->reset();
        return ret;
    }

    ret = control->view.init_blob(control->storage.data, blob_size);
    if (ret.code() != 0) {
        control->reset();
        return ret;
    }
    return sketch2::Ret(0);
}

AllowlistStorageKind allowlist_storage_kind_for_testing(const AllowlistControl* control) {
    if (control == nullptr) {
        return AllowlistStorageKind::Heap;
    }
    return control->storage.kind;
}

const std::filesystem::path& allowlist_temp_path_for_testing(const AllowlistControl* control) {
    static const std::filesystem::path kEmpty;
    if (control == nullptr) {
        return kEmpty;
    }
    return control->storage.path;
}

} // namespace sketch2api::detail
