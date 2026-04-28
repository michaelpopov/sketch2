#include "bitset_filter_control.h"

#include "utils/singleton.h"

#include <fcntl.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace sketch2 {

namespace {

bool align_up_size(size_t value, size_t alignment, size_t* out) {
    const size_t mask = alignment - 1u;
    if (value > std::numeric_limits<size_t>::max() - mask) {
        return false;
    }
    *out = (value + mask) & ~mask;
    return true;
}

Ret allocate_file_size(int fd, size_t size) {
    if (size > static_cast<size_t>(std::numeric_limits<off_t>::max())) {
        return Ret("bitset filter builder: serialized blob is too large");
    }
#if defined(__linux__)
    const int rc = posix_fallocate(fd, 0, static_cast<off_t>(size));
    if (rc != 0) {
        return Ret("bitset filter builder: failed to allocate spill file");
    }
    return Ret(0);
#else
    if (ftruncate(fd, static_cast<off_t>(size)) != 0) {
        return Ret("bitset filter builder: failed to resize spill file");
    }
    return Ret(0);
#endif
}

class FileDescriptorGuard {
public:
    FileDescriptorGuard() = default;
    explicit FileDescriptorGuard(int fd) noexcept
        : fd_(fd) {}

    FileDescriptorGuard(const FileDescriptorGuard&) = delete;
    FileDescriptorGuard& operator=(const FileDescriptorGuard&) = delete;

    ~FileDescriptorGuard() {
        close_fd();
    }

    void reset(int fd) noexcept {
        close_fd();
        fd_ = fd;
    }

    void close_fd() noexcept {
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
    }

    int release_fd() noexcept {
        const int fd = fd_;
        fd_ = -1;
        return fd;
    }

    int fd() const noexcept {
        return fd_;
    }

private:
    int fd_ = -1;
};

class FilePathGuard {
public:
    FilePathGuard() = default;
    explicit FilePathGuard(std::filesystem::path path)
        : path_(std::move(path)) {}

    FilePathGuard(const FilePathGuard&) = delete;
    FilePathGuard& operator=(const FilePathGuard&) = delete;

    ~FilePathGuard() {
        if (!path_.empty()) {
            std::error_code ec;
            std::filesystem::remove(path_, ec);
        }
    }

    void release() {
        path_.clear();
    }

private:
    std::filesystem::path path_;
};

Ret create_spill_temp_file(const std::filesystem::path& template_path,
        FileDescriptorGuard* file_out, std::filesystem::path* path_out) {
    if (file_out == nullptr || path_out == nullptr) {
        return Ret("bitset filter builder: invalid spill file output");
    }

    const std::string path_string = template_path.string();
    std::vector<char> writable(path_string.begin(), path_string.end());
    writable.push_back('\0');

    const int fd = mkstemp(writable.data());
    if (fd < 0) {
        return Ret("bitset filter builder: failed to create spill file");
    }

    FileDescriptorGuard file(fd);
    std::filesystem::path path(writable.data());
    *path_out = std::move(path);
    file_out->reset(file.release_fd());
    return Ret(0);
}

Ret init_mapped_storage_from_fd(BitsetFilterControl* control, const ChunkedBits& bits,
        size_t blob_size, FileDescriptorGuard* file, BitsetFilterStorageKind kind) {
    Ret ret = allocate_file_size(file->fd(), blob_size);
    if (ret.code() != 0) {
        return ret;
    }

    control->reset();
    control->storage.kind = kind;
    control->storage.size = blob_size;
    control->storage.fd = file->release_fd();

    ret = control->storage.region.init(
        control->storage.fd, 0, blob_size, false, MappedRegionAccess::Writable,
        "bitset filter builder: mmap spill file");
    if (ret.code() != 0) {
        control->reset();
        return ret;
    }

    control->storage.data = control->storage.region.mutable_data();
    if (control->storage.data == nullptr) {
        control->reset();
        return Ret("bitset filter builder: mapped spill file is not writable");
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

    return Ret(0);
}

} // namespace

BitsetFilterStorage::~BitsetFilterStorage() {
    reset();
}

void BitsetFilterStorage::reset() {
    if (data != nullptr && kind == BitsetFilterStorageKind::Heap) {
        std::free(data);
    }
    if (kind == BitsetFilterStorageKind::MappedFile ||
            kind == BitsetFilterStorageKind::MappedFileTemporary) {
        region.reset();
        if (fd >= 0) {
            close(fd);
        }
    }
    kind = BitsetFilterStorageKind::Heap;
    data = nullptr;
    size = 0;
    fd = -1;
}

void BitsetFilterControl::reset() {
    view = ChunkedBitsView();
    storage.reset();
}

Ret BitsetFilterControl::create(ChunkedBits& bits, std::unique_ptr<BitsetFilterControl>* out) {
    if (out == nullptr) {
        return Ret("bitset filter builder: invalid control output");
    }
    out->reset();

    std::unique_ptr<BitsetFilterControl> control(new BitsetFilterControl());
    Ret ret = control->init_from_builder_(bits);
    if (ret.code() != 0) {
        return ret;
    }
    *out = std::move(control);
    return Ret(0);
}

Ret BitsetFilterControl::create_empty(std::unique_ptr<BitsetFilterControl>* out) {
    if (out == nullptr) {
        return Ret("bitset filter builder: invalid control output");
    }
    out->reset(new BitsetFilterControl());
    return Ret(0);
}

Ret BitsetFilterControl::init_from_builder_(ChunkedBits& bits) {
    Ret ret = bits.finish();
    if (ret.code() != 0) {
        return Ret("bitset filter builder: finish failed");
    }

    const size_t blob_size = bits.serialized_size_bytes();
    if (blob_size == 0) {
        return Ret("bitset filter builder: serialized blob size is unavailable");
    }

    const Singleton& singleton = get_singleton();
    if (!bits.name().empty()) {
        return init_named_mapped_from_bits_(bits, blob_size, singleton.bitset_filter_spill_dir());
    }
    if (blob_size <= singleton.bitset_filter_spill_threshold_bytes()) {
        return init_heap_from_bits_(bits, blob_size);
    }
    return init_temp_mapped_from_bits_(bits, blob_size, singleton.bitset_filter_spill_dir());
}

Ret BitsetFilterControl::init_heap_from_bits_(
        const ChunkedBits& bits, size_t blob_size) {
    size_t allocation_size = 0;
    if (!align_up_size(blob_size, kChunkedBitsBlobAlignment, &allocation_size)) {
        return Ret("bitset filter builder: serialized blob is too large");
    }

    void* blob = std::aligned_alloc(kChunkedBitsBlobAlignment, allocation_size);
    if (blob == nullptr) {
        return Ret("sketch2: out of memory");
    }

    reset();
    storage.kind = BitsetFilterStorageKind::Heap;
    storage.data = blob;
    storage.size = blob_size;

    Ret ret = bits.serialize(storage.data, blob_size);
    if (ret.code() != 0) {
        reset();
        return ret;
    }

    ret = view.init_blob(storage.data, blob_size);
    if (ret.code() != 0) {
        reset();
        return ret;
    }
    return Ret(0);
}

Ret BitsetFilterControl::init_named_mapped_from_bits_(
        const ChunkedBits& bits, size_t blob_size,
        const std::filesystem::path& spill_dir) {
    const std::filesystem::path final_path = named_bitset_filter_path(bits.name());
    FileDescriptorGuard file;
    std::filesystem::path path;
    Ret ret = create_spill_temp_file(
        spill_dir / (bits.name() + kBitsetFilterNamedTempFileTemplateSuffix), &file, &path);
    if (ret.code() != 0) {
        return ret;
    }
    FilePathGuard path_guard(path);

    ret = init_mapped_storage_from_fd(
        this, bits, blob_size, &file, BitsetFilterStorageKind::MappedFile);
    if (ret.code() != 0) {
        return ret;
    }
    ret = storage.region.sync("bitset filter builder: failed to sync named spill file");
    if (ret.code() != 0) {
        reset();
        return ret;
    }

    std::error_code ec;
    std::filesystem::rename(path, final_path, ec);
    if (ec) {
        reset();
        return Ret("bitset filter builder: failed to publish named spill file");
    }
    path_guard.release();
    return Ret(0);
}

Ret BitsetFilterControl::init_temp_mapped_from_bits_(
        const ChunkedBits& bits, size_t blob_size,
        const std::filesystem::path& spill_dir) {
    FileDescriptorGuard file;
    std::filesystem::path path;
    Ret ret = create_spill_temp_file(spill_dir / "sketch2_bitset_filter_XXXXXX", &file, &path);
    if (ret.code() != 0) {
        return ret;
    }
    FilePathGuard path_guard(path);

    std::error_code ec;
    std::filesystem::remove(path, ec);
    if (ec) {
        return Ret("bitset filter builder: failed to remove spill file directory entry");
    }
    path_guard.release();

    return init_mapped_storage_from_fd(
        this, bits, blob_size, &file, BitsetFilterStorageKind::MappedFileTemporary);
}

BitsetFilterStorageKind bitset_filter_storage_kind_for_testing(const BitsetFilterControl* control) {
    if (control == nullptr) {
        return BitsetFilterStorageKind::Heap;
    }
    return control->storage.kind;
}

std::filesystem::path named_bitset_filter_path(const std::string& name) {
    return get_singleton().bitset_filter_spill_dir() / (name + kBitsetFilterNamedFileSuffix);
}

Ret drop_named_bitset_filter(const char* name, bool* removed_out) {
    *removed_out = false;

    if (validate_chunked_bits_name(name).code() != 0) {
        return Ret("bitset_drop: invalid bitset filter name");
    }

    std::error_code ec;
    const bool removed = std::filesystem::remove(named_bitset_filter_path(name), ec);
    if (ec) {
        return Ret("bitset_drop: failed to remove named bitset filter");
    }

    *removed_out = removed;
    return Ret(0);
}

} // namespace sketch2
