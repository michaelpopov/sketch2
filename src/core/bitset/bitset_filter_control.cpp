#include "bitset_filter_control.h"

#include "bitset_file_cache.h"
#include "utils/checked_arithmetic.h"
#include "utils/file_descriptor_guard.h"
#include "utils/log.h"
#include "utils/singleton.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace sketch2 {

namespace {

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

struct AlignedFreeDeleter {
    void operator()(void* p) const noexcept { std::free(p); }
};
using HeapBlobPtr = std::unique_ptr<void, AlignedFreeDeleter>;

// Construct a heap-backed BitsetFilterStorage that takes ownership of the
// blob held by `blob_guard`. The guard is only released after make_unique
// succeeds, so a bad_alloc leaves the blob owned by (and freed by) the
// caller's guard during stack unwinding.
std::unique_ptr<BitsetFilterStorage> make_heap_storage(
        HeapBlobPtr& blob_guard, size_t blob_size) {
    auto storage = std::make_unique<BitsetFilterStorage>();
    storage->kind = BitsetFilterStorageKind::Heap;
    storage->heap_size = blob_size;
    storage->heap_blob = blob_guard.release();
    return storage;
}

// Construct a mapped-file BitsetFilterStorage that takes ownership of the fd
// held by `fd_guard`. Same exception-safety contract as make_heap_storage:
// on bad_alloc the fd remains owned by the caller's guard.
std::unique_ptr<BitsetFilterStorage> make_mapped_storage(
        FileDescriptorGuard& fd_guard, BitsetFilterStorageKind kind) {
    auto storage = std::make_unique<BitsetFilterStorage>();
    storage->kind = kind;
    storage->fd = fd_guard.release_fd();
    return storage;
}

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

// Open `path` read-only, validate it as a regular non-empty file, mmap it,
// and build a ChunkedBitsView over the blob. Errors carry the caller-supplied
// prefix so the load and post-publish call sites both produce messages keyed
// to their context (e.g. "bitset_load" vs "bitset filter builder").
Ret open_validate_and_map_named_filter(
        const std::filesystem::path& path,
        const std::string& error_prefix,
        std::unique_ptr<BitsetFilterStorage>* storage_out,
        ChunkedBitsView* view_out) {
    FileDescriptorGuard file(open(path.c_str(), O_RDONLY));
    if (file.fd() < 0) {
        return Ret(error_prefix + ": failed to open named bitset filter");
    }

    struct stat statbuf{};
    if (fstat(file.fd(), &statbuf) != 0) {
        return Ret(error_prefix + ": failed to stat named bitset filter");
    }
    if (!S_ISREG(statbuf.st_mode)) {
        return Ret(error_prefix + ": named bitset filter is not a regular file");
    }
    if (statbuf.st_size <= 0) {
        return Ret(error_prefix + ": named bitset filter is empty");
    }
    if (static_cast<uintmax_t>(statbuf.st_size) >
            static_cast<uintmax_t>(std::numeric_limits<size_t>::max())) {
        return Ret(error_prefix + ": named bitset filter is too large");
    }
    const size_t blob_size = static_cast<size_t>(statbuf.st_size);

    auto storage = make_mapped_storage(file, BitsetFilterStorageKind::MappedFile);

    Ret ret = storage->region.init(
        storage->fd, 0, blob_size, /*is_seq=*/true, MappedRegionAccess::ReadOnly,
        error_prefix + ": mmap named bitset filter");
    if (ret.code() != 0) {
        return ret;
    }
    if (storage->data() == nullptr) {
        return Ret(error_prefix + ": mapped bitset filter is unavailable");
    }

    ChunkedBitsView view;
    ret = view.init_blob(storage->data(), storage->size());
    if (ret.code() != 0) {
        return Ret(error_prefix + ": malformed bitset filter: " + ret.message());
    }

    *storage_out = std::move(storage);
    *view_out = std::move(view);
    return Ret(0);
}

// Best-effort publish into the cache under `name` by duplicating the
// borrower's already-validated fd. Avoids the TOCTOU window of reopening
// the path: a concurrent rename between validation and cache publish could
// otherwise cause the cache to hold a different (and unvalidated) file than
// the one the borrower mapped. Failures are logged with `log_prefix` and
// swallowed: the caller's mapping is already established, so a missing
// cache entry only costs a future re-open.
void try_publish_to_cache(
        const std::string& name,
        int source_fd,
        size_t size,
        const char* log_prefix) {
    FileDescriptorGuard cache_fd(::dup(source_fd));
    if (cache_fd.fd() < 0) {
        LOG_WARN << log_prefix << ": failed to dup fd for cache publish of "
                 << name;
        return;
    }
    const Ret insert_ret = bitset_file_cache().insert(
        name, std::move(cache_fd), size);
    if (insert_ret.code() != 0) {
        LOG_WARN << log_prefix << ": failed to add " << name
                 << " to cache: " << insert_ret.message();
    }
}

} // namespace

BitsetFilterStorage::~BitsetFilterStorage() {
    reset();
}

void BitsetFilterStorage::reset() {
    if (kind == BitsetFilterStorageKind::Heap) {
        if (heap_blob != nullptr) {
            std::free(heap_blob);
        }
    } else {
        region.reset();
        if (fd >= 0) {
            ::close(fd);
        }
    }
    kind = BitsetFilterStorageKind::Heap;
    heap_blob = nullptr;
    heap_size = 0;
    fd = -1;
}

const void* BitsetFilterStorage::data() const {
    return kind == BitsetFilterStorageKind::Heap
        ? heap_blob
        : static_cast<const void*>(region.data());
}

void* BitsetFilterStorage::writable_data() {
    return kind == BitsetFilterStorageKind::Heap
        ? heap_blob
        : static_cast<void*>(region.mutable_data());
}

size_t BitsetFilterStorage::size() const {
    return kind == BitsetFilterStorageKind::Heap ? heap_size : region.size();
}

void BitsetFilterControl::reset() {
    view = ChunkedBitsView();
    storage.reset();   // drops control's reference; storage survives if cache holds it.
}

void BitsetFilterControlDeleter::operator()(BitsetFilterControl* ptr) const {
    if (ptr != nullptr) {
        ptr->release();
    }
}

Ret BitsetFilterControl::create(ChunkedBits& bits, BitsetFilterControlPtr* out) {
    if (out == nullptr) {
        return Ret("bitset filter builder: invalid control output");
    }
    out->reset();

    BitsetFilterControlPtr control(new BitsetFilterControl());
    Ret ret = control->init_from_builder_(bits);
    if (ret.code() != 0) {
        return ret;
    }
    *out = std::move(control);
    return Ret(0);
}

Ret BitsetFilterControl::create_empty(BitsetFilterControlPtr* out) {
    if (out == nullptr) {
        return Ret("bitset filter builder: invalid control output");
    }
    out->reset(new BitsetFilterControl());
    return Ret(0);
}

void BitsetFilterControl::retain() {
    ref_count_.fetch_add(1, std::memory_order_relaxed);
}

void BitsetFilterControl::release() {
    if (ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        delete this;
    }
}

Ret BitsetFilterControl::load_named(
        const char* name, BitsetFilterControlPtr* out) {
    if (out == nullptr) {
        return Ret("bitset_load: invalid control output");
    }
    out->reset();

    BitsetFilterControlPtr control(new BitsetFilterControl());
    Ret ret = control->init_named_mapped_from_file_(name);
    if (ret.code() != 0) {
        return ret;
    }
    *out = std::move(control);
    return Ret(0);
}

Ret BitsetFilterControl::init_from_builder_(ChunkedBits& bits) {
    Ret ret = bits.finish();
    if (ret.code() != 0) {
        return Ret("bitset filter builder: " + ret.message());
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
    if (!align_up(blob_size, kChunkedBitsBlobAlignment, &allocation_size)) {
        return Ret("bitset filter builder: serialized blob is too large");
    }

    void* blob = std::aligned_alloc(kChunkedBitsBlobAlignment, allocation_size);
    if (blob == nullptr) {
        return Ret("sketch2: out of memory");
    }
    HeapBlobPtr blob_guard(blob);
    auto new_storage = make_heap_storage(blob_guard, blob_size);

    Ret ret = bits.serialize(new_storage->heap_blob, blob_size);
    if (ret.code() != 0) {
        return ret;
    }

    ChunkedBitsView new_view;
    ret = new_view.init_blob(new_storage->data(), new_storage->size());
    if (ret.code() != 0) {
        return ret;
    }

    reset();
    view = std::move(new_view);
    storage = std::move(new_storage);
    return Ret(0);
}

Ret BitsetFilterControl::init_mapped_storage_from_fd_(
        const ChunkedBits& bits, size_t blob_size, int* fd,
        BitsetFilterStorageKind kind) {
    if (fd == nullptr || *fd < 0) {
        return Ret("bitset filter builder: invalid spill file");
    }

    Ret ret = allocate_file_size(*fd, blob_size);
    if (ret.code() != 0) {
        return ret;
    }

    // Move fd into a local guard before any throwable allocation, so a
    // bad_alloc from make_shared (or any later throw before storage adopts
    // the fd) cannot leak it.
    FileDescriptorGuard local_fd(*fd);
    *fd = -1;

    auto new_storage = make_mapped_storage(local_fd, kind);

    ret = new_storage->region.init(
        new_storage->fd, 0, blob_size, false, MappedRegionAccess::Writable,
        "bitset filter builder: mmap spill file");
    if (ret.code() != 0) {
        return ret;
    }

    if (new_storage->writable_data() == nullptr) {
        return Ret("bitset filter builder: mapped spill file is not writable");
    }

    ret = bits.serialize(new_storage->writable_data(), blob_size);
    if (ret.code() != 0) {
        return ret;
    }

    ChunkedBitsView new_view;
    ret = new_view.init_blob(new_storage->data(), new_storage->size());
    if (ret.code() != 0) {
        return ret;
    }

    reset();
    view = std::move(new_view);
    storage = std::move(new_storage);
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

    int fd = file.release_fd();
    ret = init_mapped_storage_from_fd_(bits, blob_size, &fd, BitsetFilterStorageKind::MappedFile);
    // Reclaims fd on early failure inside init_mapped_storage_from_fd_ (where
    // *fd is left at the original value); no-op on success (*fd is set to -1
    // once storage adopts ownership).
    file.reset(fd);
    if (ret.code() != 0) {
        return ret;
    }
    ret = storage->region.sync("bitset filter builder: failed to sync named spill file");
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

    // Evict any prior cache entry now that the on-disk file has been replaced.
    // Without this, concurrent bitset_load(name) calls in the window between
    // the rename and the cache republish below would dup the stale fd (still
    // valid against the now-unlinked old inode) and return old data. Evicting
    // here also guarantees that if the reopen/republish step fails, the cache
    // does not retain a permanently stale entry — the next load will reopen
    // from disk and repopulate.
    bitset_file_cache().remove(bits.name());

    // Downgrade the in-memory mapping to read-only before caching, so cache
    // hits never expose writable access to readers.
    return swap_to_readonly_storage_for_cache_(bits.name(), final_path);
}

Ret BitsetFilterControl::swap_to_readonly_storage_for_cache_(
        const std::string& name,
        const std::filesystem::path& final_path) {
    if (storage == nullptr) {
        return Ret("bitset filter builder: missing named bitset storage");
    }

    // Some filesystems, notably Windows-mounted paths under WSL, do not make a
    // renamed file visible by its final pathname until descriptors and mappings
    // from the pre-rename file are gone. Drop the writable mapping/fd before
    // reopening the published path read-only.
    reset();

    std::unique_ptr<BitsetFilterStorage> readonly_storage;
    ChunkedBitsView readonly_view;
    Ret ret = open_validate_and_map_named_filter(
        final_path, "bitset filter builder", &readonly_storage, &readonly_view);
    if (ret.code() != 0) {
        return ret;
    }

    view = std::move(readonly_view);
    storage = std::move(readonly_storage);

    try_publish_to_cache(name, storage->fd, storage->size(), "bitset filter builder");
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

    int fd = file.release_fd();
    ret = init_mapped_storage_from_fd_(
        bits, blob_size, &fd, BitsetFilterStorageKind::MappedFileTemporary);
    // Reclaims fd on early failure inside init_mapped_storage_from_fd_ (where
    // *fd is left at the original value); no-op on success (*fd is set to -1
    // once storage adopts ownership).
    file.reset(fd);
    return ret;
}

Ret BitsetFilterControl::init_named_mapped_from_file_(const char* name) {
    if (validate_chunked_bits_name(name).code() != 0) {
        return Ret("bitset_load: invalid bitset filter name");
    }

    std::unique_ptr<BitsetFilterStorage> cached;
    Ret acquire_ret = bitset_file_cache().acquire(name, &cached);
    if (acquire_ret.code() != 0) {
        return Ret("bitset_load: " + acquire_ret.message());
    }
    if (cached != nullptr) {
        ChunkedBitsView new_view;
        Ret ret = new_view.init_blob(cached->data(), cached->size());
        if (ret.code() != 0) {
            return Ret("bitset_load: malformed bitset filter: " + ret.message());
        }
        reset();
        view = std::move(new_view);
        storage = std::move(cached);
        return Ret(0);
    }

    const std::filesystem::path file_path = named_bitset_filter_path(name);
    std::unique_ptr<BitsetFilterStorage> new_storage;
    ChunkedBitsView new_view;
    Ret ret = open_validate_and_map_named_filter(
        file_path, "bitset_load", &new_storage, &new_view);
    if (ret.code() != 0) {
        return ret;
    }

    reset();
    view = std::move(new_view);
    storage = std::move(new_storage);

    try_publish_to_cache(name, storage->fd, storage->size(), "bitset_load");
    return Ret(0);
}

BitsetFilterStorageKind bitset_filter_storage_kind_for_testing(const BitsetFilterControl* control) {
    if (control == nullptr || control->storage == nullptr) {
        return BitsetFilterStorageKind::Heap;
    }
    return control->storage->kind;
}

std::filesystem::path named_bitset_filter_path(const std::string& name) {
    return get_singleton().bitset_filter_spill_dir() / (name + kBitsetFilterNamedFileSuffix);
}

Ret load_named_bitset_filter(const char* name, BitsetFilterControlPtr* out) {
    return BitsetFilterControl::load_named(name, out);
}

Ret drop_named_bitset_filter(const char* name, bool* removed_out) {
    *removed_out = false;

    if (validate_chunked_bits_name(name).code() != 0) {
        return Ret("bitset_drop: invalid bitset filter name");
    }

    bitset_file_cache().remove(name);

    std::error_code ec;
    const bool removed = std::filesystem::remove(named_bitset_filter_path(name), ec);
    if (ec) {
        return Ret("bitset_drop: failed to remove named bitset filter");
    }

    *removed_out = removed;
    return Ret(0);
}

} // namespace sketch2
