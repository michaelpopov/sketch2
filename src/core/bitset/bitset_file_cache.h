// Process-wide cache of named bitset filter files kept open as fds.
//
// Entries are keyed by the bitset name (the same name used to derive the
// on-disk path via named_bitset_filter_path). The cache owns one read-only
// fd per entry plus the file size; it does NOT own a shared mapping.
// Each call to acquire() builds a fresh BitsetFilterStorage with its own
// MappedRegion over the cached fd, so concurrent readers never share a
// mapping (and thus never share MADV_SEQUENTIAL prefetch state).

#pragma once

#include "bitset_filter_control.h"
#include "utils/file_descriptor_guard.h"
#include "utils/shared_types.h"

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace sketch2 {

class BitsetFileCache {
public:
    BitsetFileCache() = default;
    ~BitsetFileCache() = default;

    BitsetFileCache(const BitsetFileCache&) = delete;
    BitsetFileCache& operator=(const BitsetFileCache&) = delete;
    BitsetFileCache(BitsetFileCache&&) = delete;
    BitsetFileCache& operator=(BitsetFileCache&&) = delete;

    // Insert or replace the entry under `name`, taking ownership of `fd`.
    // Returns an error for invalid fd, empty name, or a full cache (no
    // auto-eviction; callers must remove() before insert beyond the limit).
    // On error the caller's `fd` is left at -1 because ownership has already
    // been moved into the function; the descriptor is closed by the local
    // guard during stack unwinding.
    Ret insert(const std::string& name, FileDescriptorGuard fd, size_t size);

    // On hit, builds a fresh BitsetFilterStorage with its own MappedRegion
    // over the cached fd and returns it via `out` (with storage->fd == -1
    // because the cache retains the descriptor). On miss, `*out` is nullptr
    // and the return is success.
    Ret acquire(const std::string& name,
                std::unique_ptr<BitsetFilterStorage>* out);

    // Cheap presence check. Does not mmap.
    bool contains(const std::string& name) const;

    // Remove the entry for `name`. Returns true if an entry was removed. The
    // cached fd is closed after the lock is released.
    bool remove(const std::string& name);

    // Drop every entry; cached fds are closed after the lock is released.
    // Live BitsetFilterStorage objects from prior acquire() calls are
    // unaffected — each has its own independent mapping.
    void clear();

    size_t size() const;

private:
    struct CachedNamedFile {
        FileDescriptorGuard fd;
        size_t size = 0;
    };

    mutable std::mutex mutex_;
    std::unordered_map<std::string, CachedNamedFile> entries_;
};

BitsetFileCache& bitset_file_cache();

} // namespace sketch2
