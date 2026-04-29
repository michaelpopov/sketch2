#include "bitset_file_cache.h"

#include "utils/mapped_region.h"
#include "utils/shared_consts.h"

#include <unistd.h>

#include <string>
#include <utility>

namespace sketch2 {

Ret BitsetFileCache::insert(
        const std::string& name, FileDescriptorGuard fd, size_t size) {
    if (fd.fd() < 0) {
        return Ret("BitsetFileCache::insert: fd must be valid");
    }
    if (name.empty()) {
        return Ret("BitsetFileCache::insert: name must not be empty");
    }
    if (size == 0) {
        return Ret("BitsetFileCache::insert: size must be greater than zero");
    }

    // Move any prior entry into a local so its destructor (which closes the
    // cached fd) runs after the lock is released.
    CachedNamedFile previous;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(name);
        if (it != entries_.end()) {
            previous = std::move(it->second);
            it->second.fd = std::move(fd);
            it->second.size = size;
        } else {
            if (entries_.size() >= kMaxLoadedBitsetCount) {
                // The cache does not auto-evict; callers must explicitly
                // remove an entry (or call clear()) before a subsequent
                // insert can succeed. The error is returned to the caller,
                // which is responsible for logging so we do not double-log.
                return Ret("BitsetFileCache::insert: cache is full (limit "
                    + std::to_string(kMaxLoadedBitsetCount)
                    + "); evict an entry before inserting another");
            }
            CachedNamedFile entry;
            entry.fd = std::move(fd);
            entry.size = size;
            entries_.emplace(name, std::move(entry));
        }
    }
    return Ret(0);
}

Ret BitsetFileCache::acquire(
        const std::string& name,
        std::unique_ptr<BitsetFilterStorage>* out) {
    if (out == nullptr) {
        return Ret("BitsetFileCache::acquire: invalid output");
    }
    out->reset();

    // Duplicate the cached fd while holding the mutex so a concurrent
    // remove()/clear() cannot close the descriptor between lookup and dup.
    // The mmap itself is performed after the mutex is released to avoid
    // serializing concurrent acquires on a syscall that may block.
    int dup_fd = -1;
    size_t blob_size = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(name);
        if (it == entries_.end()) {
            return Ret(0);
        }
        dup_fd = ::dup(it->second.fd.fd());
        blob_size = it->second.size;
    }
    if (dup_fd < 0) {
        return Ret("BitsetFileCache::acquire: failed to duplicate cached fd");
    }
    FileDescriptorGuard local_fd(dup_fd);

    auto storage = std::make_unique<BitsetFilterStorage>();
    storage->kind = BitsetFilterStorageKind::MappedFile;
    // The cache retains the canonical fd; the borrower's storage carries
    // only the mapping. The dup'd fd is closed below — once mmap succeeds
    // the kernel keeps its own reference, so the mapping survives.
    storage->fd = -1;

    Ret ret = storage->region.init(
        local_fd.fd(), 0, blob_size, /*is_seq=*/true,
        MappedRegionAccess::ReadOnly,
        "BitsetFileCache::acquire: mmap cached bitset filter");
    if (ret.code() != 0) {
        return ret;
    }

    *out = std::move(storage);
    return Ret(0);
}

bool BitsetFileCache::contains(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.count(name) > 0;
}

bool BitsetFileCache::remove(const std::string& name) {
    CachedNamedFile evicted;
    bool removed = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(name);
        if (it != entries_.end()) {
            evicted = std::move(it->second);
            entries_.erase(it);
            removed = true;
        }
    }
    return removed;
}

void BitsetFileCache::clear() {
    std::unordered_map<std::string, CachedNamedFile> evicted;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        evicted.swap(entries_);
    }
}

size_t BitsetFileCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}

BitsetFileCache& bitset_file_cache() {
    static BitsetFileCache instance;
    return instance;
}

} // namespace sketch2
