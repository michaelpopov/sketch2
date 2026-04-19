// Declares DatasetReader, DatasetRangeReader, and associated read infrastructure.

#pragma once
#include "dataset.h"
#include "core/utils/rw_lock.h"
#include "core/utils/update_notifier.h"
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace sketch2 {

class DataReader;
using DataReaderPtr = std::shared_ptr<DataReader>;

class DatasetReader;  // forward declaration for DatasetRangeReader

// DatasetRangeReader iterates through the set of data-file ranges that make up
// a dataset and fetches the reader for the range covering a specific id.
class DatasetRangeReader {
public:
    Ret init(const DatasetReader* dataset, std::shared_ptr<const std::vector<DatasetItem>> items);
    std::pair<DataReaderPtr, Ret> next();

    // Get a DataReader for the file range containing id.
    std::pair<DataReaderPtr, Ret> get(uint64_t id);

private:
    std::pair<DataReaderPtr, Ret> get_or_load_reader_(size_t index);

    const DatasetReader* dataset_ = nullptr;
    std::shared_ptr<const std::vector<DatasetItem>> items_;
    // Per-scan cache aligned with items_. Once resolved, next()/get() can reuse
    // the reader handle without touching DatasetReader's shared cache again.
    std::vector<DataReaderPtr> readers_;
    std::vector<uint8_t> reader_resolved_;
    size_t current_ = 0;
};

using DatasetRangeReaderPtr = std::unique_ptr<DatasetRangeReader>;

// DatasetReader owns the read infrastructure: file cache, update notifier
// (checker mode), and the public read API (reader(), get(), get_vector()).
class DatasetReader : public Dataset {
    friend class DatasetRangeReader;
public:
    using Dataset::Dataset;
    ~DatasetReader() override = default;

    DatasetRangeReaderPtr reader() const;
    std::pair<DataReaderPtr, Ret> get(uint64_t id) const;
    std::pair<const uint8_t*, Ret> get_vector(uint64_t id) const;
    std::pair<std::string, Ret> get_vector_string(uint64_t id, size_t digits = 2) const;

protected:
    mutable sketch::RWLock cache_lock_;
    // Guarded by cache_lock_. Replaced atomically as a shared immutable snapshot.
    mutable std::shared_ptr<const std::vector<DatasetItem>> items_cache_;
    // Guarded by cache_lock_.
    mutable std::unordered_map<uint64_t, DataReaderPtr> reader_cache_;

    void invalidate_data_caches_();

private:
    mutable std::mutex update_notifier_mutex_;
    mutable std::unique_ptr<UpdateNotifier> update_notifier_;

    Ret ensure_update_notifier_() const;
    // Returns a shared immutable snapshot of dataset items, refreshing caches
    // only when the notifier reports an update or the cache is absent.
    std::pair<std::shared_ptr<const std::vector<DatasetItem>>, Ret> get_items_snapshot_() const;
    std::pair<DataReaderPtr, Ret> open_reader_(const DatasetItem& item) const;
    std::pair<DataReaderPtr, Ret> get_cached_reader_(const DatasetItem& item) const;
};

} // namespace sketch2
