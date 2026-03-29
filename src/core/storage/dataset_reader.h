// Declares DatasetReader, DatasetRangeReader, and associated read infrastructure.

#pragma once
#include "dataset.h"
#include "core/utils/rw_lock.h"
#include "core/utils/update_notifier.h"
#include <memory>
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
    Ret init(const DatasetReader* dataset, std::vector<DatasetItem> items);
    std::pair<DataReaderPtr, Ret> next();

    // Get a DataReader for the file range containing id.
    std::pair<DataReaderPtr, Ret> get(uint64_t id);

private:
    const DatasetReader* dataset_ = nullptr;
    std::vector<DatasetItem> items_;
    int current_ = -1;
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

protected:
    mutable sketch::RWLock cache_lock_;
    mutable bool items_cache_valid_ = false;
    mutable std::vector<DatasetItem> items_cache_;
    mutable std::unordered_map<uint64_t, DataReaderPtr> reader_cache_;

    void invalidate_data_caches_();

private:
    mutable std::unique_ptr<UpdateNotifier> update_notifier_;

    Ret ensure_update_notifier_() const;
    Ret ensure_items_cache_() const;
    const DatasetItem* find_item_(uint64_t file_id) const;
    std::pair<DataReaderPtr, Ret> open_reader_(const DatasetItem& item) const;
    std::pair<DataReaderPtr, Ret> get_cached_reader_(const DatasetItem& item) const;
};

} // namespace sketch2
