// Implements DatasetReader, DatasetRangeReader, and the shared free helpers
// collect_dataset_items() and dataset_owner_lock_path().

#include "dataset_reader.h"
#include "core/storage/data_file_layout.h"
#include "core/storage/data_reader.h"
#include "core/utils/log.h"
#include "core/utils/timer.h"
#include <algorithm>
#include <cassert>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>

namespace sketch2 {

namespace {

bool parse_dataset_file_id(const std::string& name, const std::string& ext, uint64_t* out) {
    if (name.size() <= ext.size() || name.rfind(ext) != name.size() - ext.size()) {
        return false;
    }

    const std::string id_part = name.substr(0, name.size() - ext.size());
    if (id_part.empty()) {
        return false;
    }

    for (char c : id_part) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            return false;
        }
    }

    *out = std::stoull(id_part);
    return true;
}

} // namespace

// Free function definitions declared in dataset.h.

std::string dataset_owner_lock_path(const DatasetMetadata& metadata) {
    return metadata.dirs.front() + "/" + kOwnerLockFileName;
}

// Scans every dataset directory, groups matching .data/.delta files by numeric
// file id, validates that each group has a base data file, and returns the
// resulting items sorted by id for deterministic reads and merges.
Ret collect_dataset_items(const std::string& name, const DatasetMetadata& metadata, std::vector<DatasetItem>* items) {
    if (metadata.dirs.empty()) {
        return Ret("DatasetReader::init: dirs are not set");
    }

    std::unordered_map<uint64_t, DatasetItem> items_map;

    Timer timer("collect_dataset_items");
    for (const std::string& dir : metadata.dirs) {
        std::error_code ec;
        if (!std::filesystem::exists(dir, ec) || !std::filesystem::is_directory(dir, ec)) {
            return Ret("DatasetReader::init: invalid directory: " + dir);
        }

        auto dir_iter = std::filesystem::directory_iterator(dir, ec);
        for (; dir_iter != std::filesystem::directory_iterator(); dir_iter.increment(ec)) {
            if (ec) {
                return Ret("DatasetReader::init: failed to iterate directory: " + dir);
            }

            const auto& entry = *dir_iter;
            if (!entry.is_regular_file(ec)) {
                continue;
            }

            const std::string file_name = entry.path().filename().string();
            const std::string file_path = entry.path().string();

            uint64_t file_id = 0;
            if (parse_dataset_file_id(file_name, kDataExt, &file_id)) {
                DatasetItem& item = items_map[file_id];
                item.id = file_id;
                if (!item.data_file_path.empty()) {
                    return Ret("DatasetReader::init: duplicate data file id " + std::to_string(file_id));
                }
                item.data_file_path = file_path;
                continue;
            }

            if (parse_dataset_file_id(file_name, kDeltaExt, &file_id)) {
                DatasetItem& item = items_map[file_id];
                item.id = file_id;
                if (!item.delta_file_path.empty()) {
                    return Ret("DatasetReader::init: duplicate delta file id " + std::to_string(file_id));
                }
                item.delta_file_path = file_path;
                continue;
            }
        }
    }

    std::vector<DatasetItem> sorted_items;
    sorted_items.reserve(items_map.size());
    for (const auto& [file_id, item] : items_map) {
        if (item.data_file_path.empty()) {
            return Ret("DatasetReader::init: missing data file for id " + std::to_string(file_id));
        }
        sorted_items.push_back(item);
    }

    std::sort(sorted_items.begin(), sorted_items.end(),
        [](const DatasetItem& lhs, const DatasetItem& rhs) {
            return lhs.id < rhs.id;
        });

    *items = std::move(sorted_items);

    LOG_TRACE << "collect_dataset_items: collected items cache for " << name << " in " << timer.elapsed_ms() << " ms";
    return Ret(0);
}

/***********************************************************
 *  DatasetReader
 */

Ret DatasetReader::ensure_update_notifier_() const {
    if (update_notifier_) {
        return Ret(0);
    }
    if (metadata_.dirs.empty()) {
        return Ret(0);
    }

    update_notifier_ = std::make_unique<UpdateNotifier>();
    const std::string path = dataset_owner_lock_path(metadata_);
    return update_notifier_->init_checker(path);
}

Ret DatasetReader::ensure_items_cache_() const {
    CHECK(ensure_update_notifier_());
    if (update_notifier_ && update_notifier_->check_updated()) {
        items_cache_valid_ = false;
        items_cache_.clear();
        reader_cache_.clear();
    }

    if (items_cache_valid_) {
        return Ret(0);
    }
    CHECK(collect_dataset_items(name_, metadata_, &items_cache_));
    items_cache_valid_ = true;
    return Ret(0);
}

const DatasetItem* DatasetReader::find_item_(uint64_t file_id) const {
    auto it = std::lower_bound(items_cache_.begin(), items_cache_.end(), file_id,
        [](const DatasetItem& item, uint64_t value) {
            return item.id < value;
        });
    if (it == items_cache_.end() || it->id != file_id) {
        return nullptr;
    }
    return &(*it);
}

// Lazily opens and caches the DataReader for a dataset file pair, attaching the
// delta reader when present and verifying cosine metadata for cosine datasets.
std::pair<DataReaderPtr, Ret> DatasetReader::open_reader_(const DatasetItem& item) const {
    DataReaderPtr reader = std::make_shared<DataReader>();
    Ret ret(0);

    if (item.delta_file_path.empty()) {
        ret = reader->init(item.data_file_path);
    } else {
        auto delta_reader = std::make_unique<DataReader>();
        ret = delta_reader->init(item.delta_file_path);
        if (ret.code() != 0) {
            return {nullptr, ret};
        }
        ret = reader->init(item.data_file_path, std::move(delta_reader));
    }

    if (ret.code() != 0) {
        return {nullptr, ret};
    }

    if (metadata_.dist_func == DistFunc::COS && !reader->has_cosine_inv_norms()) {
        return {nullptr, Ret("Dataset: cosine file is missing stored inverse norms: " +
            item.data_file_path)};
    }

    return {reader, Ret(0)};
}

// Thread-safe: read lock for cache hit (common path); on miss the reader is
// opened outside any lock, then inserted under write lock.  If two threads
// race on the same item, the first insertion wins and the duplicate is dropped.
// If the open fails (e.g. writer unlinked a file mid-merge), the cache is
// invalidated and the open is retried once with refreshed paths.
std::pair<DataReaderPtr, Ret> DatasetReader::get_cached_reader_(const DatasetItem& item) const {
    {
        sketch::ReadGuard rg(cache_lock_);
        const auto cache_it = reader_cache_.find(item.id);
        if (cache_it != reader_cache_.end()) {
            return {cache_it->second, Ret(0)};
        }
    }

    // Cache miss — open the reader outside any lock so concurrent cache hits
    // are not blocked by file I/O.
    auto [reader, ret] = open_reader_(item);

    // If the open failed, the file paths from the cached DatasetItem may be
    // stale (e.g. a concurrent writer merged and unlinked a delta file).
    // Invalidate the cache, re-lookup the item, and retry once.
    if (ret.code() != 0) {
        DatasetItem refreshed;
        {
            sketch::WriteGuard wg(cache_lock_);
            items_cache_valid_ = false;
            reader_cache_.erase(item.id);
            const Ret cache_ret = ensure_items_cache_();
            if (cache_ret.code() != 0) {
                return {nullptr, cache_ret};
            }
            const DatasetItem* found = find_item_(item.id);
            if (!found) {
                return {nullptr, Ret(0)};
            }
            refreshed = *found;
        }
        std::tie(reader, ret) = open_reader_(refreshed);
        if (ret.code() != 0) {
            return {nullptr, ret};
        }
    }

    sketch::WriteGuard wg(cache_lock_);
    auto [it, inserted] = reader_cache_.emplace(item.id, reader);
    return {it->second, Ret(0)};
}

void DatasetReader::invalidate_data_caches_() {
    sketch::WriteGuard wg(cache_lock_);
    items_cache_valid_ = false;
    items_cache_.clear();
    reader_cache_.clear();
}

DatasetRangeReaderPtr DatasetReader::reader() const {
    DatasetRangeReaderPtr result = std::make_unique<DatasetRangeReader>();
    std::vector<DatasetItem> items_copy;
    {
        sketch::WriteGuard wg(cache_lock_);
        const Ret cache_ret = ensure_items_cache_();
        if (cache_ret.code() != 0) {
            throw std::runtime_error(cache_ret.message());
        }
        items_copy = items_cache_;
    }
    const auto ret = result->init(this, std::move(items_copy));
    if (ret.code() != 0) {
        throw std::runtime_error(ret.message());
    }
    return result;
}

std::pair<DataReaderPtr, Ret> DatasetReader::get(uint64_t id) const {
    DatasetItem item;
    {
        sketch::WriteGuard wg(cache_lock_);
        const Ret ret = ensure_items_cache_();
        if (ret.code() != 0) {
            return {nullptr, ret};
        }
        const DatasetItem* found = find_item_(id / metadata_.range_size);
        if (!found) {
            return {nullptr, Ret(0)};
        }
        item = *found;
    }
    return get_cached_reader_(item);
}

std::pair<const uint8_t*, Ret> DatasetReader::get_vector(uint64_t id) const {
    auto [reader, ret] = get(id);
    if (ret.code() != 0) {
        return {nullptr, ret};
    }
    if (!reader) {
        return {nullptr, Ret(0)};
    }
    return {reader->get(id), Ret(0)};
}

/***********************************************************
 *  DatasetRangeReader
 */

Ret DatasetRangeReader::init(const DatasetReader* dataset, std::vector<DatasetItem> items) {
    if (!dataset) {
        return Ret("DatasetRangeReader::init: dataset is null");
    }
    dataset_ = dataset;
    items_ = std::move(items);
    current_ = -1;
    return Ret(0);
}

std::pair<DataReaderPtr, Ret> DatasetRangeReader::next() {
    ++current_;
    if (static_cast<size_t>(current_) >= items_.size()) {
        return {nullptr, Ret(0)};
    }

    return dataset_->get_cached_reader_(items_[current_]);
}

std::pair<DataReaderPtr, Ret> DatasetRangeReader::get(uint64_t id) {
    if (!dataset_) {
        return {nullptr, Ret("DatasetRangeReader::get: reader is not initialized")};
    }

    const uint64_t file_id = id / dataset_->metadata_.range_size;
    auto it = std::lower_bound(items_.begin(), items_.end(), file_id,
        [](const DatasetItem& item, uint64_t value) {
            return item.id < value;
        });

    if (it == items_.end() || it->id != file_id) {
        return {nullptr, Ret(0)};
    }

    return dataset_->get_cached_reader_(*it);
}

} // namespace sketch2
