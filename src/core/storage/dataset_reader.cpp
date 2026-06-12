// Implements DatasetReader, DatasetRangeReader, and the shared free helpers
// collect_dataset_items() and dataset_owner_lock_path().

#include "dataset_reader.h"
#include "core/storage/data_file_layout.h"
#include "core/storage/data_reader.h"
#include "core/utils/log.h"
#include "core/utils/shared_consts.h"
#include "core/utils/string_utils.h"
#include "core/utils/timer.h"
#include <algorithm>
#include <cassert>
#include <charconv>
#include <cctype>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <system_error>
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

    uint64_t parsed_id = 0;
    const char* const id_begin = id_part.data();
    const char* const id_end = id_begin + id_part.size();
    const auto result = std::from_chars(id_begin, id_end, parsed_id, 10);
    if (result.ec != std::errc{} || result.ptr != id_end) {
        return false;
    }

    *out = parsed_id;
    return true;
}

const DatasetItem* find_item_by_id(std::span<const DatasetItem> items, uint64_t file_id) {
    auto it = std::lower_bound(items.begin(), items.end(), file_id,
        [](const DatasetItem& item, uint64_t value) {
            return item.id < value;
        });
    if (it == items.end() || it->id != file_id) {
        return nullptr;
    }
    return &(*it);
}

} // namespace

// Free function definitions declared in dataset.h.

std::string dataset_owner_lock_path(const DatasetMetadata& metadata, const std::string& dataset_name) {
    if (metadata.dirs.empty()) {
        return {};
    }
    const std::string name = dataset_name.empty() ? "dataset" : dataset_name;
    return metadata.dirs.front() + "/" + name + kLockExt;
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
        // A range with no .data file is considered deleted/absent unless a
        // dangling .delta is present. In that dangling-delta case we fail so
        // callers don't silently read from an incomplete file pair.
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
    std::lock_guard<std::mutex> lock(update_notifier_mutex_);
    if (update_notifier_) {
        return Ret(0);
    }
    if (metadata_.dirs.empty()) {
        return Ret(0);
    }

    update_notifier_ = std::make_unique<UpdateNotifier>();
    const std::string path = dataset_owner_lock_path(metadata_, name_);
    return update_notifier_->init_checker(path);
}

std::pair<std::shared_ptr<const std::vector<DatasetItem>>, Ret> DatasetReader::get_items_snapshot_() const {
    const Ret notifier_ret = ensure_update_notifier_();
    if (notifier_ret.code() != 0) {
        return {nullptr, notifier_ret};
    }

    bool cache_updated = false;
    {
        std::lock_guard<std::mutex> notifier_lock(update_notifier_mutex_);
        cache_updated = update_notifier_ && update_notifier_->check_updated();
        if (cache_updated) {
            sketch::WriteGuard wg(cache_lock_);
            items_cache_.reset();
            reader_cache_.clear();
        }
    }

    if (!cache_updated) {
        sketch::ReadGuard rg(cache_lock_);
        if (items_cache_) {
            return {items_cache_, Ret(0)};
        }
    }

    sketch::WriteGuard wg(cache_lock_);
    if (!items_cache_) {
        std::vector<DatasetItem> items;
        const Ret collect_ret = collect_dataset_items(name_, metadata_, &items);
        if (collect_ret.code() != 0) {
            return {nullptr, collect_ret};
        }
        items_cache_ = std::make_shared<const std::vector<DatasetItem>>(std::move(items));
    }

    return {items_cache_, Ret(0)};
}

// Lazily opens and caches the DataReader for a dataset file pair, attaching the
// delta reader when present and verifying stored norms metadata when required.
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

    if (dataset_requires_stored_norms(metadata_.dist_func)) {
        if (!reader->has_norms()) {
            return {nullptr, Ret("Dataset: file is missing stored norms: " + item.data_file_path)};
        }
        if (!reader->has_matching_stored_norms(metadata_.dist_func)) {
            return {nullptr, Ret("Dataset: file has incompatible stored norms: " + item.data_file_path)};
        }
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
            items_cache_.reset();
            reader_cache_.erase(item.id);
        }
        auto [items_snapshot, cache_ret] = get_items_snapshot_();
        if (cache_ret.code() != 0) {
            return {nullptr, cache_ret};
        }
        const DatasetItem* found = find_item_by_id(*items_snapshot, item.id);
        if (!found) {
            return {nullptr, Ret(0)};
        }
        refreshed = *found;
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
    items_cache_.reset();
    reader_cache_.clear();
}

DatasetRangeReaderPtr DatasetReader::reader() const {
    DatasetRangeReaderPtr result = std::make_unique<DatasetRangeReader>();
    auto [items_snapshot, cache_ret] = get_items_snapshot_();
    if (cache_ret.code() != 0) {
        throw std::runtime_error(cache_ret.message());
    }
    const auto ret = result->init(this, std::move(items_snapshot));
    if (ret.code() != 0) {
        throw std::runtime_error(ret.message());
    }
    return result;
}

std::pair<DataReaderPtr, Ret> DatasetReader::get(uint64_t id) const {
    auto [items_snapshot, ret] = get_items_snapshot_();
    if (ret.code() != 0) {
        return {nullptr, ret};
    }
    const DatasetItem* found = find_item_by_id(*items_snapshot, id / metadata_.range_size);
    if (!found) {
        return {nullptr, Ret(0)};
    }
    const DatasetItem item = *found;
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

std::pair<std::string, Ret> DatasetReader::get_vector_string(uint64_t id, size_t digits) const {
    auto [vec_data, ret] = get_vector(id);
    if (ret.code() != 0 || vec_data == nullptr) {
        return { std::string{}, ret };
    }

    const uint16_t dim = static_cast<uint16_t>(metadata_.dim);
    const DataType type = metadata_.type;

    size_t buf_size = std::max<size_t>(64, static_cast<size_t>(dim) * 32);
    std::vector<char> buf(buf_size);
    Ret ret_print = print_vector(const_cast<uint8_t*>(vec_data), type, dim, buf.data(), buf.size(), digits);
    if (ret_print.code() != 0) {
        return { std::string{}, ret_print };
    }

    return { std::string(buf.data()), Ret(0) };
}

/***********************************************************
 *  DatasetRangeReader
 */

Ret DatasetRangeReader::init(const DatasetReader* dataset,
        std::shared_ptr<const std::vector<DatasetItem>> items) {
    if (!dataset) {
        return Ret("DatasetRangeReader::init: dataset is null");
    }
    if (!items) {
        return Ret("DatasetRangeReader::init: items are not initialized");
    }
    dataset_ = dataset;
    items_ = std::move(items);
    readers_.assign(items_->size(), nullptr);
    reader_resolved_.assign(items_->size(), 0);
    current_ = 0;
    return Ret(0);
}

std::pair<DataReaderPtr, Ret> DatasetRangeReader::get_or_load_reader_(size_t index) {
    assert(index < readers_.size());

    if (reader_resolved_[index]) {
        return {readers_[index], Ret(0)};
    }

    auto [reader, ret] = dataset_->get_cached_reader_((*items_)[index]);
    if (ret.code() != 0) {
        return {nullptr, ret};
    }

    readers_[index] = reader;
    // Cache both successful opens and "resolved to missing" so repeated get()
    // calls on a stale snapshot do not re-enter the shared reader cache.
    reader_resolved_[index] = 1;
    return {reader, Ret(0)};
}

std::pair<DataReaderPtr, Ret> DatasetRangeReader::next() {
    if (!dataset_ || !items_) {
        return {nullptr, Ret("DatasetRangeReader::next: reader is not initialized")};
    }

    const auto& items = *items_;
    while (true) {
        if (current_ >= items.size()) {
            return {nullptr, Ret(0)};
        }

        auto [reader, ret] = get_or_load_reader_(current_++);
        if (ret.code() != 0) {
            return {nullptr, ret};
        }
        if (reader) {
            return {reader, Ret(0)};
        }
    }
}

std::pair<DataReaderPtr, Ret> DatasetRangeReader::get(uint64_t id) {
    if (!dataset_ || !items_) {
        return {nullptr, Ret("DatasetRangeReader::get: reader is not initialized")};
    }

    const uint64_t file_id = id / dataset_->metadata_.range_size;
    const DatasetItem* found = find_item_by_id(*items_, file_id);
    if (!found) {
        return {nullptr, Ret(0)};
    }

    const size_t index = static_cast<size_t>(found - items_->data());
    return get_or_load_reader_(index);
}

} // namespace sketch2
