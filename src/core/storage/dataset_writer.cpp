// Implements DatasetWriter: write operations, accumulator management, and WAL.

#include "dataset_writer.h"
#include "core/storage/data_file_layout.h"
#include "core/storage/data_reader.h"
#include "core/storage/data_merger.h"
#include "core/storage/data_writer.h"
#include "core/storage/input_reader.h"
#include "core/utils/file_lock.h"
#include "core/utils/log.h"
#include "core/utils/singleton.h"
#include "core/utils/thread_pool.h"
#include "core/utils/timer.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fcntl.h>
#include <filesystem>
#include <experimental/scope>
#include <future>
#include <limits>
#include <unistd.h>

namespace sketch2 {

namespace {

std::string dataset_accumulator_wal_path(const DatasetMetadata& metadata) {
    return metadata.dirs.front() + "/" + kAccumulatorWalFileName;
}

struct StoreRangeTask {
    uint64_t file_id;
    uint64_t range_start;
    uint64_t range_end;
};

} // namespace

/***********************************************************
 *  DatasetWriter lifecycle
 */

DatasetWriter::~DatasetWriter() {
    if (accumulator_ && owner_lock_ &&
            (accumulator_->vectors_count() != 0 || accumulator_->deleted_count() != 0)) {
        try {
            const Ret ret = store_accumulator();
            if (ret.code() != 0) {
                LOG_ERROR << "DatasetWriter destructor failed to flush accumulator: " << ret.message();
            }
        } catch (const std::exception& ex) {
            LOG_ERROR << "DatasetWriter destructor failed to flush accumulator: " << ex.what();
        } catch (...) {
            LOG_ERROR << "DatasetWriter destructor failed to flush accumulator: unknown error";
        }
    }

    if (owner_path_registered_ && !metadata_.dirs.empty()) {
        const std::string lock_path = dataset_owner_lock_path(metadata_);
        const bool ok = Singleton::instance().release_file_path(lock_path);
        if (!ok) {
            LOG_ERROR << "DatasetWriter destructor failed to release locked file path";
        }
        owner_path_registered_ = false;
    }
}

Ret DatasetWriter::init(const DatasetMetadata& metadata) {
    Ret ret = Dataset::init(metadata);
    if (ret.code() != 0) return ret;
    return init_writer_();
}

Ret DatasetWriter::init(const std::vector<std::string>& dirs, uint64_t range_size,
        DataType type, uint64_t dim, uint64_t accumulator_size, DistFunc dist_func) {
    Ret ret = Dataset::init(dirs, range_size, type, dim, accumulator_size, dist_func);
    if (ret.code() != 0) return ret;
    return init_writer_();
}

Ret DatasetWriter::init(const std::string& path) {
    Ret ret = Dataset::init(path);
    if (ret.code() != 0) return ret;
    return init_writer_();
}

Ret DatasetWriter::init_writer_() {
    // Replay WAL only if no other process currently owns this dataset.
    // Use a temporary lock that is released immediately after replay so that
    // ownership is still acquired lazily when first write happens.
    const std::string lock_path = dataset_owner_lock_path(metadata_);
    {
        FileLockGuard temp_lock;
        if (!temp_lock.try_lock(lock_path)) {
            return Ret(0);
        }
        CHECK(init_accumulator_());
    }
    return Ret(0);
}

/***********************************************************
 *  Public write operations
 */

Ret DatasetWriter::store(const std::string& input_path) {
    Ret ret{0};
    bool should_notify = false;
    try {
        std::lock_guard<std::mutex> lg(write_mutex_);
        CHECK(ensure_owner_lock_());
        should_notify = true;
        Timer timer("DatasetWriter::store");
        ret = store_(input_path);
        LOG_INFO << "Completed DatasetWriter::store for " << name() << " in " << timer.elapsed_ms() << " ms";
    } catch (const std::exception& ex) {
        ret = Ret(ex.what());
    }

    if (should_notify) {
        notify_update_("DatasetWriter::store");
    }

    return ret;
}

Ret DatasetWriter::store_accumulator() {
    Ret ret{0};
    bool should_notify = false;
    try {
        std::lock_guard<std::mutex> lg(write_mutex_);
        CHECK(ensure_owner_lock_());
        should_notify = true;
        Timer timer("DatasetWriter::store_accumulator");
        ret = store_accumulator_();
        LOG_INFO << "Completed DatasetWriter::store_accumulator for " << name() << " in " << timer.elapsed_ms() << " ms";
    } catch (const std::exception& ex) {
        ret = Ret(ex.what());
    }

    if (should_notify) {
        notify_update_("DatasetWriter::store_accumulator");
    }

    return ret;
}

Ret DatasetWriter::merge() {
    Ret ret{0};
    bool should_notify = false;
    try {
        std::lock_guard<std::mutex> lg(write_mutex_);
        CHECK(ensure_owner_lock_());
        should_notify = true;
        Timer timer("DatasetWriter::merge");
        ret = merge_();
        LOG_INFO << "Completed DatasetWriter::merge for " << name() << " in " << timer.elapsed_ms() << " ms";
    } catch (const std::exception& ex) {
        ret = Ret(ex.what());
    }

    if (should_notify) {
        notify_update_("DatasetWriter::merge");
    }

    return ret;
}

Ret DatasetWriter::add_vector(uint64_t id, const uint8_t* data) {
    std::lock_guard<std::mutex> lg(write_mutex_);
    CHECK(ensure_owner_lock_());
    if (!data) {
        return Ret("Dataset: invalid data argument");
    }
    CHECK(init_accumulator_());
    if (!accumulator_->can_add_vector(id)) {
        CHECK(store_accumulator_());
    }
    return accumulator_->add_vector(id, data);
}

Ret DatasetWriter::delete_vector(uint64_t id) {
    std::lock_guard<std::mutex> lg(write_mutex_);
    CHECK(ensure_owner_lock_());
    CHECK(init_accumulator_());
    if (!accumulator_->can_delete_vector(id)) {
        CHECK(store_accumulator_());
    }
    return accumulator_->delete_vector(id);
}

/***********************************************************
 *  Private helpers
 */

Ret DatasetWriter::ensure_owner_lock_() {
    if (owner_lock_ || metadata_.dirs.empty()) {
        return Ret(0);
    }

    owner_lock_ = std::make_unique<FileLockGuard>();
    const std::string lock_path = dataset_owner_lock_path(metadata_);
    CHECK(owner_lock_->lock(lock_path));
    if (!Singleton::instance().check_file_path(lock_path)) {
        owner_lock_.reset();
        return Ret("DatasetWriter: file already in use");
    }
    owner_path_registered_ = true;

    // Ensure the lock file contains a valid 8-byte counter so that
    // checker-mode DatasetReader::update_notifier_ can read it without hitting
    // the conservative "short read → treat as updated" path on every access.
    {
        const int fd = open(lock_path.c_str(), O_RDWR);
        if (fd >= 0) {
            uint64_t value = 0;
            if (pread(fd, &value, sizeof(value), 0) < static_cast<ssize_t>(sizeof(value))) {
                value = 0;
                const ssize_t ret = pwrite(fd, &value, sizeof(value), 0);
                if (ret < 0) {
                    (void)Singleton::instance().release_file_path(lock_path);
                    owner_path_registered_ = false;
                    owner_lock_.reset();
                    return Ret("Dataset: failed to initialize owner lock counter in " + lock_path +
                        ": " + std::string(std::strerror(errno)));
                }
                if (ret != static_cast<ssize_t>(sizeof(value))) {
                    (void)Singleton::instance().release_file_path(lock_path);
                    owner_path_registered_ = false;
                    owner_lock_.reset();
                    return Ret("Dataset: short write while initializing owner lock counter in " + lock_path);
                }
                (void)fdatasync(fd);
            }
            (void)close(fd);
        }
    }

    return Ret(0);
}

Ret DatasetWriter::init_accumulator_() {
    if (accumulator_) {
        return Ret(0);
    }
    if (metadata_.dirs.empty() || metadata_.range_size == 0) {
        return Ret("Dataset: not initialized.");
    }

    accumulator_ = std::make_unique<Accumulator>();
    CHECK(accumulator_->init(
        metadata_.accumulator_size,
        metadata_.type,
        metadata_.dim,
        metadata_.dist_func == DistFunc::COS));
    assert(accumulator_->has_cosine_inv_norms() == (metadata_.dist_func == DistFunc::COS));
    return accumulator_->attach_wal(dataset_accumulator_wal_path(metadata_));
}

Ret DatasetWriter::ensure_update_notifier_() const {
    if (update_notifier_) {
        return Ret(0);
    }
    if (metadata_.dirs.empty()) {
        return Ret(0);
    }
    update_notifier_ = std::make_unique<UpdateNotifier>();
    return update_notifier_->init_updater(dataset_owner_lock_path(metadata_));
}

void DatasetWriter::notify_update_(const char* caller) {
    const Ret nr = ensure_update_notifier_();
    if (nr.code() != 0) {
        LOG_ERROR << caller << ": " << nr.message();
        return;
    }
    if (!update_notifier_) {
        return;
    }
    const Ret ur = update_notifier_->update();
    if (ur.code() != 0) {
        LOG_ERROR << caller << ": " << ur.message();
    }
}

/***********************************************************
 *  Private write implementations
 */

// Splits a textual input file into dataset file ranges and delegates each
// touched range to store_and_merge(), which decides whether the new data becomes
// a base file, a delta file, or a trigger for a larger merge.
Ret DatasetWriter::store_(const std::string& input_path) {
    if (metadata_.dirs.empty() || metadata_.range_size == 0) {
        return Ret("Dataset: not initialized.");
    }

    InputReader reader;
    CHECK(reader.init(input_path));

    if (metadata_.dim != reader.dim()) {
        return Ret("Dataset: mismatched dim");
    }

    if (metadata_.type != reader.type()) {
        return Ret("Dataset: mismatched type");
    }

    if (reader.count() == 0) {
        return Ret(0);
    }

    const uint64_t min_id = reader.id(0);
    const uint64_t max_id = reader.id(reader.count() - 1);

    const uint64_t first_file = min_id / metadata_.range_size;
    const uint64_t last_file  = max_id / metadata_.range_size;
    std::vector<StoreRangeTask> tasks;
    tasks.reserve(static_cast<size_t>(last_file - first_file + 1));

    for (uint64_t file_id = first_file; file_id <= last_file; ++file_id) {
        const uint64_t range_start = file_id * metadata_.range_size;
        const uint64_t range_end   = range_start + metadata_.range_size;

        if (reader.is_range_present(range_start, range_end)) {
            tasks.push_back({file_id, range_start, range_end});
        }
    }

    const auto& thread_pool = get_singleton().thread_pool();
    if (!thread_pool || tasks.size() <= 1) {
        for (const StoreRangeTask& task : tasks) {
            CHECK(store_and_merge(reader, task.file_id, task.range_start, task.range_end));
        }
        return Ret(0);
    }

    std::vector<std::future<Ret>> futures;
    futures.reserve(tasks.size());
    for (const StoreRangeTask& task : tasks) {
        // This parallel path relies on InputReader staying immutable after
        // init(), and on store_and_merge() touching only per-range files
        // instead of shared dataset caches.
        futures.push_back(thread_pool->submit([this, &reader, task]() -> Ret {
            return store_and_merge(reader, task.file_id, task.range_start, task.range_end);
        }));
    }

    Ret first_error(0);
    for (size_t i = 0; i < futures.size(); ++i) {
        const Ret ret = futures[i].get();
        if (ret.code() != 0) {
            const StoreRangeTask& task = tasks[i];
            LOG_ERROR << "DatasetWriter::store_: range task failed for file_id=" << task.file_id
                      << " range=[" << task.range_start << ", " << task.range_end
                      << "): " << ret.message();
        }
        if (first_error.code() == 0 && ret.code() != 0) {
            first_error = ret;
        }
    }

    if (first_error.code() != 0) {
        return first_error;
    }

    return Ret(0);
}

// Flushes buffered accumulator updates into the on-disk file set.
Ret DatasetWriter::store_accumulator_() {
    if (metadata_.dirs.empty() || metadata_.range_size == 0) {
        return Ret("Dataset: not initialized.");
    }
    CHECK(init_accumulator_());

    const std::vector<uint64_t> vector_ids = accumulator_->get_vector_ids();
    const std::vector<uint64_t> deleted_ids = accumulator_->get_deleted_ids();
    if (vector_ids.empty() && deleted_ids.empty()) {
        accumulator_->clear();
        return Ret(0);
    }

    std::vector<uint64_t> affected_file_ids;
    affected_file_ids.reserve(vector_ids.size() + deleted_ids.size());
    for (uint64_t id : vector_ids) {
        affected_file_ids.push_back(id / metadata_.range_size);
    }
    for (uint64_t id : deleted_ids) {
        affected_file_ids.push_back(id / metadata_.range_size);
    }
    std::sort(affected_file_ids.begin(), affected_file_ids.end());
    affected_file_ids.erase(std::unique(affected_file_ids.begin(), affected_file_ids.end()), affected_file_ids.end());

    const auto build_range_slice = [&](uint64_t file_id,
            std::vector<uint64_t>& out_ids, std::vector<uint64_t>& out_deleted) {
        const uint64_t range_start = file_id * metadata_.range_size;
        const uint64_t range_end = range_start + metadata_.range_size;
        out_ids.assign(
            std::lower_bound(vector_ids.begin(), vector_ids.end(), range_start),
            std::lower_bound(vector_ids.begin(), vector_ids.end(), range_end));
        out_deleted.assign(
            std::lower_bound(deleted_ids.begin(), deleted_ids.end(), range_start),
            std::lower_bound(deleted_ids.begin(), deleted_ids.end(), range_end));
    };

    const auto& thread_pool = get_singleton().thread_pool();
    if (!thread_pool || affected_file_ids.size() <= 1) {
        for (uint64_t file_id : affected_file_ids) {
            std::vector<uint64_t> range_ids, range_deleted_ids;
            build_range_slice(file_id, range_ids, range_deleted_ids);
            CHECK(store_and_merge_accumulator(file_id, range_ids, range_deleted_ids));
        }
    } else {
        std::vector<std::future<Ret>> futures;
        futures.reserve(affected_file_ids.size());
        for (uint64_t file_id : affected_file_ids) {
            futures.push_back(thread_pool->submit([this, &vector_ids, &deleted_ids, file_id]() -> Ret {
                std::vector<uint64_t> range_ids, range_deleted_ids;
                const uint64_t range_start = file_id * metadata_.range_size;
                const uint64_t range_end = range_start + metadata_.range_size;
                range_ids.assign(
                    std::lower_bound(vector_ids.begin(), vector_ids.end(), range_start),
                    std::lower_bound(vector_ids.begin(), vector_ids.end(), range_end));
                range_deleted_ids.assign(
                    std::lower_bound(deleted_ids.begin(), deleted_ids.end(), range_start),
                    std::lower_bound(deleted_ids.begin(), deleted_ids.end(), range_end));
                return store_and_merge_accumulator(file_id, range_ids, range_deleted_ids);
            }));
        }

        Ret first_error(0);
        for (size_t i = 0; i < futures.size(); ++i) {
            const Ret ret = futures[i].get();
            if (ret.code() != 0) {
                LOG_ERROR << "DatasetWriter::store_accumulator_: range task failed for file_id="
                          << affected_file_ids[i] << ": " << ret.message();
                if (first_error.code() == 0) {
                    first_error = ret;
                }
            }
        }
        if (first_error.code() != 0) {
            return first_error;
        }
    }

    CHECK(accumulator_->reset_wal());
    accumulator_->clear();
    return Ret(0);
}

// Forces every existing delta file to be folded into its corresponding data file.
Ret DatasetWriter::merge_() {
    if (metadata_.dirs.empty() || metadata_.range_size == 0) {
        return Ret("Dataset: not initialized.");
    }

    std::vector<DatasetItem> all_items;
    CHECK(collect_dataset_items(name_, metadata_, &all_items));

    std::vector<DatasetItem> to_merge;
    for (DatasetItem& item : all_items) {
        if (!item.delta_file_path.empty()) {
            to_merge.push_back(std::move(item));
        }
    }

    const auto& thread_pool = get_singleton().thread_pool();
    if (!thread_pool || to_merge.size() <= 1) {
        for (const DatasetItem& item : to_merge) {
            const std::string output_path_base = item_path_base(item.id);
            DataReader data_reader;
            CHECK(data_reader.init(item.data_file_path));
            DataReader delta_reader;
            CHECK(delta_reader.init(item.delta_file_path));
            CHECK(merge_data_file(data_reader, delta_reader, output_path_base, kDeltaExt));
        }
        return Ret(0);
    }

    std::vector<std::future<Ret>> futures;
    futures.reserve(to_merge.size());
    for (const DatasetItem& item : to_merge) {
        futures.push_back(thread_pool->submit([this, item]() -> Ret {
            const std::string output_path_base = item_path_base(item.id);
            DataReader data_reader;
            CHECK(data_reader.init(item.data_file_path));
            DataReader delta_reader;
            CHECK(delta_reader.init(item.delta_file_path));
            return merge_data_file(data_reader, delta_reader, output_path_base, kDeltaExt);
        }));
    }

    Ret first_error(0);
    for (size_t i = 0; i < futures.size(); ++i) {
        const Ret ret = futures[i].get();
        if (ret.code() != 0) {
            LOG_ERROR << "DatasetWriter::merge_: task failed for item id=" << to_merge[i].id
                      << ": " << ret.message();
            if (first_error.code() == 0) {
                first_error = ret;
            }
        }
    }
    return first_error;
}

Ret DatasetWriter::store_and_merge(const InputReader& reader, uint64_t file_id,
        uint64_t range_start, uint64_t range_end) const {
    const std::string output_path_base = item_path_base(file_id);
    const std::string output_path = output_path_base + kTempExt;

    std::experimental::scope_exit file_guard([output_path]() {
        std::error_code ec;
        std::filesystem::remove(output_path, ec);
    });

    {
        InputReaderView view(reader, range_start, range_end);
        DataWriter writer;
        CHECK(writer.load(view, output_path, metadata_.dist_func == DistFunc::COS));
    }

    DataReader output_reader;
    CHECK(output_reader.init(output_path));

    const std::string data_path = output_path_base + kDataExt;
    if (!std::filesystem::exists(data_path)) {
        if (output_reader.deleted_count() != 0) {
            return Ret("DatasetWriter::store_and_merge: invalid deleted items");
        }
        std::filesystem::rename(output_path, data_path);
        return Ret(0);
    }

    const std::string delta_path = output_path_base + kDeltaExt;
    if (!std::filesystem::exists(delta_path)) {
        {
            DataReader data_reader;
            CHECK(data_reader.init(data_path));

            const bool is_merge = check_data_file_merge(data_reader, output_reader);
            if (is_merge) {
                CHECK(merge_data_file(data_reader, output_reader, output_path_base, kTempExt));
                return Ret(0);
            }
        }

        std::filesystem::rename(output_path, delta_path);
        return Ret(0);
    }

    {
        DataReader delta_reader;
        CHECK(delta_reader.init(delta_path));
        CHECK(merge_delta_file(delta_reader, output_reader, output_path_base));
    }

    {
        DataReader data_reader;
        CHECK(data_reader.init(data_path));
        DataReader delta_reader;
        CHECK(delta_reader.init(delta_path));

        const bool is_data_delta_merge = check_data_delta_merge(data_reader, delta_reader);
        if (is_data_delta_merge) {
            CHECK(merge_data_file(data_reader, delta_reader, output_path_base, kDeltaExt));
        }
    }

    return Ret(0);
}

Ret DatasetWriter::store_and_merge_accumulator(uint64_t file_id,
        const std::vector<uint64_t>& ids, const std::vector<uint64_t>& deleted_ids) {
    const std::string output_path_base = item_path_base(file_id);
    FileLockGuard file_lock;
    CHECK(file_lock.lock(output_path_base + kLockExt));
    if (ids.empty() && deleted_ids.empty()) {
        return Ret(0);
    }

    const auto write_from_accumulator_staged = [this, &ids, &deleted_ids, &output_path_base](const std::string& final_path) -> Ret {
        const std::string staging_path = output_path_base + kTempExt;
        std::experimental::scope_exit staging_guard([staging_path]() {
            std::error_code ec;
            std::filesystem::remove(staging_path, ec);
        });

        CHECK(write_accumulator_range_(staging_path, ids, deleted_ids));
        std::filesystem::rename(staging_path, final_path);
        return Ret(0);
    };

    const std::string data_path = output_path_base + kDataExt;
    if (!std::filesystem::exists(data_path)) {
        if (ids.empty()) {
            return Ret(0);
        }
        CHECK(write_from_accumulator_staged(data_path));
        return Ret(0);
    }

    const std::string delta_path = output_path_base + kDeltaExt;
    if (!std::filesystem::exists(delta_path)) {
        DataReader data_reader;
        CHECK(data_reader.init(data_path));

        const uint64_t output_count = static_cast<uint64_t>(ids.size() + deleted_ids.size());
        const bool is_merge = (data_reader.count() < output_count * metadata_.data_merge_ratio);
        if (is_merge) {
            DataMerger processor;
            const std::string merge_path = output_path_base + kMergeExt;
            CHECK(processor.merge_data_file(data_reader, *accumulator_, ids, deleted_ids, merge_path));
            std::filesystem::rename(merge_path, data_path);
            return Ret(0);
        }

        CHECK(write_from_accumulator_staged(delta_path));
        return Ret(0);
    }

    {
        DataReader delta_reader;
        CHECK(delta_reader.init(delta_path));
        DataMerger processor;
        const std::string merge_path = output_path_base + kMergeExt;
        CHECK(processor.merge_delta_file(delta_reader, *accumulator_, ids, deleted_ids, merge_path));
        std::filesystem::rename(merge_path, delta_path);
    }

    {
        DataReader data_reader;
        CHECK(data_reader.init(data_path));
        DataReader delta_reader;
        CHECK(delta_reader.init(delta_path));

        const bool is_data_delta_merge = check_data_delta_merge(data_reader, delta_reader);
        if (is_data_delta_merge) {
            CHECK(merge_data_file(data_reader, delta_reader, output_path_base, kDeltaExt));
        }
    }

    return Ret(0);
}

bool DatasetWriter::check_data_file_merge(const DataReader& data_reader,
        const DataReader& output_reader) const {
    const uint64_t output_count = output_reader.count() + output_reader.deleted_count();
    return (data_reader.count() < output_count * metadata_.data_merge_ratio);
}

bool DatasetWriter::check_data_delta_merge(const DataReader& data_reader,
        const DataReader& delta_reader) const {
    const uint64_t delta_count = delta_reader.count() + delta_reader.deleted_count();
    return (data_reader.count() < delta_count * metadata_.data_merge_ratio);
}

Ret DatasetWriter::merge_data_file(const DataReader& data_reader, const DataReader& output_reader,
        const std::string& output_path_base, const std::string& ext) const {
    const std::string source_path = output_path_base + ext;
    std::experimental::scope_exit file_guard([source_path]() {
        std::error_code ec;
        std::filesystem::remove(source_path, ec);
    });

    DataMerger processor;
    const std::string merge_path = output_path_base + kMergeExt;
    CHECK(processor.merge_data_file(data_reader, output_reader, merge_path));

    const std::string data_path = output_path_base + kDataExt;
    std::filesystem::rename(merge_path, data_path);

    return Ret(0);
}

Ret DatasetWriter::merge_delta_file(const DataReader& delta_reader, const DataReader& output_reader,
        const std::string& output_path_base) const {
    const std::string source_path = output_path_base + kTempExt;
    std::experimental::scope_exit file_guard([source_path]() {
        std::error_code ec;
        std::filesystem::remove(source_path, ec);
    });

    DataMerger processor;
    const std::string merge_path = output_path_base + kMergeExt;
    CHECK(processor.merge_delta_file(delta_reader, output_reader, merge_path));

    const std::string delta_path = output_path_base + kDeltaExt;
    std::filesystem::rename(merge_path, delta_path);

    return Ret(0);
}

Ret DatasetWriter::write_accumulator_range_(const std::string& path,
        const std::vector<uint64_t>& ids, const std::vector<uint64_t>& deleted_ids) const {
    uint64_t min_id = ids.empty() ? 0 : ids.front();
    uint64_t max_id = ids.empty() ? 0 : ids.back();
#ifndef NDEBUG
    assert(std::is_sorted(ids.begin(), ids.end()));
    assert(std::adjacent_find(ids.begin(), ids.end()) == ids.end());
    assert(std::is_sorted(deleted_ids.begin(), deleted_ids.end()));
    assert(std::adjacent_find(deleted_ids.begin(), deleted_ids.end()) == deleted_ids.end());
    assert(!accumulator_ || accumulator_->has_cosine_inv_norms() == (metadata_.dist_func == DistFunc::COS));
#endif

    DataFileHeader hdr = make_data_header(
        min_id,
        max_id,
        static_cast<uint32_t>(ids.size()),
        static_cast<uint32_t>(deleted_ids.size()),
        metadata_.type,
        static_cast<uint16_t>(metadata_.dim),
        metadata_.dist_func == DistFunc::COS);

    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        return Ret("DatasetWriter::write_accumulator_range_: failed to open output file: " + path);
    }

    std::vector<char> file_buffer(kFileBufferSize);
    (void)setvbuf(f, file_buffer.data(), _IOFBF, file_buffer.size());

    std::experimental::scope_exit file_guard([&f]() {
        if (f) fclose(f);
    });

    CHECK(write_header_and_data_padding(f, hdr, "DatasetWriter::write_accumulator_range_"));

    const size_t vec_size = static_cast<size_t>(metadata_.dim) * data_type_size(metadata_.type);
    const IdsLayout ids_layout = compute_ids_layout(hdr, ids.size());
    std::vector<float> cosine_inv_norms;
    if (metadata_.dist_func == DistFunc::COS) {
        cosine_inv_norms.reserve(ids.size());
    }
    for (uint64_t id : ids) {
        const uint8_t* data = accumulator_->get_vector(id);
        if (!data) {
            return Ret("DatasetWriter::write_accumulator_range_: missing vector for id " + std::to_string(id));
        }
        CHECK(write_vector_record(f, data, vec_size, hdr.vector_stride,
            "DatasetWriter::write_accumulator_range_: failed to write vector data for id " + std::to_string(id)));
        if (metadata_.dist_func == DistFunc::COS) {
            cosine_inv_norms.push_back(accumulator_->get_vector_cosine_inv_norm(id));
        }
    }

#ifndef NDEBUG
    assert(metadata_.dist_func != DistFunc::COS || cosine_inv_norms.size() == ids.size());
#endif
    CHECK(write_f32_array(f, cosine_inv_norms,
        "DatasetWriter::write_accumulator_range_: failed to write cosine inverse norms"));
    CHECK(write_zero_padding(f, ids_layout.ids_padding,
        "DatasetWriter::write_accumulator_range_: failed to write id alignment padding"));
    CHECK(write_u64_array(f, ids, "DatasetWriter::write_accumulator_range_: failed to write ids"));
    CHECK(write_u64_array(f, deleted_ids, "DatasetWriter::write_accumulator_range_: failed to write deleted_ids"));

    const int n1 = fflush(f);
    const int n2 = fsync(fileno(f));
    const int n3 = fclose(f);
    f = nullptr;
    if (n1 != 0 || n2 != 0 || n3 != 0) {
        return Ret("DatasetWriter::write_accumulator_range_: failed to flush and close file");
    }

    return Ret(0);
}

} // namespace sketch2
