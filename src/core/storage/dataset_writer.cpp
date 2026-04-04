#include "dataset_writer.h"
#include "core/storage/data_file_layout.h"
#include "core/storage/data_reader.h"
#include "core/storage/data_merger.h"
#include "core/storage/data_writer.h"
#include "core/storage/input_reader.h"
#include "core/storage/input_writer.h"
#include "core/utils/file_lock.h"
#include "core/utils/log.h"
#include "core/utils/singleton.h"
#include "core/utils/thread_pool.h"
#include "core/utils/timer.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <experimental/scope>
#include <future>
#include <limits>
#include <sstream>
#include <unistd.h>

namespace sketch2 {

namespace {

struct StoreRangeTask {
    uint64_t file_id;
    uint64_t range_start;
    uint64_t range_end;
};

struct InputRangeStats {
    uint64_t active_count = 0;
    uint64_t deleted_count = 0;
};

std::string dataset_input_path(const DatasetMetadata& metadata, const std::string& dataset_name) {
    // Use the per-dataset lock file as the base for the staged input filename.
    std::filesystem::path path = dataset_owner_lock_path(metadata, dataset_name);
    path.replace_extension(".input");
    return path.string();
}

InputRangeStats summarize_input_view(const InputReaderView& view) {
    // The store path only needs a coarse summary here: how many live rows and
    // tombstones the incoming range contains. That is enough to drive the
    // merge heuristic without first converting the whole range into a temp
    // persisted file just to reopen it as a DataReader.
    InputRangeStats stats;
    for (size_t i = 0; i < view.count(); ++i) {
        if (view.is_no_data(i)) {
            ++stats.deleted_count;
        } else {
            ++stats.active_count;
        }
    }
    return stats;
}

} // namespace

/***********************************************************
 *  DatasetWriter lifecycle
 */

DatasetWriter::~DatasetWriter() {
    if (owner_path_registered_ && !metadata_.dirs.empty()) {
        const std::string lock_path = dataset_owner_lock_path(metadata_, name_);
        const bool ok = Singleton::instance().release_file_path(lock_path);
        if (!ok) {
            LOG_ERROR << "DatasetWriter destructor failed to release locked file path";
        }
        owner_path_registered_ = false;
    }
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
    const std::string lock_path = dataset_owner_lock_path(metadata_, name_);
    {
        FileLockGuard temp_lock;
        if (!temp_lock.try_lock(lock_path)) {
            return Ret(0);
        }
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
        LOG_TRACE << "Storing data from input file " << input_path;
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

Ret DatasetWriter::merge() {
    Ret ret{0};
    bool should_notify = false;
    try {
        std::lock_guard<std::mutex> lg(write_mutex_);
        CHECK(ensure_owner_lock_());
        should_notify = true;
        Timer timer("DatasetWriter::merge");
        ret = merge_();

        if (ret.code() == 0) {
            LOG_INFO << "Successfully completed DatasetWriter::merge for " << name() << " in " << timer.elapsed_ms() << " ms";
            ret = garbage_collect_();
            if (ret.code() != 0) {
                LOG_WARN << "Failed to delete unused data files for " << name() << ": " << ret.message();
            }
        } else {
            LOG_WARN << "Failed to complete DatasetWriter::merge for " << name() << ": " << ret.message();
        }

    } catch (const std::exception& ex) {
        ret = Ret(ex.what());
    }

    if (should_notify) {
        notify_update_("DatasetWriter::merge");
    }

    return ret;
}

Ret DatasetWriter::start_writing() {
    try {
        std::lock_guard<std::mutex> lg(write_mutex_);
        CHECK(ensure_owner_lock_());
        if (input_writer_) {
            return Ret("DatasetWriter::start_writing: input writer is active already");
        }

        input_writer_ = std::make_unique<InputWriter>();
        const Ret ret = input_writer_->init(metadata_.type, metadata_.dim, dataset_input_path(metadata_, name_));
        if (ret.code() != 0) {
            input_writer_.reset();
            return ret;
        }
        return Ret(0);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret DatasetWriter::write_vector(uint64_t id, const char* vector) {
    try {
        std::lock_guard<std::mutex> lg(write_mutex_);
        CHECK(ensure_owner_lock_());
        if (!input_writer_) {
            return Ret("DatasetWriter::write_vector: input writer is not active");
        }
        return input_writer_->write_vector(id, vector);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret DatasetWriter::write_deleted(uint64_t id) {
    try {
        std::lock_guard<std::mutex> lg(write_mutex_);
        CHECK(ensure_owner_lock_());
        if (!input_writer_) {
            return Ret("DatasetWriter::write_deleted: input writer is not active");
        }
        return input_writer_->write_deleted(id);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret DatasetWriter::abort_writing() {
    try {
        std::lock_guard<std::mutex> lg(write_mutex_);
        CHECK(ensure_owner_lock_());
        if (!input_writer_) {
            return Ret("DatasetWriter::abort_writing: input writer is not active");
        }

        const std::string input_path = dataset_input_path(metadata_, name_);
        CHECK(input_writer_->abort_writing());
        input_writer_.reset();

        std::error_code ec;
        std::filesystem::remove(input_path, ec);
        if (ec) {
            return Ret("DatasetWriter::abort_writing: failed to remove temporary input file");
        }
        return Ret(0);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret DatasetWriter::complete_writing() {
    Ret ret{0};
    bool should_notify = false;
    try {
        std::lock_guard<std::mutex> lg(write_mutex_);
        CHECK(ensure_owner_lock_());
        if (!input_writer_) {
            return Ret("DatasetWriter::complete_writing: input writer is not active");
        }

        const std::string input_path = dataset_input_path(metadata_, name_);
        std::experimental::scope_exit cleanup([&input_path]() {
            std::error_code ec;
            std::filesystem::remove(input_path, ec);
        });

        CHECK(input_writer_->close_file());
        input_writer_.reset();

        should_notify = true;
        ret = store_(input_path);
    } catch (const std::exception& ex) {
        ret = Ret(ex.what());
    }

    if (should_notify) {
        notify_update_("DatasetWriter::complete_writing");
    }

    return ret;
}

/***********************************************************
 *  Private helpers
 */

Ret DatasetWriter::ensure_owner_lock_() {
    if (owner_lock_ || metadata_.dirs.empty()) {
        return Ret(0);
    }

    owner_lock_ = std::make_unique<FileLockGuard>();
    const std::string lock_path = dataset_owner_lock_path(metadata_, name_);
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

Ret DatasetWriter::ensure_update_notifier_() {
    if (update_notifier_) {
        return Ret(0);
    }
    if (metadata_.dirs.empty()) {
        return Ret(0);
    }
    update_notifier_ = std::make_unique<UpdateNotifier>();
    return update_notifier_->init_updater(dataset_owner_lock_path(metadata_, name_));
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

Ret DatasetWriter::garbage_collect_() {
    std::vector<DatasetItem> all_items;
    CHECK(collect_dataset_items(name_, metadata_, &all_items));

    for (const DatasetItem& item : all_items) {
        if (!item.delta_file_path.empty()) {
            continue;
        }

        DataReader data_reader;
        CHECK(data_reader.init(item.data_file_path));
        if (data_reader.count() == 0) {
            std::error_code ec;
            std::filesystem::remove(item.data_file_path, ec);
            if (ec) {
                LOG_WARN << "Failed to remove file " << item.data_file_path;
            } else {
                LOG_TRACE << "Removed file " << item.data_file_path;
            }
        }
    }

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
    const std::string temp_path_base = item_path_base(file_id);
    const std::string temp_path = temp_path_base + kTempExt;
    const std::string data_path = temp_path_base + kDataExt;
    const std::string delta_path = temp_path_base + kDeltaExt;

    // One range of ids maps to one persisted data/delta pair on disk. The
    // InputReaderView keeps this function focused on a single range so every
    // decision below is purely local to that range.
    InputReaderView view(reader, range_start, range_end);
    const InputRangeStats stats = summarize_input_view(view);

    // Writing a temp file is still the simplest way to produce a persisted
    // object that can be renamed into place. We keep that path for "rename"
    // cases, but avoid it for the branches that immediately merge the result.
    auto write_temp_file = [&]() -> Ret {
        DataWriter writer;
        return writer.load(view, temp_path, metadata_.dist_func == DistFunc::COS);
    };

    // Execution flow:
    // 1. No base file yet            -> create first `.data` file by temp+rename.
    // 2. Base exists, no delta yet   -> either merge directly into `.data` or
    //                                   create first `.delta`.
    // 3. Base and delta both exist   -> merge incoming updates straight into the
    //                                   existing delta, then optionally compact
    //                                   delta back into data.
    //
    // The key idea is that rename paths still use a temp file because it keeps
    // the code straightforward and atomic, while merge paths now skip the temp
    // file to avoid writing the same logical update twice.

    // If a data file for the range doesn't exist, the temp file becomes the
    // first base file for that range. Deletes are rejected here because a brand
    // new range cannot meaningfully "delete" ids that were never persisted.
    if (!std::filesystem::exists(data_path)) {
        if (stats.deleted_count != 0) {
            return Ret("DatasetWriter::store_and_merge: invalid deleted items");
        }
        std::experimental::scope_exit file_guard([temp_path]() {
            std::error_code ec;
            std::filesystem::remove(temp_path, ec);
        });
        CHECK(write_temp_file());
        std::filesystem::rename(temp_path, data_path);
        LOG_TRACE << "DatasetWriter: loaded data to data file " << data_path;
        return Ret(0);
    }

    // A base file already exists, but there is no delta yet.
    if (!std::filesystem::exists(delta_path)) {
        DataReader data_reader;
        CHECK(data_reader.init(data_path));

        // If the new batch is "large enough" relative to the base file, writing
        // a separate delta would just postpone an inevitable full rewrite. In
        // that case we merge the input view straight into the base file now.
        //
        // This is the main optimization added here: the old code first wrote
        // `view` to `*.tmp`, reopened it as DataReader, and only then merged it.
        // We now pass the view directly to DataMerger and skip that extra write.
        const bool is_merge = check_data_file_merge(data_reader, stats.active_count + stats.deleted_count);
        if (is_merge) {
            CHECK(merge_data_file(data_reader, view, temp_path_base));
            return Ret(0);
        }

        // The incoming batch is small enough to keep as a delta. This path still
        // uses temp+rename because it is a cheap, explicit way to create the
        // first delta file with the normal on-disk format.
        std::experimental::scope_exit file_guard([temp_path]() {
            std::error_code ec;
            std::filesystem::remove(temp_path, ec);
        });
        CHECK(write_temp_file());
        std::filesystem::rename(temp_path, delta_path);
        LOG_TRACE << "DatasetWriter: loaded data to delta file " << delta_path;
        return Ret(0);
    }

    // Both files already exist. There is no rename shortcut left here, so the
    // best option is to merge the input view directly into the persisted delta.
    // That keeps the existing "delta absorbs new writes" behavior while removing
    // the old temp-file detour for this hot path.
    {
        DataReader delta_reader;
        CHECK(delta_reader.init(delta_path));
        CHECK(merge_delta_file(delta_reader, view, temp_path_base));
    }

    // After the delta was updated, re-evaluate the normal compaction heuristic.
    // If the delta grew too large relative to the base file, fold it back into
    // the base file so reads do not pay an ever-growing overlay cost forever.
    {
        DataReader data_reader;
        CHECK(data_reader.init(data_path));
        DataReader delta_reader;
        CHECK(delta_reader.init(delta_path));

        const bool is_data_delta_merge = check_data_delta_merge(data_reader, delta_reader);
        if (is_data_delta_merge) {
            CHECK(merge_data_file(data_reader, delta_reader, temp_path_base, kDeltaExt));
        }
    }

    return Ret(0);
}

bool DatasetWriter::check_data_file_merge(const DataReader& data_reader,
        const DataReader& output_reader) const {
    // This overload is convenient when the candidate updater is already stored
    // as a DataReader-backed file.
    const uint64_t output_count = output_reader.count() + output_reader.deleted_count();
    return check_data_file_merge(data_reader, output_count);
}

bool DatasetWriter::check_data_file_merge(const DataReader& data_reader,
        uint64_t output_count) const {
    // The heuristic compares update volume to the size of the current base
    // file. Large updates go straight into a rewritten base file; small updates
    // become or remain a delta.
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
    std::error_code ec;
    std::filesystem::rename(merge_path, data_path, ec);
    if (ec) {
        std::filesystem::remove(merge_path, ec);
        return Ret("DatasetWriter::merge_data_file: failed to rename merge file to data file");
    }

    return Ret(0);
}

Ret DatasetWriter::merge_data_file(const DataReader& data_reader, const InputReaderView& output_reader,
        const std::string& output_path_base) const {
    // This overload exists specifically for the direct-input fast path in
    // store_and_merge(). It produces the same final `*.merge -> *.data`
    // transition as the DataReader overload, but the updater comes straight
    // from parsed input instead of a temp persisted file.
    DataMerger processor;
    const std::string merge_path = output_path_base + kMergeExt;
    CHECK(processor.merge_data_file(data_reader, output_reader, merge_path));

    const std::string data_path = output_path_base + kDataExt;
    std::error_code ec;
    std::filesystem::rename(merge_path, data_path, ec);
    if (ec) {
        std::filesystem::remove(merge_path, ec);
        return Ret("DatasetWriter::merge_data_file: failed to rename merge file to data file");
    }

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
    std::error_code ec;
    std::filesystem::rename(merge_path, delta_path, ec);
    if (ec) {
        std::filesystem::remove(merge_path, ec);
        return Ret("DatasetWriter::merge_delta_file: failed to rename merge file to delta file");
    }

    return Ret(0);
}

Ret DatasetWriter::merge_delta_file(const DataReader& delta_reader, const InputReaderView& output_reader,
        const std::string& output_path_base) const {
    // Same idea as merge_data_file(view): keep the atomic "write merge output,
    // then rename it into place" behavior, but avoid staging the updater as a
    // separate temp data file first.
    DataMerger processor;
    const std::string merge_path = output_path_base + kMergeExt;
    CHECK(processor.merge_delta_file(delta_reader, output_reader, merge_path));

    const std::string delta_path = output_path_base + kDeltaExt;
    std::error_code ec;
    std::filesystem::rename(merge_path, delta_path, ec);
    if (ec) {
        std::filesystem::remove(merge_path, ec);
        return Ret("DatasetWriter::merge_delta_file: failed to rename merge file to delta file");
    }

    return Ret(0);
}

} // namespace sketch2
