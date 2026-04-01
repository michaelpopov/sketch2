// Declares DatasetWriter: write and merge operations.

#pragma once
#include "dataset.h"
#include "input_writer.h"
#include "core/utils/file_lock.h"
#include "core/utils/update_notifier.h"
#include <memory>
#include <mutex>
#include <string>

namespace sketch2 {

class DataReader;
class InputReader;

// DatasetWriter owns the write infrastructure: mutex, owner lock, and the updater-
// mode UpdateNotifier for cross-process cache invalidation.
class DatasetWriter : public Dataset {
public:
    ~DatasetWriter() override;

    // init() overrides replay any pending WAL; owner lock is acquired lazily
    // when a write path first needs ownership. Only the INI-based path is
    // exposed publicly; the other overloads remain for internal helpers.
    Ret init(const std::string& path);

    Ret store(const std::string& input_path);
    Ret merge();

    Ret start_writing();
    Ret write_vector(uint64_t id, const char* vector);
    Ret write_deleted(uint64_t id);
    Ret abort_writing();
    Ret complete_writing();

private:
    std::mutex write_mutex_;
    std::unique_ptr<FileLockGuard> owner_lock_;
    bool owner_path_registered_ = false;
    std::unique_ptr<UpdateNotifier> update_notifier_;
    std::unique_ptr<InputWriter> input_writer_;

    Ret init_writer_();
    Ret ensure_owner_lock_();
    Ret ensure_update_notifier_();
    void notify_update_(const char* caller);

    Ret store_(const std::string& input_path);
    Ret merge_();
    Ret store_and_merge(const InputReader& reader, uint64_t file_id,
        uint64_t range_start, uint64_t range_end) const;

    bool check_data_file_merge(const DataReader& data_reader,
        const DataReader& output_reader) const;
    bool check_data_delta_merge(const DataReader& data_reader,
        const DataReader& delta_reader) const;
    Ret merge_data_file(const DataReader& data_reader, const DataReader& output_reader,
        const std::string& output_path_base, const std::string& ext) const;
    Ret merge_delta_file(const DataReader& delta_reader, const DataReader& output_reader,
        const std::string& output_path_base) const;
};

} // namespace sketch2
