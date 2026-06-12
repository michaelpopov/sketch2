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
class InputReaderView;

// DatasetWriter owns the write infrastructure: mutexes, owner lock, and the
// updater-mode UpdateNotifier for cross-process cache invalidation.
//
// Lifetime contract: callers must not destroy a DatasetWriter while any public
// method is still running on another thread. Internal mutexes serialize access
// to writer state, but they do not provide object-lifetime management.
class DatasetWriter : public Dataset {
public:
    ~DatasetWriter() override;

    // init() cleans up stale staged-input files if this process can briefly
    // take ownership. The owner lock is acquired lazily when a write path first
    // needs ownership. Only the INI-based path is exposed publicly; the other
    // overloads remain for internal helpers.
    Ret init(const std::string& path);

    Ret store(const std::string& input_path);
    Ret merge();

    Ret start_writing();
    Ret write_vector(uint64_t id, const char* vector);
    Ret write_deleted(uint64_t id);
    Ret abort_writing();
    Ret complete_writing();

private:
    // Locking protocol:
    // - input_session_mutex_ guards the active InputWriter session state.
    // - dataset_files_mutex_ guards store_/merge_ and persisted dataset files.
    // - state_mutex_ guards shared lazy-initialized writer state such as
    //   owner_lock_ and update_notifier_.
    // - Never hold input_session_mutex_ and dataset_files_mutex_
    //   simultaneously; complete_writing() hands work off between them.
    // - state_mutex_ is always the innermost lock.
    // - These mutexes protect state access only; they do not make destruction
    //   safe while other threads are executing methods on this object.
    std::mutex input_session_mutex_;
    std::mutex dataset_files_mutex_;
    std::mutex state_mutex_;
    std::unique_ptr<FileLockGuard> owner_lock_;
    bool owner_path_registered_ = false;
    std::unique_ptr<UpdateNotifier> update_notifier_;
    std::unique_ptr<InputWriter> input_writer_;
    std::string active_input_path_;
    uint64_t next_input_session_id_ = 0;

    Ret init_writer_();
    Ret ensure_owner_lock_();
    Ret ensure_update_notifier_();
    void notify_update_(const char* caller);

    Ret store_(const std::string& input_path);
    Ret merge_();
    Ret store_and_merge(const InputReader& reader, uint64_t file_id,
        uint64_t range_start, uint64_t range_end) const;

    bool check_data_file_merge(const DataReader& data_reader,
        uint64_t output_count) const;
    bool check_data_delta_merge(const DataReader& data_reader,
        const DataReader& delta_reader) const;
    Ret merge_data_file(const DataReader& data_reader, const DataReader& output_reader,
        const std::string& output_path_base, const std::string& ext) const;
    Ret merge_data_file(const DataReader& data_reader, const InputReaderView& output_reader,
        const std::string& output_path_base) const;
    Ret merge_delta_file(const DataReader& delta_reader, const InputReaderView& output_reader,
        const std::string& output_path_base) const;
    Ret garbage_collect_();
};

} // namespace sketch2
