// Declares DatasetWriter: write operations, accumulator, and WAL management.

#pragma once
#include "dataset.h"
#include "core/storage/accumulator.h"
#include "core/utils/file_lock.h"
#include "core/utils/update_notifier.h"
#include <memory>
#include <mutex>

namespace sketch2 {

class DataReader;
class InputReader;

// DatasetWriter owns the write infrastructure: mutex, owner lock, accumulator,
// and the updater-mode UpdateNotifier for cross-process cache invalidation.
class DatasetWriter : public Dataset {
public:
    ~DatasetWriter() override;

    // init() overrides replay any pending WAL; owner lock is acquired lazily
    // when a write path first needs ownership.
    Ret init(const DatasetMetadata& metadata);
    Ret init(const std::vector<std::string>& dirs, uint64_t range_size,
            DataType type = DataType::f32, uint64_t dim = 4,
            uint64_t accumulator_size = kAccumulatorBufferSize,
            DistFunc dist_func = DistFunc::L1);
    Ret init(const std::string& path);

    Ret store(const std::string& input_path);
    Ret store_accumulator();
    Ret merge();
    Ret add_vector(uint64_t id, const uint8_t* data);
    Ret delete_vector(uint64_t id);

    size_t accumulator_vectors_count() const { return accumulator_ ? accumulator_->vectors_count() : 0; }
    size_t accumulator_deleted_count() const { return accumulator_ ? accumulator_->deleted_count() : 0; }

private:
    std::mutex write_mutex_;
    mutable std::unique_ptr<FileLockGuard> owner_lock_;
    bool owner_path_registered_ = false;
    mutable std::unique_ptr<Accumulator> accumulator_;
    mutable std::unique_ptr<UpdateNotifier> update_notifier_;

    Ret init_writer_();
    Ret ensure_owner_lock_();
    Ret init_accumulator_();
    Ret ensure_update_notifier_() const;
    void notify_update_(const char* caller);

    Ret store_(const std::string& input_path);
    Ret store_accumulator_();
    Ret merge_();
    Ret store_and_merge(const InputReader& reader, uint64_t file_id,
        uint64_t range_start, uint64_t range_end) const;
    Ret store_and_merge_accumulator(uint64_t file_id,
        const std::vector<uint64_t>& ids, const std::vector<uint64_t>& deleted_ids);
    Ret write_accumulator_range_(const std::string& path,
        const std::vector<uint64_t>& ids, const std::vector<uint64_t>& deleted_ids) const;

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
