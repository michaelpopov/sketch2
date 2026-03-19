// Declares DatasetWriter: write operations, accumulator, and WAL management.

#pragma once
#include "dataset_reader.h"
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
class DatasetWriter : public DatasetReader {
public:
    using DatasetReader::DatasetReader;
    ~DatasetWriter() override;

    // AccumulatorIterator exposes pending in-memory writes for administrative
    // lookups (e.g. finding a vector id by value before it is flushed).
    class AccumulatorIterator {
    public:
        void next();
        bool eof() const;
        uint64_t id() const;
        const uint8_t* data() const;
        float cosine_inv_norm() const;

    private:
        friend class DatasetWriter;
        explicit AccumulatorIterator(Accumulator::Iterator iterator)
            : iterator_(std::move(iterator)) {}

        Accumulator::Iterator iterator_;
    };

    // init() overrides acquire the owner lock and replay any pending WAL.
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

    // Checks the accumulator first (unflushed writes visible without merging),
    // then falls back to the persisted read path.
    std::pair<const uint8_t*, Ret> get_vector(uint64_t id) const {
        auto [vec, in_accumulator] = get_vector_from_accumulator_(id);
        if (in_accumulator) {
            return {vec, Ret(0)};
        }
        return DatasetReader::get_vector(id);
    }

    AccumulatorIterator accumulator_begin() const;

    bool has_accumulator() const { return static_cast<bool>(accumulator_); }
    bool accumulator_has_cosine_inv_norms() const {
        return accumulator_ && accumulator_->has_cosine_inv_norms();
    }
    size_t accumulator_vectors_count() const { return accumulator_ ? accumulator_->vectors_count() : 0; }
    size_t accumulator_deleted_count() const { return accumulator_ ? accumulator_->deleted_count() : 0; }

private:
    std::mutex write_mutex_;
    mutable std::unique_ptr<FileLockGuard> owner_lock_;
    mutable std::unique_ptr<Accumulator> accumulator_;
    mutable std::unique_ptr<UpdateNotifier> update_notifier_;

    Ret init_writer_();
    Ret ensure_owner_lock_();
    Ret init_accumulator_();
    Ret ensure_update_notifier_() const;
    void notify_update_(const char* caller);

    std::pair<const uint8_t*, bool> get_vector_from_accumulator_(uint64_t id) const;

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
