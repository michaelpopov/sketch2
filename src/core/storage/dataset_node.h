// Declares DatasetNode: an adapter that combines DatasetReader + DatasetWriter.

#pragma once
#include "dataset_reader.h"
#include "dataset_writer.h"
#include <memory>
#include <string>

namespace sketch2 {

class DatasetNode {
public:
    DatasetNode() = default;

    Ret init(const std::string& path);

    Ret store(const std::string& input_path);
    Ret store_accumulator();
    Ret merge();
    Ret add_vector(uint64_t id, const uint8_t* data);
    Ret delete_vector(uint64_t id);

    DatasetRangeReaderPtr reader() const;
    std::pair<DataReaderPtr, Ret> get(uint64_t id) const;
    std::pair<const uint8_t*, Ret> get_vector(uint64_t id) const;

    DataType type() const;
    DistFunc dist_func() const;
    uint64_t dim() const;
    uint64_t range_size() const;
    const std::vector<std::string>& dirs() const;

    size_t accumulator_vectors_count() const;
    size_t accumulator_deleted_count() const;

    const DatasetReader& reader_dataset() const;
    operator const DatasetReader&() const { return reader_dataset(); }

    Ret init_for_test(const DatasetMetadata& metadata);
    Ret init_for_test(const std::vector<std::string>& dirs, uint64_t range_size,
        DataType type = DataType::f32, uint64_t dim = 4,
        uint64_t accumulator_size = kAccumulatorBufferSize,
        DistFunc dist_func = DistFunc::L1);

private:
    Ret ensure_initialized_() const;

    std::unique_ptr<DatasetReader> reader_;
    std::unique_ptr<DatasetWriter> writer_;
};

} // namespace sketch2
