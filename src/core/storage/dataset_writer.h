#pragma once
#include "dataset_reader.h"

namespace sketch2 {

class DatasetWriter : public DatasetReader {
public:
    using DatasetReader::DatasetReader;
    ~DatasetWriter() override = default;

    Ret init(const DatasetMetadata& metadata) {
        Ret ret = Dataset::init(metadata);
        if (ret.code() != 0) return ret;
        return init_writer_();
    }

    Ret init(const std::vector<std::string>& dirs, uint64_t range_size,
            DataType type = DataType::f32, uint64_t dim = 4,
            uint64_t accumulator_size = kAccumulatorBufferSize,
            DistFunc dist_func = DistFunc::L1) {
        Ret ret = Dataset::init(dirs, range_size, type, dim, accumulator_size, dist_func);
        if (ret.code() != 0) return ret;
        return init_writer_();
    }

    Ret init(const std::string& path) {
        Ret ret = Dataset::init(path);
        if (ret.code() != 0) return ret;
        return init_writer_();
    }
};

} // namespace sketch2
