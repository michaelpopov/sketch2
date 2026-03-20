// Implements DatasetNode: a small adapter over DatasetReader + DatasetWriter.

#include "dataset_node.h"
#include <stdexcept>

namespace sketch2 {

Ret DatasetNode::ensure_initialized_() const {
    if (!reader_ || !writer_) {
        return Ret("DatasetNode: not initialized.");
    }
    return Ret(0);
}

Ret DatasetNode::init(const DatasetMetadata& metadata) {
    if (reader_ || writer_) {
        return Ret("DatasetNode is already initialized.");
    }

    auto reader = std::make_unique<DatasetReader>();
    auto writer = std::make_unique<DatasetWriter>();
    CHECK(reader->init(metadata));
    CHECK(writer->init(metadata));

    reader_ = std::move(reader);
    writer_ = std::move(writer);
    return Ret(0);
}

Ret DatasetNode::init(const std::vector<std::string>& dirs, uint64_t range_size,
        DataType type, uint64_t dim, uint64_t accumulator_size, DistFunc dist_func) {
    if (reader_ || writer_) {
        return Ret("DatasetNode is already initialized.");
    }

    auto reader = std::make_unique<DatasetReader>();
    auto writer = std::make_unique<DatasetWriter>();
    CHECK(reader->init(dirs, range_size, type, dim, accumulator_size, dist_func));
    CHECK(writer->init(dirs, range_size, type, dim, accumulator_size, dist_func));

    reader_ = std::move(reader);
    writer_ = std::move(writer);
    return Ret(0);
}

Ret DatasetNode::init(const std::string& path) {
    if (reader_ || writer_) {
        return Ret("DatasetNode is already initialized.");
    }

    auto reader = std::make_unique<DatasetReader>();
    auto writer = std::make_unique<DatasetWriter>();
    CHECK(reader->init(path));
    CHECK(writer->init(path));

    reader_ = std::move(reader);
    writer_ = std::move(writer);
    return Ret(0);
}

Ret DatasetNode::store(const std::string& input_path) {
    CHECK(ensure_initialized_());
    return writer_->store(input_path);
}

Ret DatasetNode::store_accumulator() {
    CHECK(ensure_initialized_());
    return writer_->store_accumulator();
}

Ret DatasetNode::merge() {
    CHECK(ensure_initialized_());
    return writer_->merge();
}

Ret DatasetNode::add_vector(uint64_t id, const uint8_t* data) {
    CHECK(ensure_initialized_());
    return writer_->add_vector(id, data);
}

Ret DatasetNode::delete_vector(uint64_t id) {
    CHECK(ensure_initialized_());
    return writer_->delete_vector(id);
}

DatasetRangeReaderPtr DatasetNode::reader() const {
    if (!reader_) {
        throw std::runtime_error("DatasetNode::reader: not initialized");
    }
    return reader_->reader();
}

std::pair<DataReaderPtr, Ret> DatasetNode::get(uint64_t id) const {
    const Ret ret = ensure_initialized_();
    if (ret.code() != 0) {
        return {nullptr, ret};
    }
    return reader_->get(id);
}

std::pair<const uint8_t*, Ret> DatasetNode::get_vector(uint64_t id) const {
    const Ret ret = ensure_initialized_();
    if (ret.code() != 0) {
        return {nullptr, ret};
    }
    return reader_->get_vector(id);
}

DataType DatasetNode::type() const {
    if (!reader_) {
        throw std::runtime_error("DatasetNode::type: not initialized");
    }
    return reader_->type();
}

DistFunc DatasetNode::dist_func() const {
    if (!reader_) {
        throw std::runtime_error("DatasetNode::dist_func: not initialized");
    }
    return reader_->dist_func();
}

uint64_t DatasetNode::dim() const {
    if (!reader_) {
        throw std::runtime_error("DatasetNode::dim: not initialized");
    }
    return reader_->dim();
}

uint64_t DatasetNode::range_size() const {
    if (!reader_) {
        throw std::runtime_error("DatasetNode::range_size: not initialized");
    }
    return reader_->range_size();
}

const std::vector<std::string>& DatasetNode::dirs() const {
    if (!reader_) {
        throw std::runtime_error("DatasetNode::dirs: not initialized");
    }
    return reader_->dirs();
}

size_t DatasetNode::accumulator_vectors_count() const {
    if (!writer_) {
        throw std::runtime_error("DatasetNode::accumulator_vectors_count: not initialized");
    }
    return writer_->accumulator_vectors_count();
}

size_t DatasetNode::accumulator_deleted_count() const {
    if (!writer_) {
        throw std::runtime_error("DatasetNode::accumulator_deleted_count: not initialized");
    }
    return writer_->accumulator_deleted_count();
}

const DatasetReader& DatasetNode::reader_dataset() const {
    if (!reader_) {
        throw std::runtime_error("DatasetNode::reader_dataset: not initialized");
    }
    return *reader_;
}

} // namespace sketch2
