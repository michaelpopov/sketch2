// Implements DatasetNode: a small adapter over DatasetReader + DatasetWriter.

#include "dataset.h"
#include "dataset_node.h"
#include <filesystem>
#include <stdexcept>
#include <unistd.h>

namespace sketch2 {

Ret DatasetNode::ensure_initialized_() const {
    if (!reader_ || !writer_) {
        return Ret("DatasetNode: not initialized.");
    }
    return Ret(0);
}

Ret DatasetNode::init_for_test(const DatasetMetadata& metadata) {
    if (reader_ || writer_) {
        return Ret("DatasetNode is already initialized.");
    }

    if (metadata.dirs.empty()) {
        return Ret("DatasetNode: dirs must not be empty.");
    }

    if (!std::filesystem::exists(metadata.dirs.front())) {
        return Ret("DatasetNode: dir must exist.");
    }

    // Write the ini either alongside the dataset.
    std::string ini_path = metadata.dirs.front() + "/dataset.ini";

    CHECK(write_dataset_ini(metadata, ini_path));

    auto reader = std::make_unique<DatasetReader>();
    auto writer = std::make_unique<DatasetWriter>();
    CHECK(reader->init(ini_path));
    CHECK(writer->init(ini_path));

    reader_ = std::move(reader);
    writer_ = std::move(writer);
    return Ret(0);
}

Ret DatasetNode::init_for_test(const std::vector<std::string>& dirs, uint64_t range_size,
        DataType type, uint64_t dim, DistFunc dist_func) {
    DatasetMetadata metadata;
    metadata.dirs = dirs;
    metadata.range_size = range_size;
    metadata.type = type;
    metadata.dim = dim;
    metadata.dist_func = dist_func;
    return init_for_test(metadata);
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

Ret DatasetNode::merge() {
    CHECK(ensure_initialized_());
    return writer_->merge();
}

Ret DatasetNode::start_writing() {
    CHECK(ensure_initialized_());
    return writer_->start_writing();
}

Ret DatasetNode::write_vector(uint64_t id, const char* vector) {
    CHECK(ensure_initialized_());
    return writer_->write_vector(id, vector);
}

Ret DatasetNode::write_deleted(uint64_t id) {
    CHECK(ensure_initialized_());
    return writer_->write_deleted(id);
}

Ret DatasetNode::abort_writing() {
    CHECK(ensure_initialized_());
    return writer_->abort_writing();
}

Ret DatasetNode::complete_writing() {
    CHECK(ensure_initialized_());
    return writer_->complete_writing();
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

std::pair<std::string, Ret> DatasetNode::get_vector_string(uint64_t id, size_t digits) const {
    const Ret ret = ensure_initialized_();
    if (ret.code() != 0) {
        return {std::string{}, ret};
    }
    return reader_->get_vector_string(id, digits);
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

const DatasetReader& DatasetNode::reader_dataset() const {
    if (!reader_) {
        throw std::runtime_error("DatasetNode::reader_dataset: not initialized");
    }
    return *reader_;
}

} // namespace sketch2
