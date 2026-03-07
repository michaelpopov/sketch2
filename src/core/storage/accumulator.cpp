#include "accumulator.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace sketch2 {

Ret Accumulator::init(size_t size, DataType type, uint64_t dim) {
    if (is_initialized_()) {
        return Ret("Accumulator is already initialized.");
    }
    if (size == 0) {
        return Ret("Accumulator: buffer is too small");
    }
    if (dim < 4) {
        return Ret("Accumulator: dim must be >= 4.");
    }

    try {
        validate_type(type);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }

    type_ = type;
    dim_ = dim;
    data_size_ = size;
    used_size_ = 0;
    vector_data_.reserve(data_size_);
    const size_t record_size = vector_record_size_();
    if (record_size != 0) {
        const size_t max_vectors = data_size_ / record_size;
        vector_ids_.reserve(max_vectors);
        vector_index_.reserve(max_vectors);
    }
    return Ret(0);
}

void Accumulator::clear() {
    vector_index_.clear();
    vector_ids_.clear();
    vector_data_.clear();
    deleted_ids_.clear();
    used_size_ = 0;
}

Ret Accumulator::add_vector(uint64_t id, const uint8_t* data) {
    if (!is_initialized_()) {
        return Ret("Accumulator: not initialized");
    }
    if (!data) {
        return Ret("Accumulator: vector data is null");
    }
    if (!can_add_vector(id)) {
        return Ret("Accumulator: buffer full");
    }

    const bool had_deleted = deleted_ids_.find(id) != deleted_ids_.end();
    const auto vector_it = vector_index_.find(id);
    const bool had_vector = vector_it != vector_index_.end();

    if (had_deleted) {
        deleted_ids_.erase(id);
        used_size_ -= sizeof(uint64_t);
    }

    const size_t vector_size = vector_size_();
    if (had_vector) {
        std::memcpy(vector_data_.data() + vector_it->second * vector_size, data, vector_size);
    } else {
        const size_t slot = vector_ids_.size();
        vector_index_[id] = slot;
        vector_ids_.push_back(id);
        const size_t old_size = vector_data_.size();
        vector_data_.resize(old_size + vector_size);
        std::memcpy(vector_data_.data() + old_size, data, vector_size);
        used_size_ += vector_record_size_();
    }

    return Ret(0);
}

Ret Accumulator::delete_vector(uint64_t id) {
    if (!is_initialized_()) {
        return Ret("Accumulator: not initialized");
    }

    const auto vector_it = vector_index_.find(id);
    const bool had_vector = vector_it != vector_index_.end();
    if (!had_vector && deleted_ids_.find(id) != deleted_ids_.end()) {
        return Ret(0);
    }
    if (!can_delete_vector(id)) {
        return Ret("Accumulator: buffer full");
    }

    if (had_vector) {
        const size_t slot = vector_it->second;
        const size_t last_slot = vector_ids_.size() - 1;
        if (slot != last_slot) {
            const size_t vector_size = vector_size_();
            const uint64_t moved_id = vector_ids_[last_slot];
            std::memcpy(
                vector_data_.data() + slot * vector_size,
                vector_data_.data() + last_slot * vector_size,
                vector_size);
            vector_ids_[slot] = moved_id;
            vector_index_[moved_id] = slot;
        }
        vector_ids_.pop_back();
        vector_data_.resize(vector_ids_.size() * vector_size_());
        vector_index_.erase(vector_it);
        used_size_ -= vector_record_size_();
    }

    if (deleted_ids_.insert(id).second) {
        used_size_ += sizeof(uint64_t);
    }

    return Ret(0);
}

bool Accumulator::can_add_vector(uint64_t id) const {
    if (!is_initialized_()) {
        throw std::runtime_error("Accumulator::can_add_vector: not initialized");
    }
    return add_vector_size_(id) <= data_size_;
}

bool Accumulator::can_delete_vector(uint64_t id) const {
    if (!is_initialized_()) {
        throw std::runtime_error("Accumulator::can_delete_vector: not initialized");
    }
    return delete_vector_size_(id) <= data_size_;
}

std::vector<uint64_t> Accumulator::get_vector_ids() const {
    if (!is_initialized_()) {
        throw std::runtime_error("Accumulator::get_vector_ids: not initialized");
    }

    std::vector<uint64_t> ids;
    ids = vector_ids_;
    std::sort(ids.begin(), ids.end());
    return ids;
}

std::vector<uint64_t> Accumulator::get_deleted_ids() const {
    if (!is_initialized_()) {
        throw std::runtime_error("Accumulator::get_deleted_ids: not initialized");
    }

    std::vector<uint64_t> ids;
    ids.reserve(deleted_ids_.size());
    for (uint64_t id : deleted_ids_) {
        ids.push_back(id);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

const uint8_t* Accumulator::get_vector(uint64_t id) const {
    if (!is_initialized_()) {
        throw std::runtime_error("Accumulator::get_vector: not initialized");
    }

    const auto it = vector_index_.find(id);
    if (it == vector_index_.end()) {
        return nullptr;
    }
    return vector_data_.data() + it->second * vector_size_();
}

size_t Accumulator::add_vector_size_(uint64_t id) const {
    const bool had_deleted = deleted_ids_.find(id) != deleted_ids_.end();
    const bool had_vector = vector_index_.find(id) != vector_index_.end();
    const size_t freed_size = had_deleted ? sizeof(uint64_t) : 0;
    const size_t required_size = had_vector ? 0 : vector_record_size_();
    return used_size_ - freed_size + required_size;
}

size_t Accumulator::delete_vector_size_(uint64_t id) const {
    const bool had_vector = vector_index_.find(id) != vector_index_.end();
    const bool had_deleted = deleted_ids_.find(id) != deleted_ids_.end();
    const size_t freed_size = had_vector ? vector_record_size_() : 0;
    const size_t required_size = had_deleted ? 0 : sizeof(uint64_t);
    return used_size_ - freed_size + required_size;
}

} // namespace sketch2
