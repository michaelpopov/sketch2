#pragma once

#include "utils/shared_types.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sketch2 {

class Accumulator {
public:
    Ret init(size_t size, DataType type, uint64_t dim);
    void clear();

    Ret add_vector(uint64_t id, const uint8_t* data);
    Ret delete_vector(uint64_t id);

    bool can_add_vector(uint64_t id) const;
    bool can_delete_vector(uint64_t id) const;

    size_t vectors_count() const { return vector_ids_.size(); }
    size_t deleted_count() const { return deleted_ids_.size(); }

    std::vector<uint64_t> get_vector_ids() const;
    std::vector<uint64_t> get_deleted_ids() const;

    const uint8_t* get_vector(uint64_t id) const;

private:
    size_t vector_size_() const { return static_cast<size_t>(dim_) * data_type_size(type_); }
    size_t vector_record_size_() const { return sizeof(uint64_t) + vector_size_(); }
    bool is_initialized_() const { return data_size_ != 0; }
    size_t add_vector_size_(uint64_t id) const;
    size_t delete_vector_size_(uint64_t id) const;

    size_t data_size_ = 0;
    size_t used_size_ = 0;
    DataType type_ = DataType::f32;
    uint64_t dim_ = 0;
    std::unordered_map<uint64_t, size_t> vector_index_;
    std::vector<uint64_t> vector_ids_;
    std::vector<uint8_t> vector_data_;
    std::unordered_set<uint64_t> deleted_ids_;
};

} // namespace sketch2
