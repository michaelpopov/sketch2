#pragma once

#include "core/storage/accumulator_wal.h"
#include "utils/shared_types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sketch2 {

class AlignedByteBuffer {
public:
    AlignedByteBuffer() = default;
    ~AlignedByteBuffer() { reset(); }

    AlignedByteBuffer(const AlignedByteBuffer&) = delete;
    AlignedByteBuffer& operator=(const AlignedByteBuffer&) = delete;

    AlignedByteBuffer(AlignedByteBuffer&& other) noexcept { move_from_(std::move(other)); }
    AlignedByteBuffer& operator=(AlignedByteBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            move_from_(std::move(other));
        }
        return *this;
    }

    Ret init(size_t size, size_t alignment);
    void clear() { size_ = 0; }
    Ret resize(size_t size);
    void reset();

    uint8_t* data() { return data_; }
    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    bool empty() const { return size_ == 0; }

private:
    void move_from_(AlignedByteBuffer&& other) noexcept;

    uint8_t* data_ = nullptr;
    size_t size_ = 0;
    size_t capacity_ = 0;
    size_t alignment_ = 0;
};

class Accumulator {
public:
    class Iterator {
    public:
        Iterator() = default;
        void next();
        bool eof() const;
        uint64_t id() const;
        const uint8_t* data() const;
        float cosine_inv_norm() const;

    private:
        friend class Accumulator;
        Iterator(const Accumulator* accumulator, size_t index)
            : accumulator_(accumulator), index_(index) {}

        const Accumulator* accumulator_ = nullptr;
        size_t index_ = 0;
    };

    Accumulator() = default;
    ~Accumulator();

    Ret init(size_t size, DataType type, uint64_t dim, bool has_cosine_inv_norms = false);
    void clear();
    Ret attach_wal(const std::string& path);
    Ret reset_wal();

    Ret add_vector(uint64_t id, const uint8_t* data);
    Ret delete_vector(uint64_t id);

    bool can_add_vector(uint64_t id) const;
    bool can_delete_vector(uint64_t id) const;

    size_t vectors_count() const { return vector_ids_.size(); }
    size_t deleted_count() const { return deleted_ids_.size(); }

    std::vector<uint64_t> get_vector_ids() const;
    std::vector<uint64_t> get_deleted_ids() const;

    const uint8_t* get_vector(uint64_t id) const;
    float get_vector_cosine_inv_norm(uint64_t id) const;
    bool is_deleted(uint64_t id) const;
    bool is_updated(uint64_t id) const;
    bool has_cosine_inv_norms() const { return has_cosine_inv_norms_; }
    Iterator begin() const;

private:
    size_t vector_size_() const { return static_cast<size_t>(dim_) * data_type_size(type_); }
    size_t vector_stride_() const { return align_up(vector_size_(), static_cast<size_t>(kDataAlignment)); }
    size_t vector_record_size_() const {
        return sizeof(uint64_t) + vector_stride_() + (has_cosine_inv_norms_ ? sizeof(float) : 0u);
    }
    bool is_initialized_() const { return data_size_ != 0; }
    size_t add_vector_size_(uint64_t id) const;
    size_t delete_vector_size_(uint64_t id) const;
    uint8_t* vector_slot_(size_t slot) { return vector_data_.data() + slot * vector_stride_(); }
    const uint8_t* vector_slot_(size_t slot) const { return vector_data_.data() + slot * vector_stride_(); }

    size_t data_size_ = 0;
    size_t used_size_ = 0;
    DataType type_ = DataType::f32;
    uint64_t dim_ = 0;
    bool has_cosine_inv_norms_ = false;
    std::unordered_map<uint64_t, size_t> vector_index_;
    std::vector<uint64_t> vector_ids_;
    std::vector<float> vector_cosine_inv_norms_;
    AlignedByteBuffer vector_data_;
    std::unordered_set<uint64_t> deleted_ids_;
    std::unique_ptr<AccumulatorWal> wal_;

    Ret apply_add_vector_(uint64_t id, const uint8_t* data);
    Ret apply_delete_vector_(uint64_t id);
    void assert_invariants_() const;

    friend class AccumulatorWal;
};

} // namespace sketch2
