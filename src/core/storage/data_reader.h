#pragma once
#include "utils/shared_types.h"
#include "core/utils/dynamic_bitset.h"
#include "core/storage/data_file.h"
#include <cstdint>
#include <memory>
#include <string>

namespace sketch2 {

class DataReader {
public:
    // Iterator produces visible base rows first and attached delta rows second,
    // so ids may be non-monotonic when a delta is present.
    class Iterator {
    public:
        void           next();
        bool           eof()  const;
        const uint8_t* data() const;
        float          cosine_inv_norm() const;
        uint64_t       id()   const;

    private:
        friend class DataReader;
        Iterator(const DataReader* reader, const DataReader* delta_reader, size_t index, const uint64_t* ids);

        const DataReader*  reader_ = nullptr;
        const DataReader*  delta_reader_ = nullptr;
        size_t             index_  = 0;
        const uint64_t*    ids_    = nullptr; // cached pointer to the id array
        const size_t       count_;
    };

    // Iterates visible rows from the base data file only, ordered by id.
    class OrderedIterator {
    public:
        void           next();
        bool           eof()  const;
        const uint8_t* data() const;
        float          cosine_inv_norm() const;
        uint64_t       id()   const;

    private:
        enum class Source {
            Base,
            Delta,
        };

        friend class DataReader;
        OrderedIterator(const DataReader* reader, Source source, size_t index)
            : reader_(reader), source_(source), index_(index) {}

        const DataReader* reader_ = nullptr;
        Source            source_ = Source::Base;
        size_t            index_  = 0;
    };

    ~DataReader();

    Ret init(const std::string& path, std::unique_ptr<DataReader> delta = nullptr);

    DataType type() const;
    size_t dim() const;
    size_t size() const;  // size of one vector in bytes
    size_t stride() const { return stride_; } // distance between persisted vector records in bytes
    size_t count() const; // number of vectors
    bool has_cosine_inv_norms() const;

    Iterator        begin() const;
    OrderedIterator base_begin() const;
    OrderedIterator delta_begin() const;
    uint64_t       id(size_t index) const;
    float          cosine_inv_norm(size_t index) const;
    const uint8_t* get(uint64_t id) const;   // lookup by vector id
    const uint8_t* at(size_t index) const;   // lookup by position; might return nullptr if the vector is deleted
    bool           is_hidden(size_t index) const;

    size_t deleted_count() const { return hdr_->deleted_count; }
    uint64_t deleted_id(size_t index) const;

    bool check_consistency() const;
    bool has_delta() const { return delta_ != nullptr; }

private:
    const uint8_t*           map_     = nullptr;
    size_t                   map_len_ = 0;
    const DataFileHeader*    hdr_     = nullptr;
    const uint64_t*          ids_     = nullptr; // cached pointer to the ids section
    const float*             cosine_inv_norms_ = nullptr; // cached pointer to optional cosine inverse norms
    const uint64_t*          deleted_ids_ = nullptr; // cached pointer to the deleted ids section
    DataType                 type_    = DataType::f32;
    size_t                   size_    = 0;        // size of one vector in bytes
    size_t                   stride_  = 0;        // bytes between persisted vectors

    DynamicBitset           bitset_;
    std::unique_ptr<DataReader> delta_;

    Ret init_(const std::string &path, std::unique_ptr<DataReader> delta);
    Ret init_delta();
    const uint8_t* get_by_pos(uint32_t pos) { return map_ + hdr_->data_offset + stride_ * pos; }
    void assert_invariants_() const;
};

} // namespace sketch2
