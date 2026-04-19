// Declares the binary data-file reader and its iterators.

#pragma once
#include "utils/shared_types.h"
#include "core/utils/dynamic_bitset.h"
#include "core/storage/compact_ids.h"
#include "core/utils/mapped_region.h"
#include "core/storage/data_file.h"
#include "core/storage/data_file_layout.h"
#include <cassert>
#include <cstring>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace sketch2 {

// DataReader exists to expose persisted storage files as fast queryable views.
// It memory-maps the binary file layout, optionally layers a delta over a base
// file, and provides iterators and point lookups over the visible rows.
class DataReader {
public:
    // Iterator produces visible base rows first and attached delta rows second,
    // so ids may be non-monotonic when a delta is present.
    // Iterator exists to walk the visible rows of a reader, including attached
    // delta rows after the base rows.
    class Iterator {
    public:
        void           next();
        bool           eof()  const;
        const uint8_t* data() const;
        float          get_norm() const;
        uint64_t       id()   const;

    private:
        friend class DataReader;
        Iterator(const DataReader* reader, const DataReader* delta_reader, size_t index);

        const DataReader*  reader_ = nullptr;
        const DataReader*  delta_reader_ = nullptr;
        size_t             index_  = 0;
        const size_t       count_;
    };

    // Iterates visible rows from the base data file only, ordered by id.
    // OrderedIterator exists to scan either the base side or the delta side in
    // sorted-id order without interleaving the two streams.
    class OrderedIterator {
    public:
        inline void next() {
            ++index_;
            if (!reader_ || source_ != Source::Base) {
                return;
            }
            while (index_ < reader_->count() && reader_->is_hidden(index_)) {
                ++index_;
            }
        }

        inline bool eof() const {
            if (!reader_) {
                return true;
            }
            if (source_ == Source::Base) {
                return index_ >= reader_->count();
            }
            return !reader_->delta_ || index_ >= reader_->delta_->count();
        }

        inline const uint8_t* data() const {
            if (eof()) {
                throw std::out_of_range("DataReader::OrderedIterator::data: index out of range");
            }

            if (source_ == Source::Base) {
                return reader_->at_unchecked_(index_);
            }

            return reader_->delta_->at(index_);
        }

        inline float get_norm() const {
            if (eof()) {
                throw std::out_of_range("DataReader::OrderedIterator::get_norm: index out of range");
            }

            if (source_ == Source::Base) {
                return reader_->get_norm(index_);
            }

            return reader_->delta_->get_norm(index_);
        }

        inline uint64_t id() const {
            if (eof()) {
                throw std::out_of_range("DataReader::OrderedIterator::id: index out of range");
            }

            if (source_ == Source::Base) {
                return reader_->ids_.id(index_);
            }

            return reader_->delta_->ids_.id(index_);
        }

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

    ~DataReader() = default;

    Ret init(const std::string& path, std::unique_ptr<DataReader> delta = nullptr);

    DataType type() const;
    size_t dim() const;
    size_t size() const;  // size of one vector in bytes
    size_t stride() const { return stride_; } // distance between persisted vector records in bytes
    inline size_t count() const { // number of vectors
        if (!initialized_) {
            throw std::runtime_error("DataReader::count: reader is not initialized");
        }
        return static_cast<size_t>(hdr_.count);
    }
    uint32_t norm_flags() const;
    bool has_norms() const;
    bool has_matching_stored_norms(DistFunc dist_func) const;

    Iterator        begin() const;
    OrderedIterator base_begin() const;
    OrderedIterator delta_begin() const;
    uint64_t       id(size_t index) const;
    inline float   get_norm_from_record_unchecked(const uint8_t* record) const {
        assert(initialized_);
        assert(record != nullptr);
        assert(data_file_has_norms(hdr_));

        float value = 0.0f;
        std::memcpy(&value, record + norm_offset_in_record_, sizeof(value));
        return value;
    }
    inline float   get_norm(size_t index) const {
        if (index >= count()) {
            throw std::out_of_range("DataReader::get_norm: index out of range");
        }
        if (!data_file_has_norms(hdr_)) {
            throw std::logic_error("DataReader::get_norm: inline norms are absent");
        }
        if (is_hidden(index)) {
            throw std::logic_error("DataReader::get_norm: index points to hidden row");
        }
        return get_norm_from_record_unchecked(vectors_region_.data() + index * stride_);
    }
    const uint8_t* get(uint64_t id) const;   // lookup by vector id
    inline const uint8_t* at(size_t index) const {   // lookup by position; might return nullptr if the vector is deleted
        if (index >= count()) {
            throw std::out_of_range("DataReader::at: index out of range");
        }

        if (index < changed_bitset_.size() && changed_bitset_.get(index)) {
            return nullptr;
        }

        return vectors_region_.data() + index * stride_;
    }
    inline bool    is_hidden(size_t index) const {
        return (index < changed_bitset_.size() && changed_bitset_.get(index));
    }
    std::string    path() const { return path_; }

    size_t deleted_count() const { return deleted_ids_.count(); }
    uint64_t deleted_id(size_t index) const;

    bool check_consistency() const;
    bool has_delta() const { return delta_ != nullptr; }

private:
    inline const uint8_t* at_unchecked_(size_t index) const {
        if (index >= count()) {
            throw std::out_of_range("DataReader::at_unchecked_: index out of range");
        }
        return vectors_region_.data() + index * stride_;
    }

    MappedRegion             vectors_region_;
    MappedRegion             ids_region_;
    MappedRegion             deleted_ids_region_;
    DataFileHeader           hdr_     = {};
    bool                     initialized_ = false;
    CompactIds               ids_;
    CompactIds               deleted_ids_;
    DataType                 type_    = DataType::f32;
    size_t                   vector_size_ = 0;    // size of one vector in bytes
    size_t                   norm_offset_in_record_ = 0;    // inline norm slot within one persisted record
    size_t                   stride_  = 0;        // bytes between persisted vectors
    std::string              path_ = "<undefined>";

    DynamicBitset           changed_bitset_;
    std::unique_ptr<DataReader> delta_;

    Ret init_(const std::string &path, std::unique_ptr<DataReader> delta);
    void reset_state_();
    Ret read_header_(int fd, const std::string& path, size_t* file_size);
    Ret validate_header_and_layout_(size_t file_size, DataMetadataLayout* metadata_layout);
    Ret validate_delta_(const std::unique_ptr<DataReader>& delta) const;
    Ret map_regions_(int fd, size_t file_size, const DataMetadataLayout& metadata_layout);
    Ret init_delta();
    void assert_invariants_() const;
};

} // namespace sketch2
