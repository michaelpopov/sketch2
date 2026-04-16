// Declares the binary data-file reader and its iterators.

#pragma once
#include "utils/shared_types.h"
#include "core/utils/dynamic_bitset.h"
#include "core/utils/compact_ids_ext.h"
#include "core/utils/mapped_region.h"
#include "core/storage/data_file.h"
#include "core/storage/data_file_layout.h"
#include <cstdint>
#include <memory>
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
        float          cosine_inv_norm() const;
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

    ~DataReader() = default;

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
    std::string    path() const { return path_; }

    size_t deleted_count() const { return deleted_ids_.count(); }
    uint64_t deleted_id(size_t index) const;

    bool check_consistency() const;
    bool has_delta() const { return delta_ != nullptr; }

private:
    MappedRegion             vectors_region_;
    MappedRegion             ids_region_;
    MappedRegion             deleted_ids_region_;
    MappedRegion             norms_region_;
    DataFileHeader           hdr_     = {};
    bool                     initialized_ = false;
    CompactIdsExt            ids_;
    CompactIdsExt            deleted_ids_;
    const float*             norms_   = nullptr; // optional cosine inverse norms in mapped metadata
    DataType                 type_    = DataType::f32;
    size_t                   vector_size_ = 0;    // size of one vector in bytes
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
