// Declares the input reader and subrange view types for text and binary imports.

#pragma once
#include "utils/dynamic_bitset.h"
#include "utils/shared_types.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sketch2 {

class LinesInfo {
public:
    void clear();
    void reserve(size_t count);
    void set_u64_offsets(bool enabled);

    bool is_u64_offsets() const { return is_u64_offsets_; }
    size_t size() const { return ids_.size(); }
    bool empty() const { return ids_.empty(); }

    void add(uint64_t id, uint64_t offset);

    uint64_t id(size_t index) const;
    uint64_t offset(size_t index) const;

    void sort();
    size_t lower_bound_index(uint64_t value) const;
    size_t lower_bound_index(size_t first, uint64_t value) const;

private:
    void check_index(size_t index) const;

    std::vector<uint64_t> ids_;
    std::vector<uint32_t> offsets32_;
    std::vector<uint64_t> offsets64_;
    bool is_u64_offsets_ = false;
};

// InputReader exists to parse the text and binary import formats used by tests
// and bulk loading. It indexes ids and payload spans in a memory-mapped file so
// later range checks and record reads can avoid rescanning the whole input.
// After init() succeeds, concurrent read-only access is supported because the
// reader only serves immutable metadata and mmap-backed payload slices.
class InputReader {
public:
    ~InputReader();

    Ret init(const std::string& path);

    size_t   count() const;
    DataType type()  const;
    size_t   dim()   const;
    size_t   size()  const; // sizeof(type) * dim

    uint64_t id(size_t index) const;
    Ret data(size_t index, uint8_t* buf, size_t size) const;
    bool is_binary() const;
    // Reports whether text-mode payloads use comma separators. Callers that
    // borrow raw text slices must pass the matching parser mode downstream.
    bool is_comma_delimited() const;
    Ret raw_data(size_t index, const uint8_t** data) const;
    // Returns a borrowed [begin, end) slice into the mmap-backed text payload.
    // The returned pointers stay valid only while this InputReader remains
    // alive and mapped.
    Ret text_data_range(size_t index, const char** begin, const char** end) const;
    bool is_no_data(size_t index) const;

    bool is_range_present(uint64_t start_range, uint64_t end_range) const;

private:
    friend class InputReaderView;

    const uint8_t*        map_     = nullptr;
    size_t                map_len_ = 0;
    DataType              type_    = DataType::f32;
    size_t                dim_     = 0;
    bool                  binary_  = false;
    bool                  bit_indexed_ = false;
    LinesInfo             lines_;
    bool                  is_comma_delimited_ = true;

    Ret init_(const std::string &path);
    std::pair<size_t, size_t> find_index_range(uint64_t start, uint64_t end) const;
    Ret parse_input_header(const char* begin, const char* end);
    Ret process_text_data(const char* record_begin, const char* end);
    Ret process_binary_data(const char* record_begin, const char* end);
    Ret process_binary_indexed_data(const char* record_begin, const char* end);
    Ret find_text_vector_end(size_t index, const char** vec_end) const;
};

// InputReaderView exists to present a cheap subrange view over an InputReader
// so storage code can process one id-range slice at a time without copying data.
class InputReaderView {
public:
    InputReaderView(const InputReader& reader, uint64_t start, uint64_t end);

    size_t   count() const;
    DataType type()  const;
    size_t   dim()   const;
    size_t   size()  const; // sizeof(type) * dim

    uint64_t id(size_t index) const;
    Ret data(size_t index, uint8_t* buf, size_t size) const;
    bool is_binary() const;
    bool is_comma_delimited() const;
    Ret raw_data(size_t index, const uint8_t** data) const;
    // Returns a borrowed [begin, end) slice into the backing InputReader mmap.
    // The slice becomes invalid once that InputReader is destroyed or unmapped.
    Ret text_data_range(size_t index, const char** begin, const char** end) const;
    bool is_no_data(size_t index) const;
private:
    const InputReader& reader_;
    size_t view_index_;
    size_t count_;
};

} // namespace sketch2
