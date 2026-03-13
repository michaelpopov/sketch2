// Declares the text input reader and subrange view types.

#pragma once
#include "utils/shared_types.h"
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace sketch2 {

struct LineInfo {
    uint64_t id;
    uint64_t offset; // byte offset where vector data starts in the mapped file
    uint64_t end;    // byte offset of closing ']' for this vector
};

// InputReader exists to parse the textual import format used by tests and bulk
// loading. It indexes ids and vector payload spans in a memory-mapped file so
// later range checks and record reads can avoid reparsing the whole input.
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
    bool is_no_data(size_t index) const;

    bool is_range_present(uint64_t start_range, uint64_t end_range) const;

private:
    friend class InputReaderView;

    const uint8_t*        map_     = nullptr;
    size_t                map_len_ = 0;
    DataType              type_    = DataType::f32;
    size_t                dim_     = 0;
    std::vector<LineInfo> lines_;

    Ret init_(const std::string &path);
    std::pair<size_t, size_t> find_index_range(uint64_t start, uint64_t end) const;
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
    bool is_no_data(size_t index) const;
private:
    const InputReader& reader_;
    size_t view_index_;
    size_t count_;
};

} // namespace sketch2
