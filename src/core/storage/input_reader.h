#pragma once
#include "utils/shared_types.h"
#include <cstdint>
#include <string>
#include <vector>

namespace sketch2 {

struct LineInfo {
    uint64_t id;
    uint64_t offset; // byte offset where vector data starts in the mapped file
};

class InputReader {
public:
    ~InputReader();

    Ret init(const std::string& path);

    size_t   count() const;
    DataType type()  const;
    size_t   dim()   const;
    size_t   size()  const; // sizeof(type) * dim

    uint64_t id(size_t index) const;
    const uint8_t* data(size_t index) const;
    bool is_no_data(size_t index) const;

private:
    const uint8_t*        map_     = nullptr;
    size_t                map_len_ = 0;
    DataType              type_    = DataType::f32;
    size_t                dim_     = 0;
    std::vector<LineInfo> lines_;
    mutable std::vector<uint8_t>  buf_;   // parsed vector data, laid out contiguously
};

} // namespace sketch2
