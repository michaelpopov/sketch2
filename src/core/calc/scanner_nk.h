// Declares the NumKong-backed scanner implementation.

#pragma once

#include "core/calc/dist_item.h"
#include "core/utils/bitset_filter.h"

#include <cstdint>
#include <vector>

namespace sketch2 {

class DatasetReader;

class ScannerNk {
public:
    Ret find(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) const;

    Ret find_items(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result, const BitsetFilter* bitset = nullptr) const;
};

} // namespace sketch2
