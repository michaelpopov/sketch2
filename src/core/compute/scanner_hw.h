// Declares the Highway-backed scanner implementation and kernel resolver.

#pragma once

#include "core/compute/compute_engine.h"
#include "core/compute/dist_item.h"
#include "core/utils/bitset_filter.h"

#include <cstdint>
#include <vector>

namespace sketch2 {

class DatasetReader;

class ScannerHw {
public:
    Ret find(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) const;

    Ret find_items(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result, const BitsetFilter* bitset = nullptr) const;
};

Ret find_items_hw(const DatasetReader& dataset, size_t count, const uint8_t* vec,
    std::vector<DistItem>* result, const BitsetFilter* bitset = nullptr);

CalcKernels resolve_hwy_kernels(DistFunc func, DataType type);

} // namespace sketch2
