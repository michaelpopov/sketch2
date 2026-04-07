// Declares the high-level top-k scanner API.

#pragma once
#include "core/compute/compute.h"
#include "core/utils/bitset_filter.h"
#include "utils/shared_types.h"
#include <cstdint>
#include <string>
#include <vector>

namespace sketch2 {

class DataReader;
class DatasetReader;

// Scanner exists to turn raw metric kernels into high-level top-k search over
// readers and datasets. It handles heap-based ranking, dispatches to the right
// metric backend, and merges persisted data with pending accumulator state.
class Scanner {
public:
    // Deprecated compatibility wrapper. Prefer find_items(...) and map ids from DistItem.
    // Uses the distance function configured in dataset metadata.
    Ret find(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) const;

    // Uses the distance function configured in dataset metadata.
    Ret find_items(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result, const BitsetFilter* bitset = nullptr) const;

private:
    Ret find_items_(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result, const BitsetFilter* bitset) const;
};

} // namespace sketch2
