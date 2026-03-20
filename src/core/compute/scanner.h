// Declares the high-level nearest-neighbor scanner API.

#pragma once
#include "utils/shared_types.h"
#include "core/compute/compute.h"
#include <cstdint>
#include <string>
#include <vector>

namespace sketch2 {

class DataReader;
class DatasetReader;

struct BitsetFilter {
    const uint8_t* data;
    const uint64_t size;
};

// Scanner exists to turn raw distance kernels into high-level top-k search over
// readers and datasets. It handles heap-based ranking, dispatches to the right
// metric backend, and merges persisted data with pending accumulator state.
class Scanner {
public:
    // Deprecated compatibility wrapper. Prefer find_items(...) and map ids from DistItem.
    // Returns up to count vector ids nearest to vec, ordered by distance.
    // func selects the distance function.
    // vec must match the type and dimension of the file.
    Ret find(const DataReader& reader, DistFunc func, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) const;

    // Deprecated compatibility wrapper. Prefer find_items(...) and map ids from DistItem.
    // Uses the distance function configured in dataset metadata.
    Ret find(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) const;

    // Returns up to count nearest items ordered by (distance, id).
    Ret find_items(const DataReader& reader, DistFunc func, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result) const;

    // Uses the distance function configured in dataset metadata.
    Ret find_items(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result, const BitsetFilter* bitset = nullptr) const;

private:
    Ret find_items_(const DataReader& reader, DistFunc func, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result) const;
    Ret find_items_(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result, const BitsetFilter* bitset) const;
};

} // namespace sketch2
