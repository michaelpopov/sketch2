// Declares the compute-engine-backed top-k scanner API.

#pragma once
#include "core/compute/compute_engine.h"
#include "core/compute/dist_item.h"
#include "core/utils/bitset_filter.h"
#include "utils/shared_types.h"
#include <cstdint>
#include <vector>

namespace sketch2 {

class DatasetReader;

// Scanner provides the same top-k search as the legacy scanner but delegates metric
// score computation to the compute engine layer instead of the
// Singleton-dispatched hand-written SIMD kernels.
class Scanner {
public:
    explicit Scanner(ComputeEngine engine = compiled_compute_engine());

    Ret find(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) const;

    Ret find_items(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result, const BitsetFilter* bitset = nullptr) const;

private:
    ComputeEngine engine_;
};

} // namespace sketch2
