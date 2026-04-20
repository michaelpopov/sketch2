// Declares the Highway-backed scan entry point and kernel resolver.

#pragma once

#include "core/compute/compute_engine.h"
#include "core/compute/dist_item.h"
#include "core/utils/bitset_filter.h"

#include <cstdint>
#include <vector>

namespace sketch2 {

class DatasetReader;

Ret find_items_hw(const DatasetReader& dataset, size_t count, const uint8_t* vec,
    std::vector<DistItem>* result, const BitsetFilter* bitset = nullptr, uint64_t query_id = 0);

// Runtime kernel resolver kept for benchmarks and kernel-focused tests.
ComputeKernels resolve_hwy_kernels(DistFunc func, DataType type);

} // namespace sketch2
