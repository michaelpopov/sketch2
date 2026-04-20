// Declares the NumKong-backed scan entry point and kernel resolver.

#pragma once

#include "core/compute/compute_engine.h"
#include "core/compute/dist_item.h"
#include "core/utils/bitset_filter.h"

#include <cstdint>
#include <vector>

namespace sketch2 {

class DatasetReader;

Ret find_items_nk(const DatasetReader& dataset, size_t count, const uint8_t* vec,
    std::vector<DistItem>* result, const BitsetFilter* bitset = nullptr, uint64_t query_id = 0);

// Runtime kernel resolver kept for benchmarks and kernel-focused tests.
ComputeKernels resolve_nk_kernels(DistFunc func, DataType type);
bool nk_compute_uses_dynamic_dispatch();
uint64_t nk_compute_compiled_capabilities();
uint64_t nk_compute_available_capabilities();
const char* nk_compute_backend_name(DistFunc func, DataType type);
const char* nk_compute_backend_name_for_capabilities(DistFunc func, DataType type, uint64_t capabilities);

} // namespace sketch2
