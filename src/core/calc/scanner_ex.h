// Declares the calc-engine-backed nearest-neighbor scanner API.

#pragma once
#include "core/calc/calc_engine.h"
#include "core/compute/compute.h"
#include "core/utils/bitset_filter.h"
#include "utils/shared_types.h"
#include <cstdint>
#include <vector>

namespace sketch2 {

class DatasetReader;

const char* calc_engine_name(CalcEngine engine);

// ScannerEx provides the same top-k nearest-neighbor search as Scanner but
// delegates distance computation to the calc engine layer instead of the
// Singleton-dispatched hand-written SIMD kernels.
class ScannerEx {
public:
    explicit ScannerEx(CalcEngine engine = CalcEngine::highway);

    Ret find(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) const;

    Ret find_items(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result, const BitsetFilter* bitset = nullptr) const;

private:
    CalcEngine engine_;

    Ret find_items_(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result, const BitsetFilter* bitset) const;
};

} // namespace sketch2
