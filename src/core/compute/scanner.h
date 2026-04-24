// Declares the Highway-backed top-k scanner API.

#pragma once
#include "core/compute/dist_item.h"
#include "core/compute/highway.h"
#include "core/compute/scanner_log_utils.h"
#include "core/utils/bitset_filter.h"
#include "utils/shared_types.h"
#include <cstdint>
#include <exception>
#include <vector>

namespace sketch2 {

class DatasetReader;

// Scanner provides top-k search backed by the Highway compute kernels, with
// exception translation to Ret at the API boundary.
class Scanner {
public:
    Scanner() = default;

    Ret find_items(const DatasetReader& dataset, size_t count, const uint8_t* vec,
            std::vector<DistItem>& result, const BitsetFilter* bitset = nullptr) const {
        result.clear();
        try {
            const uint64_t query_id = next_scanner_query_id();
            return find_items_hw(dataset, count, vec, &result, bitset, query_id);
        } catch (const std::exception& ex) {
            return Ret(ex.what());
        }
    }
};

} // namespace sketch2
