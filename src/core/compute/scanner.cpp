// Implements the Scanner facade over the Highway compute backend.

#include "core/compute/scanner.h"

#include "core/compute/scanner_log_utils.h"
#include "core/compute/highway.h"

#include <exception>

namespace sketch2 {

Ret Scanner::find_items(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result, const BitsetFilter* bitset) const {
    result.clear();
    try {
        const uint64_t query_id = next_scanner_query_id();
        return find_items_hw(dataset, count, vec, &result, bitset, query_id);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

} // namespace sketch2
