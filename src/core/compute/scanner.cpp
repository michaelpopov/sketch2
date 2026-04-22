// Implements the Scanner facade over engine-specific scanner backends.

#include "core/compute/scanner.h"

#include "core/compute/scanner_log_utils.h"

#if SKETCH2_COMPUTE_ENGINE_HIGHWAY
#include "core/compute/highway.h"
#elif SKETCH2_COMPUTE_ENGINE_NUMKONG
#include "core/compute/numkong.h"
#endif

#include <exception>

namespace sketch2 {

const char* compute_engine_name(ComputeEngine engine) {
    switch (engine) {
        case ComputeEngine::highway: return "highway";
        case ComputeEngine::numkong: return "numkong";
        default: return "unknown";
    }
}

Ret Scanner::find_items(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result, const BitsetFilter* bitset) const {
    result.clear();
    try {
        const uint64_t query_id = next_scanner_query_id();
#if SKETCH2_COMPUTE_ENGINE_HIGHWAY
        return find_items_hw(dataset, count, vec, &result, bitset, query_id);
#endif
#if SKETCH2_COMPUTE_ENGINE_NUMKONG
        return find_items_nk(dataset, count, vec, &result, bitset, query_id);
#endif
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

} // namespace sketch2
