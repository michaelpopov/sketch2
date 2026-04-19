// Implements the Scanner facade over engine-specific scanner backends.

#include "core/compute/scanner.h"

#include "core/compute/scanner_heap_utils.h"
#include "core/compute/scanner_log_utils.h"

#if SKETCH_COMPUTE_ENGINE_HIGHWAY
#include "core/compute/scanner_hw.h"
#elif SKETCH_COMPUTE_ENGINE_NUMKONG
#include "core/compute/scanner_nk.h"
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

Scanner::Scanner(ComputeEngine engine) : engine_(engine) {}

Ret Scanner::find(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) const {
    result.clear();
    try {
        std::vector<DistItem> items;
        CHECK(find_items(dataset, count, vec, items, nullptr));
        extract_ids_from_items(items, &result);
        return Ret(0);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret Scanner::find_items(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result, const BitsetFilter* bitset) const {
    result.clear();
    try {
        const uint64_t query_id = next_scanner_query_id();
        switch (engine_) {
#if SKETCH_COMPUTE_ENGINE_HIGHWAY
            case ComputeEngine::highway:
                return find_items_hw(dataset, count, vec, &result, bitset, query_id);
#endif
#if SKETCH_COMPUTE_ENGINE_NUMKONG
            case ComputeEngine::numkong:
                return find_items_nk(dataset, count, vec, &result, bitset, query_id);
#endif
            default:
                return Ret("Scanner::find_items: unsupported compute engine.");
        }
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

} // namespace sketch2
