// Implements the ScannerEx facade over engine-specific scanner backends.

#include "core/compute/scanner_ex.h"

#include "core/compute/scanner_heap_utils.h"

#if SKETCH_CALC_ENGINE_HIGHWAY
#include "core/compute/scanner_hw.h"
#elif SKETCH_CALC_ENGINE_NUMKONG
#include "core/compute/scanner_nk.h"
#endif

#include <exception>

namespace sketch2 {

const char* calc_engine_name(CalcEngine engine) {
    switch (engine) {
        case CalcEngine::highway: return "highway";
        case CalcEngine::numkong: return "numkong";
        default: return "unknown";
    }
}

ScannerEx::ScannerEx(CalcEngine engine) : engine_(engine) {}

Ret ScannerEx::find(const DatasetReader& dataset, size_t count, const uint8_t* vec,
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

Ret ScannerEx::find_items(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result, const BitsetFilter* bitset) const {
    result.clear();
    try {
        switch (engine_) {
#if SKETCH_CALC_ENGINE_HIGHWAY
            case CalcEngine::highway:
                return find_items_hw(dataset, count, vec, &result, bitset);
#endif
#if SKETCH_CALC_ENGINE_NUMKONG
            case CalcEngine::numkong:
                return find_items_nk(dataset, count, vec, &result, bitset);
#endif
            default:
                return Ret("ScannerEx::find_items: unsupported calc engine.");
        }
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

} // namespace sketch2
