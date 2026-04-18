// Implements the NumKong-backed scanner.

#include "core/calc/scanner_nk.h"

#include "core/calc/calc_engine.h"
#include "core/calc/nk_kernels.h"
#include "core/calc/scanner_engine_common.h"

#include <exception>

namespace sketch2 {

Ret ScannerNk::find(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) const {
    try {
        std::vector<DistItem> items;
        CHECK(find_items(dataset, count, vec, items, nullptr));
        extract_ids_from_items(items, &result);
        return Ret(0);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret ScannerNk::find_items(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result, const BitsetFilter* bitset) const {
    try {
        if (dataset.type() == DataType::i16) {
            return Ret("ScannerNk::find_items: NumKong does not support i16 datasets.");
        }
        const CalcKernels kernels = resolve_nk_kernels(dataset.dist_func(), dataset.type());
        return scanner_find_items_with_kernels(
            CalcEngine::numkong, dataset, count, vec, kernels, &result, bitset);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

} // namespace sketch2
