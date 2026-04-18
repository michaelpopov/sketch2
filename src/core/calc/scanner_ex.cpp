// Implements the ScannerEx facade over engine-specific scanner backends.

#include "core/calc/scanner_ex.h"

#include "core/calc/scanner_hw.h"
#include "core/calc/scanner_nk.h"

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
    switch (engine_) {
        case CalcEngine::highway: {
            ScannerHw scanner;
            return scanner.find(dataset, count, vec, result);
        }
        case CalcEngine::numkong: {
            ScannerNk scanner;
            return scanner.find(dataset, count, vec, result);
        }
        default:
            return Ret("ScannerEx::find: unsupported calc engine.");
    }
}

Ret ScannerEx::find_items(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result, const BitsetFilter* bitset) const {
    switch (engine_) {
        case CalcEngine::highway: {
            ScannerHw scanner;
            return scanner.find_items(dataset, count, vec, result, bitset);
        }
        case CalcEngine::numkong: {
            ScannerNk scanner;
            return scanner.find_items(dataset, count, vec, result, bitset);
        }
        default:
            return Ret("ScannerEx::find_items: unsupported calc engine.");
    }
}

} // namespace sketch2
