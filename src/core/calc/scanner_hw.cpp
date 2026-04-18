// Implements the Highway-backed scanner.

#include "core/calc/scanner_hw.h"

#include "core/calc/calc_engine.h"
#include "core/calc/hwy_kernels.h"
#include "core/calc/scanner_engine_common.h"

#include <exception>

namespace sketch2 {

namespace {

Ret find_items_hw(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>* result, const BitsetFilter* bitset) {
    if (vec == nullptr || count == 0 || result == nullptr) {
        return Ret("ScannerEx::find: invalid arguments.");
    }

    result->clear();
    const DistFunc func = dataset.dist_func();
    const size_t dim = dataset.dim();
    const CalcKernels kernels = resolve_hwy_kernels(func, dataset.type());

    DistHeap heap(DistItemCompare{func});
    heap.reserve(count);
    Timer timer("scanner_hw::query");

    if (func == DistFunc::COS) {
        assert(kernels.squared_norm && kernels.dot && kernels.dist_with_query_norm);
        const double query_norm_sq = kernels.squared_norm(vec, dim);
        const double query_inv = query_inverse_norm(query_norm_sq);
        const FnPtrInvNormScore stored_norm_score{kernels.dot, vec, dim, query_inv};
        const FnPtrQueryNormScore fallback_score{kernels.dist_with_query_norm, vec, dim, query_norm_sq};
        CHECK(build_dataset_heap_with_optional_stored_norms(
            dataset, count, stored_norm_score, fallback_score, func, &heap, bitset));
    } else if (func == DistFunc::L2 && kernels.squared_norm && kernels.dot) {
        assert(kernels.dist);
        const double query_norm_sq = kernels.squared_norm(vec, dim);
        const FnPtrStoredSquaredNormL2Score stored_norm_score{kernels.dot, vec, dim, query_norm_sq};
        const FnPtrDistScore fallback_score{kernels.dist, vec, dim};
        CHECK(build_dataset_heap_with_optional_stored_norms(
            dataset, count, stored_norm_score, fallback_score, func, &heap, bitset));
    } else {
        assert(kernels.dist);
        const FnPtrDistScore score{kernels.dist, vec, dim};
        CHECK(build_dataset_heap_with_score(dataset, count, score, func, &heap, bitset));
    }

    log_query(dataset.name(), func, dataset.type(), dim, count, CalcEngine::highway, timer.elapsed_ms());
    extract_items(&heap, result);
    return Ret(0);
}

} // namespace

Ret ScannerHw::find(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) const {
    try {
        std::vector<DistItem> items;
        CHECK(find_items_hw(dataset, count, vec, &items, nullptr));
        extract_ids_from_items(items, &result);
        return Ret(0);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret ScannerHw::find_items(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result, const BitsetFilter* bitset) const {
    try {
        return find_items_hw(dataset, count, vec, &result, bitset);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

} // namespace sketch2
