// Implements the NumKong-backed scanner.

#include "core/calc/scanner_nk.h"

#include "core/calc/calc_engine.h"
#include "core/calc/nk_kernels.h"
#include "core/calc/scanner_engine_common.h"

#include <exception>

namespace sketch2 {

namespace {

Ret find_items_nk(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>* result, const BitsetFilter* bitset) {
    if (vec == nullptr || count == 0 || result == nullptr) {
        return Ret("ScannerEx::find: invalid arguments.");
    }
    if (dataset.type() == DataType::i16) {
        return Ret("ScannerNk::find_items: NumKong does not support i16 datasets.");
    }

    result->clear();
    const DistFunc func = dataset.dist_func();
    const size_t dim = dataset.dim();
    const CalcKernels kernels = resolve_nk_kernels(func, dataset.type());

    DistHeap heap(DistItemCompare{func});
    heap.reserve(count);
    Timer timer("scanner_nk::query");

    if (func == DistFunc::COS) {
        assert(kernels.squared_norm && kernels.dot && kernels.dist_with_query_norm);
        const double query_norm_sq = kernels.squared_norm(vec, dim);
        const double query_inv = query_inverse_norm(query_norm_sq);
        CHECK(scan_dataset_heap_with_optional_cosine_norms(
            dataset, count, &heap, kernels.dot, kernels.dist_with_query_norm,
            vec, dim, query_inv, query_norm_sq, func, bitset));
    } else if (func == DistFunc::L2 && kernels.squared_norm && kernels.dot) {
        assert(kernels.dist);
        const double query_norm_sq = kernels.squared_norm(vec, dim);
        CHECK(scan_dataset_heap_with_optional_l2_norms(
            dataset, count, &heap, kernels.dot, kernels.dist, vec, dim,
            query_norm_sq, func, bitset));
    } else {
        assert(kernels.dist);
        CHECK(scan_dataset_heap_with_dist(
            dataset, count, &heap, kernels.dist, vec, dim, func, bitset));
    }

    log_query(dataset.name(), func, dataset.type(), dim, count, CalcEngine::numkong, timer.elapsed_ms());
    extract_items(&heap, result);
    return Ret(0);
}

} // namespace

Ret ScannerNk::find(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) const {
    try {
        std::vector<DistItem> items;
        CHECK(find_items_nk(dataset, count, vec, &items, nullptr));
        extract_ids_from_items(items, &result);
        return Ret(0);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret ScannerNk::find_items(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result, const BitsetFilter* bitset) const {
    try {
        return find_items_nk(dataset, count, vec, &result, bitset);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

} // namespace sketch2
