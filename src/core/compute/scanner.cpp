#include "scanner.h"
#include "core/compute/compute_cos.h"
#include "core/compute/compute_l1.h"
#include "core/compute/compute_l2.h"
#include "core/storage/data_reader.h"
#include "core/storage/dataset.h"
#include <algorithm>
#include <queue>
#include <stdexcept>

namespace sketch2 {

namespace {

using DistHeap = std::priority_queue<DistItem, std::vector<DistItem>, DistItem::Compare>;

#if defined(__AVX2__)
using ComputeL1Target = ComputeL1_AVX2;
using ComputeL2Target = ComputeL2_AVX2;
using ComputeCosTarget = ComputeCos_AVX2;
#elif defined(__aarch64__)
using ComputeL1Target = ComputeL1_Neon;
using ComputeL2Target = ComputeL2_Neon;
using ComputeCosTarget = ComputeCos_Neon;
#else
using ComputeL1Target = ComputeL1;
using ComputeL2Target = ComputeL2;
using ComputeCosTarget = ComputeCos;
#endif

void push_result(DistHeap* heap, size_t count, uint64_t id, double dist) {
    const DistItem item{id, dist};
    DistItem::Compare is_better;
    if (heap->size() < count) {
        heap->push(item);
    } else if (is_better(item, heap->top())) {
        heap->pop();
        heap->push(item);
    }
}

void extract_items(DistHeap* heap, std::vector<DistItem>* result) {
    result->resize(heap->size());
    for (size_t i = heap->size(); i-- > 0;) {
        (*result)[i] = heap->top();
        heap->pop();
    }
}

void extract_ids(const std::vector<DistItem>& items, std::vector<uint64_t>* result) {
    result->resize(items.size());
    for (size_t i = 0; i < items.size(); ++i) {
        (*result)[i] = items[i].id;
    }
}

template <typename Iterator>
inline void maybe_prefetch_next(const Iterator& it) {
#if defined(__GNUC__) || defined(__clang__)
    // Sequential scanner walks are a plausible place for software prefetch, but the
    // payoff is workload- and CPU-specific. Keep this conservative and re-benchmark
    // the distance on representative large datasets before tuning or expanding it.
    Iterator next_it = it;
    next_it.next();
    if (!next_it.eof()) {
        __builtin_prefetch(next_it.data(), 0, 1);
    }
#else
    (void)it;
#endif
}

template <auto DistanceFn, typename Iterator>
void scan_ordered_reader(Iterator it, const std::vector<uint64_t>& skip_ids,
        const uint8_t* vec, size_t dim, size_t count, DistHeap* heap) {
    if (skip_ids.empty()) {
        for (; !it.eof(); it.next()) {
            maybe_prefetch_next(it);
            push_result(heap, count, it.id(), DistanceFn(it.data(), vec, dim));
        }
        return;
    }

    auto skip_it = skip_ids.end();
    for (; !it.eof(); it.next()) {
        maybe_prefetch_next(it);
        const uint64_t id = it.id();
        if (skip_it == skip_ids.end()) {
            skip_it = std::lower_bound(skip_ids.begin(), skip_ids.end(), id);
        }
        while (skip_it != skip_ids.end() && *skip_it < id) {
            ++skip_it;
        }
        if (skip_it != skip_ids.end() && *skip_it == id) {
            continue;
        }
        push_result(heap, count, id, DistanceFn(it.data(), vec, dim));
    }
}

template <auto DistanceFn>
Ret scan_dataset_items(const Dataset& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result) {
    DistHeap heap;
    CHECK(dataset.prepare_read_state());
    const std::vector<uint64_t> modified_ids = dataset.accumulator_modified_ids();
    const size_t dim = dataset.dim();

    auto drs = dataset.reader();
    while (true) {
        auto [reader, ret] = drs->next();
        CHECK(ret);
        if (!reader) break;

        scan_ordered_reader<DistanceFn>(reader->base_begin(), modified_ids, vec, dim, count, &heap);
        scan_ordered_reader<DistanceFn>(reader->delta_begin(), modified_ids, vec, dim, count, &heap);
    }

    for (auto it = dataset.accumulator_begin(); !it.eof(); it.next()) {
        push_result(&heap, count, it.id(), DistanceFn(it.data(), vec, dim));
    }

    extract_items(&heap, &result);
    return Ret(0);
}

template <auto DistanceFn>
Ret scan_reader_items(const DataReader& reader, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result) {
    const size_t dim = reader.dim();
    DistHeap heap;

    for (auto it = reader.begin(); !it.eof(); it.next()) {
        push_result(&heap, count, it.id(), DistanceFn(it.data(), vec, dim));
    }

    extract_items(&heap, &result);
    return Ret(0);
}

template <typename ComputeTarget>
Ret dispatch_dataset_items(DataType type, const Dataset& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result) {
    switch (type) {
        case DataType::f32: return scan_dataset_items<&ComputeTarget::dist_f32>(dataset, count, vec, result);
        case DataType::f16: return scan_dataset_items<&ComputeTarget::dist_f16>(dataset, count, vec, result);
        case DataType::i16: return scan_dataset_items<&ComputeTarget::dist_i16>(dataset, count, vec, result);
        default: return Ret("Scanner::find: unsupported data type.");
    }
}

template <typename ComputeTarget>
Ret dispatch_reader_items(DataType type, const DataReader& reader, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result) {
    switch (type) {
        case DataType::f32: return scan_reader_items<&ComputeTarget::dist_f32>(reader, count, vec, result);
        case DataType::f16: return scan_reader_items<&ComputeTarget::dist_f16>(reader, count, vec, result);
        case DataType::i16: return scan_reader_items<&ComputeTarget::dist_i16>(reader, count, vec, result);
        default: return Ret("Scanner::find: unsupported data type.");
    }
}

} // namespace

Ret Scanner::find(const DataReader& reader, DistFunc func, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) const {
    try {
        return find_(reader, func, count, vec, result);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret Scanner::find_items(const DataReader& reader, DistFunc func, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result) const {
    try {
        return find_items_(reader, func, count, vec, result);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret Scanner::find(const Dataset& dataset, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) const {
    try {
        return find_(dataset, count, vec, result);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret Scanner::find_items(const Dataset& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result) const {
    try {
        return find_items_(dataset, count, vec, result);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret Scanner::find_(const Dataset& dataset, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) const {
    std::vector<DistItem> items;
    CHECK(find_items_(dataset, count, vec, items));
    extract_ids(items, &result);
    return Ret(0);
}

Ret Scanner::find_items_(const Dataset& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result) const {
    if (vec == nullptr || count == 0) {
        return Ret("Scanner::find: invalid arguments.");
    }

    result.clear();
    const DistFunc func = dataset.dist_func();
    const DataType type = dataset.type();
    switch (func) {
        case DistFunc::L1: return dispatch_dataset_items<ComputeL1Target>(type, dataset, count, vec, result);
        case DistFunc::L2: return dispatch_dataset_items<ComputeL2Target>(type, dataset, count, vec, result);
        case DistFunc::COS: return dispatch_dataset_items<ComputeCosTarget>(type, dataset, count, vec, result);
        default: return Ret("Scanner::find: unsupported distance function.");
    }
}

Ret Scanner::find_(const DataReader& reader, DistFunc func, size_t count, const uint8_t* vec,
    std::vector<uint64_t>& result) const {
    std::vector<DistItem> items;
    CHECK(find_items_(reader, func, count, vec, items));
    extract_ids(items, &result);
    return Ret(0);
}

Ret Scanner::find_items_(const DataReader& reader, DistFunc func, size_t count, const uint8_t* vec,
    std::vector<DistItem>& result) const {
    if (vec == nullptr || count == 0) {
        return Ret("Scanner::find: invalid arguments.");
    }

    result.clear();
    const DataType type = reader.type();
    switch (func) {
        case DistFunc::L1: return dispatch_reader_items<ComputeL1Target>(type, reader, count, vec, result);
        case DistFunc::L2: return dispatch_reader_items<ComputeL2Target>(type, reader, count, vec, result);
        case DistFunc::COS: return dispatch_reader_items<ComputeCosTarget>(type, reader, count, vec, result);
        default: return Ret("Scanner::find: not implemented");
    }
}

} // namespace sketch2
