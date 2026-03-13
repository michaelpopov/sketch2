#include "scanner.h"
#include "core/compute/compute_cos.h"
#include "core/compute/compute_l1.h"
#include "core/compute/compute_l2.h"
#include "core/storage/data_reader.h"
#include "core/storage/dataset.h"
#include <queue>
#include <stdexcept>

namespace sketch2 {

namespace {

using DistFn = double (*)(const uint8_t*, const uint8_t*, size_t);
using DistHeap = std::priority_queue<DistItem, std::vector<DistItem>, DistItem::Compare>;

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

    DistHeap heap;
    CHECK(dataset.prepare_read_state());
    const DistFunc func = dataset.dist_func();
    const DataType type = dataset.type();
    const size_t dim = dataset.dim();
    DistFn dist_fn = nullptr;
    switch (func) {
        case DistFunc::L1: dist_fn = ComputeL1::resolve_dist(type); break;
        case DistFunc::L2: dist_fn = ComputeL2::resolve_dist(type); break;
        case DistFunc::COS: dist_fn = ComputeCos::resolve_dist(type); break;
        default: return Ret("Scanner::find: unsupported distance function.");
    }

    auto drs = dataset.reader();
    while (true) {
        auto [reader, ret] = drs->next();
        CHECK(ret);
        if (!reader) break;

        for (auto it = reader->begin(); !it.eof(); it.next()) {
            if (dataset.is_modified_in_accumulator(it.id())) {
                continue;
            }
            double d = dist_fn(it.data(), vec, dim);
            push_result(&heap, count, it.id(), d);
        }
    }

    for (auto it = dataset.accumulator_begin(); !it.eof(); it.next()) {
        double d = dist_fn(it.data(), vec, dim);
        push_result(&heap, count, it.id(), d);
    }

    extract_items(&heap, &result);
    return Ret(0);
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

    DistFn dist_fn = nullptr;
    switch (func) {
        case DistFunc::L1: dist_fn = ComputeL1::resolve_dist(reader.type()); break;
        case DistFunc::L2: dist_fn = ComputeL2::resolve_dist(reader.type()); break;
        case DistFunc::COS: dist_fn = ComputeCos::resolve_dist(reader.type()); break;
        default: return Ret("Scanner::find: not implemented");
    }

    size_t   dim  = reader.dim();

    // Max-heap of DistItem capped at count — keeps the nearest seen so far.
    DistHeap heap;

    for (auto it = reader.begin(); !it.eof(); it.next()) {
        double d = dist_fn(it.data(), vec, dim);
        push_result(&heap, count, it.id(), d);
    }

    extract_items(&heap, &result);
    return Ret(0);
}

} // namespace sketch2
