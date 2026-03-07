#include "scanner.h"
#include "core/compute/compute_l1.h"
#include "core/compute/compute_l2.h"
#include "core/storage/data_reader.h"
#include "core/storage/dataset.h"
#include <queue>
#include <stdexcept>

namespace sketch2 {

namespace {

using DistFn = double (*)(const uint8_t*, const uint8_t*, size_t);

} // namespace

Ret Scanner::find(const DataReader& reader, DistFunc func, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) const {
    try {
        return find_(reader, func, count, vec, result);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret Scanner::find(const Dataset& dataset, DistFunc func, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) const {
    try {
        return find_(dataset, func, count, vec, result);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret Scanner::find_(const Dataset& dataset, DistFunc func, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) const {
    if (vec == nullptr || count == 0) {
        return Ret("Scanner::find: invalid arguments.");
    }
    if (func != DistFunc::L1 && func != DistFunc::L2) {
        return Ret("Scanner::find: unsupported distance function.");
    }

    result.clear();

    std::priority_queue<DistItem, std::vector<DistItem>, DistItem::Compare> heap;

    auto drs = dataset.reader();
    while (true) {
        auto [reader, ret] = drs->next();
        CHECK(ret);
        if (!reader) break;

        DataType type = reader->type();
        size_t   dim  = reader->dim();
        DistFn dist_fn = nullptr;
        switch (func) {
            case DistFunc::L1: dist_fn = ComputeL1::resolve_dist(type); break;
            case DistFunc::L2: dist_fn = ComputeL2::resolve_dist(type); break;
            default: return Ret("Scanner::find: unsupported distance function.");
        }

        for (auto it = reader->begin(); !it.eof(); it.next()) {
            double d = dist_fn(it.data(), vec, dim);
            if (heap.size() < count) {
                heap.push({it.id(), d});
            } else if (d < heap.top().dist) {
                heap.pop();
                heap.push({it.id(), d});
            }
        }
    }

    result.resize(heap.size());
    for (size_t i = heap.size(); i-- > 0;) {
        result[i] = heap.top().id;
        heap.pop();
    }

    return Ret(0);
}

Ret Scanner::find_(const DataReader& reader, DistFunc func, size_t count, const uint8_t* vec,
    std::vector<uint64_t>& result) const {
    if (vec == nullptr || count == 0) {
        return Ret("Scanner::find: invalid arguments.");
    }

    result.clear();

    DistFn dist_fn = nullptr;
    switch (func) {
        case DistFunc::L1: dist_fn = ComputeL1::resolve_dist(reader.type()); break;
        case DistFunc::L2: dist_fn = ComputeL2::resolve_dist(reader.type()); break;
        default: return Ret("Scanner::find: not implemented");
    }

    size_t   dim  = reader.dim();

    // Max-heap of DistItem capped at count — keeps the nearest seen so far.
    std::priority_queue<DistItem, std::vector<DistItem>, DistItem::Compare> heap;

    for (auto it = reader.begin(); !it.eof(); it.next()) {
        double d = dist_fn(it.data(), vec, dim);
        if (heap.size() < count) {
            heap.push({it.id(), d});
        } else if (d < heap.top().dist) {
            heap.pop();
            heap.push({it.id(), d});
        }
    }

    // Extract ids in ascending distance order.
    result.resize(heap.size());
    for (size_t i = heap.size(); i-- > 0;) {
        result[i] = heap.top().id;
        heap.pop();
    }

    return Ret(0);
}

} // namespace sketch2
