#include "scanner.h"
#include "core/compute/compute_l1.h"
#include <queue>
#include <stdexcept>

namespace sketch2 {

Ret Scanner::init(const std::string& path) {
    CHECK(reader_.init(path));
    is_initialized_ = true;
    return Ret(0);
}

std::vector<uint64_t> Scanner::find(DistFunc func, size_t count, const uint8_t* vec) const {
    if (!is_initialized_) {
        throw std::runtime_error("Scanner was not initialized.");
    }
    if (vec == nullptr || count == 0) {
        throw std::runtime_error("Scanner::find: invalid arguments.");
    }
    if (func != DistFunc::L1) {
        throw std::runtime_error("Scanner::find: unsupported distance function.");
    }

    ComputeL1 l1;
    ICompute* compute = nullptr;
    switch (func) {
        case DistFunc::L1: compute = &l1; break;
        default: break;
    }
    if (!compute || count == 0) {
        return {};
    }

    DataType type = reader_.type();
    size_t   dim  = reader_.dim();

    // Max-heap of DistItem capped at count — keeps the nearest seen so far.
    std::priority_queue<DistItem, std::vector<DistItem>, DistItem::Compare> heap;

    for (auto it = reader_.begin(); !it.eof(); it.next()) {
        double d = compute->dist(it.data(), vec, type, dim);
        if (heap.size() < count) {
            heap.push({it.id(), d});
        } else if (d < heap.top().dist) {
            heap.pop();
            heap.push({it.id(), d});
        }
    }

    // Extract ids in ascending distance order.
    std::vector<uint64_t> result(heap.size());
    for (size_t i = heap.size(); i-- > 0;) {
        result[i] = heap.top().id;
        heap.pop();
    }
    return result;
}

} // namespace sketch2
