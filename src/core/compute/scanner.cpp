#include "scanner.h"
#include "core/compute/compute_l1.h"
#include <queue>
#include <utility>

namespace sketch2 {

Ret Scanner::init(const std::string& path) {
    return reader_.init(path);
}

std::vector<uint64_t> Scanner::find(DistFunc func, size_t count, const uint8_t* vec) const {
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

    // Max-heap of (distance, id) capped at count — keeps the nearest seen so far.
    using Entry = std::pair<double, uint64_t>;
    std::priority_queue<Entry> heap;

    for (auto it = reader_.begin(); !it.eof(); it.next()) {
        double d = compute->dist(it.data(), vec, type, dim);
        if (heap.size() < count) {
            heap.push({d, it.id()});
        } else if (d < heap.top().first) {
            heap.pop();
            heap.push({d, it.id()});
        }
    }

    // Extract ids in ascending distance order.
    std::vector<uint64_t> result(heap.size());
    for (size_t i = heap.size(); i-- > 0;) {
        result[i] = heap.top().second;
        heap.pop();
    }
    return result;
}

} // namespace sketch2
