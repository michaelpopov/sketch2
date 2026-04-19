// Defines ranked-result types and ordering helpers shared by compute scanners.

#pragma once

#include "utils/shared_types.h"

#include <cassert>
#include <cstdint>

namespace sketch2 {

struct DistItem {
    uint64_t id;
    double score;
};

inline bool smaller_score_is_better(DistFunc func) {
    switch (func) {
        case DistFunc::DOT:
            return false;
        case DistFunc::L2:
        case DistFunc::COS:
            return true;
        default:
            assert(false && "smaller_score_is_better: unsupported distance function");
            // Fall back to ascending ordering to keep comparators noexcept-like.
            return true;
    }
}

inline bool dist_item_is_better(DistFunc func, const DistItem& a, const DistItem& b) {
    if (a.score != b.score) {
        return smaller_score_is_better(func) ? (a.score < b.score) : (a.score > b.score);
    }
    return a.id < b.id;
}

struct DistItemCompare {
    explicit DistItemCompare(DistFunc func_) : func(func_) {}

    DistFunc func;

    bool operator()(const DistItem& a, const DistItem& b) const {
        return dist_item_is_better(func, a, b);
    }
};

} // namespace sketch2
