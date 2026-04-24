// Defines ranked-result types and ordering helpers shared by compute scanners.

#pragma once

#include "utils/shared_types.h"

#include <cassert>
#include <cstdint>

namespace sketch2 {

// Per-reader top-k entry. Uses a 32-bit reader-local id (the reader base id is
// added when results are merged) and a float score to keep the hot scan heap
// at 8 bytes per item — twice the items per cache line as DistItem. Kernels
// compute scores in double; the truncation to float here means top-k order
// can differ from a full-precision implementation when neighbours' scores are
// within ~1e-7 of each other. sketch2 does not promise exact ordering.
struct LocalDistItem {
    uint32_t id;
    float score;
};

// Public top-k entry returned to callers. Carries the full 64-bit dataset id
// and a double score (promoted from the per-reader float).
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

inline bool dist_item_is_better(bool smaller_score_better, const DistItem& a, const DistItem& b) {
    if (a.score != b.score) {
        return smaller_score_better ? (a.score < b.score) : (a.score > b.score);
    }
    return a.id < b.id;
}

inline bool local_dist_item_is_better(bool smaller_score_better, const LocalDistItem& a, const LocalDistItem& b) {
    if (a.score != b.score) {
        return smaller_score_better ? (a.score < b.score) : (a.score > b.score);
    }
    return a.id < b.id;
}

struct DistItemCompare {
    explicit DistItemCompare(DistFunc func_) : smaller_score_better(smaller_score_is_better(func_)) {}

    bool smaller_score_better;

    bool operator()(const DistItem& a, const DistItem& b) const {
        return dist_item_is_better(smaller_score_better, a, b);
    }
};

struct LocalDistItemCompare {
    explicit LocalDistItemCompare(DistFunc func_) : smaller_score_better(smaller_score_is_better(func_)) {}

    bool smaller_score_better;

    bool operator()(const LocalDistItem& a, const LocalDistItem& b) const {
        return local_dist_item_is_better(smaller_score_better, a, b);
    }
};

} // namespace sketch2
