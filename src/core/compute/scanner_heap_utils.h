// Shared scanner heap/result utilities.

#pragma once

#include "core/compute/dist_item.h"
#include "core/utils/high_perf_heap.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace sketch2 {

using DistHeap = HighPerfHeap<DistItem, DistItemCompare>;
using LocalDistHeap = HighPerfHeap<LocalDistItem, LocalDistItemCompare>;

template <typename Heap, typename Item>
inline bool push_bounded_result(Heap* heap, size_t count, Item&& item) {
    return heap->push_or_replace_top(std::forward<Item>(item), count);
}

// Result ordering is defined by the heap comparator; these helpers only apply
// the bounded top-k insert/replace policy.
inline void push_result(DistHeap* heap, size_t count, uint64_t id, double score) {
    static_cast<void>(push_bounded_result(heap, count, DistItem{id, score}));
}

inline void push_result_local(LocalDistHeap* heap, size_t count, uint32_t id, double score) {
    static_cast<void>(push_bounded_result(heap, count, LocalDistItem{id, static_cast<float>(score)}));
}

inline bool push_result_local_changed(LocalDistHeap* heap, size_t count, uint32_t id, double score) {
    return push_bounded_result(heap, count, LocalDistItem{id, static_cast<float>(score)});
}

inline void extract_items(DistHeap* heap, std::vector<DistItem>* result) {
    result->resize(heap->size());
    for (size_t i = heap->size(); i-- > 0;) {
        (*result)[i] = heap->pop_top();
    }
}

inline void extract_local_items(LocalDistHeap* heap, std::vector<LocalDistItem>* result) {
    result->resize(heap->size());
    for (size_t i = heap->size(); i-- > 0;) {
        (*result)[i] = heap->pop_top();
    }
}

// Convert a per-reader local heap directly into the result vector, sorted
// best-first. Used by the single-reader fast path to skip the final-heap
// construction and its subsequent O(n log n) drain.
inline void sort_local_heap_into_result(
        LocalDistHeap* local_heap, uint64_t heap_base_id, DistFunc func,
        std::vector<DistItem>* result) {
    const auto& items = local_heap->data();
    result->resize(items.size());
    for (size_t i = 0; i < items.size(); ++i) {
        (*result)[i] = DistItem{items[i].id + heap_base_id, items[i].score};
    }
    const bool smaller_score_better = smaller_score_is_better(func);
    std::sort(result->begin(), result->end(),
        [smaller_score_better](const DistItem& a, const DistItem& b) {
            return dist_item_is_better(smaller_score_better, a, b);
        });
}

inline void extract_ids_from_items(const std::vector<DistItem>& items, std::vector<uint64_t>* result) {
    result->clear();
    result->reserve(items.size());
    for (const auto& item : items) {
        result->push_back(item.id);
    }
}

} // namespace sketch2
