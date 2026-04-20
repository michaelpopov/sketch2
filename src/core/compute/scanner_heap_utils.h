// Shared scanner heap/result utilities.

#pragma once

#include "core/compute/dist_item.h"
#include "core/utils/high_perf_heap.h"

#include <utility>
#include <vector>

namespace sketch2 {

using DistHeap = HighPerfHeap<DistItem, DistItemCompare>;
using DistHeapEx = HighPerfHeap<DistItemEx, DistItemExCompare>;

template <typename Heap, typename Item>
inline bool push_bounded_result(Heap* heap, size_t count, Item&& item) {
    return heap->push_or_replace_top(std::forward<Item>(item), count);
}

// Result ordering is defined by the heap comparator; these helpers only apply
// the bounded top-k insert/replace policy.
inline void push_result(DistHeap* heap, size_t count, uint64_t id, double score) {
    static_cast<void>(push_bounded_result(heap, count, DistItem{id, score}));
}

inline void push_result_local(DistHeapEx* heap, size_t count, uint32_t id, double score) {
    static_cast<void>(push_bounded_result(heap, count, DistItemEx{id, static_cast<float>(score)}));
}

inline bool push_result_local_changed(DistHeapEx* heap, size_t count, uint32_t id, double score) {
    return push_bounded_result(heap, count, DistItemEx{id, static_cast<float>(score)});
}

inline void extract_items(DistHeap* heap, std::vector<DistItem>* result) {
    result->resize(heap->size());
    for (size_t i = heap->size(); i-- > 0;) {
        (*result)[i] = heap->pop_top();
    }
}

inline void extract_items_ex(DistHeapEx* heap, std::vector<DistItemEx>* result) {
    result->resize(heap->size());
    for (size_t i = heap->size(); i-- > 0;) {
        (*result)[i] = heap->pop_top();
    }
}

inline void rebuild_final_heap_from_local_heap(
        DistHeapEx* local_heap, uint64_t heap_base_id, DistHeap* final_heap) {
    std::vector<DistItem> final_items;
    final_items.reserve(local_heap->size());
    for (const DistItemEx& item : local_heap->data()) {
        final_items.push_back(DistItem{item.id + heap_base_id, item.score});
    }
    final_heap->reset(std::move(final_items));
}

inline void extract_ids_from_items(const std::vector<DistItem>& items, std::vector<uint64_t>* result) {
    result->clear();
    result->reserve(items.size());
    for (const auto& item : items) {
        result->push_back(item.id);
    }
}

} // namespace sketch2
