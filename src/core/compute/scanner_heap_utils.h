// Shared scanner heap/result utilities.

#pragma once

#include "core/compute/dist_item.h"

#include <queue>
#include <vector>

namespace sketch2 {

class DistHeap : public std::priority_queue<DistItem, std::vector<DistItem>, DistItemCompare> {
public:
    using std::priority_queue<DistItem, std::vector<DistItem>, DistItemCompare>::priority_queue;

    const DistItemCompare& comparator() const {
        return this->comp;
    }

    void reserve(size_t capacity) {
        this->c.reserve(capacity);
    }
};

inline void push_result(DistHeap* heap, size_t count, uint64_t id, double score) {
    const DistItem item{id, score};
    if (heap->size() < count) {
        heap->push(item);
    } else if (heap->comparator()(item, heap->top())) {
        heap->pop();
        heap->push(item);
    }
}

inline void push_result_smaller_better(DistHeap* heap, size_t count, uint64_t id, double score) {
    const DistItem item{id, score};
    if (heap->size() < count) {
        heap->push(item);
        return;
    }

    const DistItem& top = heap->top();
    if (score < top.score || (score == top.score && id < top.id)) {
        heap->pop();
        heap->push(item);
    }
}

inline bool push_result_smaller_better_changed(DistHeap* heap, size_t count, uint64_t id, double score) {
    const DistItem item{id, score};
    if (heap->size() < count) {
        heap->push(item);
        return true;
    }

    const DistItem& top = heap->top();
    if (score < top.score || (score == top.score && id < top.id)) {
        heap->pop();
        heap->push(item);
        return true;
    }

    return false;
}

inline void push_result_larger_better(DistHeap* heap, size_t count, uint64_t id, double score) {
    const DistItem item{id, score};
    if (heap->size() < count) {
        heap->push(item);
        return;
    }

    const DistItem& top = heap->top();
    if (score > top.score || (score == top.score && id < top.id)) {
        heap->pop();
        heap->push(item);
    }
}

inline void extract_items(DistHeap* heap, std::vector<DistItem>* result) {
    result->resize(heap->size());
    for (size_t i = heap->size(); i-- > 0;) {
        (*result)[i] = heap->top();
        heap->pop();
    }
}

inline void extract_ids_from_items(const std::vector<DistItem>& items, std::vector<uint64_t>* result) {
    result->clear();
    result->reserve(items.size());
    for (const auto& item : items) {
        result->push_back(item.id);
    }
}

} // namespace sketch2
