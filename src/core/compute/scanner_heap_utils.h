// Shared scanner heap/result utilities.

#pragma once

#include "core/compute/dist_item.h"

#include <algorithm>
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

    std::vector<DistItem>& mutable_container() {
        return this->c;
    }
};

class DistHeapEx : public std::priority_queue<DistItemEx, std::vector<DistItemEx>, DistItemExCompare> {
public:
    using std::priority_queue<DistItemEx, std::vector<DistItemEx>, DistItemExCompare>::priority_queue;

    const DistItemExCompare& comparator() const {
        return this->comp;
    }

    void reserve(size_t capacity) {
        this->c.reserve(capacity);
    }

    std::vector<DistItemEx> release_container() {
        return std::move(this->c);
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

inline void push_result_local_smaller_better(DistHeapEx* heap, size_t count, uint32_t id, double score) {
    const DistItemEx item{id, static_cast<float>(score)};
    if (heap->size() < count) {
        heap->push(item);
        return;
    }

    const DistItemEx& top = heap->top();
    if (item.score < top.score || (item.score == top.score && id < top.id)) {
        heap->pop();
        heap->push(item);
    }
}

inline bool push_result_local_smaller_better_changed(
        DistHeapEx* heap, size_t count, uint32_t id, double score) {
    const DistItemEx item{id, static_cast<float>(score)};
    if (heap->size() < count) {
        heap->push(item);
        return true;
    }

    const DistItemEx& top = heap->top();
    if (item.score < top.score || (item.score == top.score && id < top.id)) {
        heap->pop();
        heap->push(item);
        return true;
    }

    return false;
}

inline void push_result_local_larger_better(DistHeapEx* heap, size_t count, uint32_t id, double score) {
    const DistItemEx item{id, static_cast<float>(score)};
    if (heap->size() < count) {
        heap->push(item);
        return;
    }

    const DistItemEx& top = heap->top();
    if (item.score > top.score || (item.score == top.score && id < top.id)) {
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

inline void extract_items_ex(DistHeapEx* heap, std::vector<DistItemEx>* result) {
    result->resize(heap->size());
    for (size_t i = heap->size(); i-- > 0;) {
        (*result)[i] = heap->top();
        heap->pop();
    }
}

inline void rebuild_final_heap_from_local_heap(
        DistHeapEx* local_heap, uint64_t heap_base_id, DistHeap* final_heap) {
    std::vector<DistItemEx> local_items = local_heap->release_container();
    std::vector<DistItem>& final_items = final_heap->mutable_container();
    final_items.clear();
    final_items.reserve(local_items.size());
    for (const DistItemEx& item : local_items) {
        final_items.push_back(DistItem{item.id + heap_base_id, item.score});
    }
    std::make_heap(final_items.begin(), final_items.end(), final_heap->comparator());
}

inline void extract_ids_from_items(const std::vector<DistItem>& items, std::vector<uint64_t>* result) {
    result->clear();
    result->reserve(items.size());
    for (const auto& item : items) {
        result->push_back(item.id);
    }
}

} // namespace sketch2
