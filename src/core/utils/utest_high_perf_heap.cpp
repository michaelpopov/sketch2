// Unit tests for the high performance heap utility.

#include "compute/dist_item.h"
#include "high_perf_heap.h"

#include <array>
#include <functional>
#include <vector>

#include <gtest/gtest.h>

namespace sketch2 {

TEST(high_perf_heap, push_and_pop_top_match_max_heap_order) {
    HighPerfHeap<int> heap;

    for (const int value : std::array<int, 7>{4, 1, 7, 3, 9, 2, 8}) {
        heap.push(value);
    }

    std::vector<int> values;
    while (!heap.empty()) {
        values.push_back(heap.pop_top());
    }

    EXPECT_EQ((std::vector<int>{9, 8, 7, 4, 3, 2, 1}), values);
}

TEST(high_perf_heap, range_constructor_heapifies_initial_values) {
    const std::array<int, 6> values{3, 11, 5, 7, 1, 9};

    HighPerfHeap<int> heap(values.begin(), values.end());

    EXPECT_EQ(heap.size(), values.size());
    EXPECT_EQ(heap.top(), 11);

    std::vector<int> ordered_values;
    while (!heap.empty()) {
        ordered_values.push_back(heap.pop_top());
    }

    EXPECT_EQ((std::vector<int>{11, 9, 7, 5, 3, 1}), ordered_values);
}

TEST(high_perf_heap, pop_removes_root_and_reheaps_remaining_values) {
    HighPerfHeap<int> heap;
    for (const int value : std::array<int, 5>{3, 10, 6, 7, 1}) {
        heap.push(value);
    }

    heap.pop();

    EXPECT_EQ(heap.top(), 7);

    std::vector<int> values;
    while (!heap.empty()) {
        values.push_back(heap.pop_top());
    }

    EXPECT_EQ((std::vector<int>{7, 6, 3, 1}), values);
}

TEST(high_perf_heap, replace_top_updates_root_with_single_reheap) {
    HighPerfHeap<int> heap;
    for (const int value : std::array<int, 5>{3, 10, 6, 7, 1}) {
        heap.push(value);
    }

    heap.replace_top(4);

    EXPECT_EQ(heap.top(), 7);

    std::vector<int> values;
    while (!heap.empty()) {
        values.push_back(heap.pop_top());
    }
    EXPECT_EQ((std::vector<int>{7, 6, 4, 3, 1}), values);
}

TEST(high_perf_heap, push_or_replace_top_keeps_smallest_k_with_default_compare) {
    HighPerfHeap<int> heap;

    for (const int value : std::array<int, 6>{8, 3, 6, 2, 5, 1}) {
        heap.push_or_replace_top(value, 3);
    }

    std::vector<int> values;
    while (!heap.empty()) {
        values.push_back(heap.pop_top());
    }

    EXPECT_EQ((std::vector<int>{3, 2, 1}), values);
}

TEST(high_perf_heap, push_or_replace_top_returns_false_when_max_size_is_zero) {
    HighPerfHeap<int> heap;

    EXPECT_FALSE(heap.push_or_replace_top(8, 0));
    EXPECT_TRUE(heap.empty());
}

TEST(high_perf_heap, push_or_replace_top_respects_custom_compare) {
    HighPerfHeap<int, std::greater<int>> heap;

    for (const int value : std::array<int, 6>{8, 3, 6, 2, 5, 9}) {
        heap.push_or_replace_top(value, 3);
    }

    std::vector<int> values;
    while (!heap.empty()) {
        values.push_back(heap.pop_top());
    }

    EXPECT_EQ((std::vector<int>{6, 8, 9}), values);
}

TEST(high_perf_heap, replace_top_if_returns_false_when_candidate_does_not_beat_root) {
    HighPerfHeap<int> heap;
    for (const int value : std::array<int, 4>{7, 4, 6, 2}) {
        heap.push(value);
    }

    EXPECT_FALSE(heap.replace_top_if(9));
    EXPECT_EQ(heap.top(), 7);
}

TEST(high_perf_heap, replace_top_if_returns_true_when_candidate_beats_root) {
    HighPerfHeap<int> heap;
    for (const int value : std::array<int, 4>{7, 4, 6, 2}) {
        heap.push(value);
    }

    EXPECT_TRUE(heap.replace_top_if(3));
    EXPECT_EQ(heap.top(), 6);

    std::vector<int> values;
    while (!heap.empty()) {
        values.push_back(heap.pop_top());
    }

    EXPECT_EQ((std::vector<int>{6, 4, 3, 2}), values);
}

TEST(high_perf_heap, emplace_and_clear_manage_heap_contents) {
    HighPerfHeap<int> heap;

    heap.emplace(5);
    heap.emplace(2);
    heap.emplace(9);

    EXPECT_EQ(heap.size(), 3U);
    EXPECT_EQ(heap.top(), 9);

    heap.clear();

    EXPECT_TRUE(heap.empty());
    EXPECT_EQ(heap.size(), 0U);
}

TEST(high_perf_heap, push_or_replace_top_honors_dist_item_id_tie_breaking) {
    HighPerfHeap<DistItem, DistItemCompare> heap(DistItemCompare{DistFunc::L2});

    EXPECT_TRUE(heap.push_or_replace_top(DistItem{30, 4.0}, 2));
    EXPECT_TRUE(heap.push_or_replace_top(DistItem{20, 4.0}, 2));
    EXPECT_TRUE(heap.push_or_replace_top(DistItem{10, 4.0}, 2));

    std::vector<DistItem> values;
    while (!heap.empty()) {
        values.push_back(heap.pop_top());
    }

    ASSERT_EQ(values.size(), 2U);
    EXPECT_EQ(values[0].id, 20U);
    EXPECT_EQ(values[0].score, 4.0);
    EXPECT_EQ(values[1].id, 10U);
    EXPECT_EQ(values[1].score, 4.0);
}

} // namespace sketch2
