// Unit tests for RoaringIds.

#include "roaring_ids.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace sketch2 {
namespace {

char* aligned_32_data(std::vector<uint8_t>& buffer) {
    const uintptr_t raw = reinterpret_cast<uintptr_t>(buffer.data());
    const uintptr_t aligned = (raw + 31u) & ~uintptr_t{31u};
    return reinterpret_cast<char*>(aligned);
}

TEST(roaring_ids, add_and_access_ids_in_sorted_order) {
    RoaringIds ids;
    EXPECT_EQ(0, ids.init_writable(0).code());
    EXPECT_EQ(0, ids.add(30).code());
    EXPECT_EQ(0, ids.add(10).code());
    EXPECT_EQ(0, ids.add(20).code());
    EXPECT_EQ(0, ids.add(20).code());

    EXPECT_EQ(3u, ids.count());
    EXPECT_EQ(10u, ids.id(0));
    EXPECT_EQ(20u, ids.id(1));
    EXPECT_EQ(30u, ids.id(2));
    EXPECT_THROW(ids.id(3), std::out_of_range);
}

TEST(roaring_ids, lower_bound_index_matches_sorted_semantics) {
    RoaringIds ids;
    EXPECT_EQ(0, ids.init_writable(0).code());
    EXPECT_EQ(0, ids.add(10).code());
    EXPECT_EQ(0, ids.add(20).code());
    EXPECT_EQ(0, ids.add(40).code());
    EXPECT_EQ(0, ids.add(50).code());

    EXPECT_EQ(0u, ids.lower_bound_index(1));
    EXPECT_EQ(0u, ids.lower_bound_index(10));
    EXPECT_EQ(1u, ids.lower_bound_index(11));
    EXPECT_EQ(2u, ids.lower_bound_index(40));
    EXPECT_EQ(4u, ids.lower_bound_index(100));
}

TEST(roaring_ids, find_index_returns_false_for_empty_states) {
    RoaringIds ids;
    size_t index = 123u;

    EXPECT_FALSE(ids.find_index(10, &index));
    EXPECT_EQ(RoaringIds::npos, index);

    EXPECT_EQ(0, ids.init_writable(1000).code());
    index = 123u;
    EXPECT_FALSE(ids.find_index(1000, &index));
    EXPECT_EQ(RoaringIds::npos, index);
}

TEST(roaring_ids, find_index_matches_exact_membership) {
    RoaringIds ids;
    EXPECT_EQ(0, ids.init_writable(1000).code());
    EXPECT_EQ(0, ids.add(1000).code());
    EXPECT_EQ(0, ids.add(1010).code());
    EXPECT_EQ(0, ids.add(1020).code());

    size_t index = RoaringIds::npos;
    EXPECT_TRUE(ids.find_index(1000, &index));
    EXPECT_EQ(0u, index);
    EXPECT_TRUE(ids.find_index(1020, &index));
    EXPECT_EQ(2u, index);

    EXPECT_FALSE(ids.find_index(999, &index));
    EXPECT_EQ(RoaringIds::npos, index);
    EXPECT_FALSE(ids.find_index(1015, &index));
    EXPECT_EQ(RoaringIds::npos, index);
    EXPECT_FALSE(ids.find_index(1000ull + std::numeric_limits<uint32_t>::max() + 1ull, &index));
    EXPECT_EQ(RoaringIds::npos, index);
}

TEST(roaring_ids, contains_matches_base_adjusted_membership) {
    RoaringIds ids;
    EXPECT_FALSE(ids.contains(100));

    EXPECT_EQ(0, ids.init_writable(1000).code());
    EXPECT_EQ(0, ids.add(1000).code());
    EXPECT_EQ(0, ids.add(1005).code());

    EXPECT_FALSE(ids.contains(999));
    EXPECT_TRUE(ids.contains(1000));
    EXPECT_FALSE(ids.contains(1001));
    EXPECT_TRUE(ids.contains(1005));
    EXPECT_FALSE(ids.contains(1000ull + std::numeric_limits<uint32_t>::max() + 1ull));
}

TEST(roaring_ids, clear_and_empty_reset_container) {
    RoaringIds ids;
    EXPECT_TRUE(ids.empty());
    EXPECT_EQ(0, ids.init_writable(100).code());
    EXPECT_TRUE(ids.empty());
    EXPECT_EQ(0, ids.add(101).code());
    EXPECT_FALSE(ids.empty());
    EXPECT_EQ(101u, ids.id_unchecked(0));

    ids.clear();

    EXPECT_TRUE(ids.empty());
    EXPECT_EQ(0u, ids.count());
    EXPECT_FALSE(ids.contains(101));
}

TEST(roaring_ids, iterator_visits_ids_in_order) {
    RoaringIds ids;
    EXPECT_EQ(0, ids.init_writable(0).code());
    EXPECT_EQ(0, ids.add(3).code());
    EXPECT_EQ(0, ids.add(1).code());
    EXPECT_EQ(0, ids.add(5).code());

    std::vector<uint64_t> visited;
    for (auto it = ids.begin(); !it.eof(); it.next()) {
        visited.push_back(it.id());
        EXPECT_EQ(visited.size() - 1u, it.index());
    }

    EXPECT_EQ((std::vector<uint64_t>{1, 3, 5}), visited);
}

TEST(roaring_ids, iterator_on_empty_container_is_eof) {
    RoaringIds ids;
    EXPECT_EQ(0, ids.init_writable(0).code());

    auto it = ids.begin();

    EXPECT_TRUE(it.eof());
    EXPECT_EQ(it, ids.end());
}

TEST(roaring_ids, iterator_access_after_eof_throws) {
    RoaringIds ids;
    EXPECT_EQ(0, ids.init_writable(0).code());
    EXPECT_EQ(0, ids.add(1).code());

    auto it = ids.begin();
    ASSERT_FALSE(it.eof());
    it.next();
    ASSERT_TRUE(it.eof());

    EXPECT_THROW(it.id(), std::out_of_range);
    EXPECT_THROW(it.index(), std::out_of_range);
    EXPECT_THROW(*it, std::out_of_range);
}

TEST(roaring_ids, range_for_visits_ids_in_order) {
    RoaringIds ids;
    EXPECT_EQ(0, ids.init_writable(100).code());
    EXPECT_EQ(0, ids.add(103).code());
    EXPECT_EQ(0, ids.add(101).code());
    EXPECT_EQ(0, ids.add(105).code());

    std::vector<uint64_t> visited;
    for (uint64_t id : ids) {
        visited.push_back(id);
    }

    EXPECT_EQ((std::vector<uint64_t>{101, 103, 105}), visited);
}

TEST(roaring_ids, frozen_round_trip_preserves_values) {
    RoaringIds ids;
    EXPECT_EQ(0, ids.init_writable(0).code());
    for (uint32_t value : {1u, 2u, 3u, 1000u, 1001u, 1002u, 70000u}) {
        EXPECT_EQ(0, ids.add(value).code());
    }
    ids.compact();

    const size_t size = ids.serialized_size_bytes();
    std::vector<uint8_t> storage(size + 31u);
    char* data = aligned_32_data(storage);
    EXPECT_EQ(0, ids.serialize(data).code());

    RoaringIds mapped;
    EXPECT_EQ(0, mapped.init_frozen_view(reinterpret_cast<const uint8_t*>(data), size, 0).code());
    EXPECT_EQ(ids.count(), mapped.count());
    EXPECT_EQ(1u, mapped.id(0));
    EXPECT_EQ(1000u, mapped.id(3));
    EXPECT_EQ(6u, mapped.lower_bound_index(70000));
    EXPECT_NE(0, mapped.add(2000).code());
}

TEST(roaring_ids, empty_frozen_round_trip_preserves_empty_state) {
    RoaringIds ids;
    EXPECT_EQ(0, ids.init_writable(1000).code());
    ids.compact();

    const size_t size = ids.serialized_size_bytes();
    ASSERT_GT(size, 0u);
    std::vector<uint8_t> storage(size + 31u);
    char* data = aligned_32_data(storage);
    EXPECT_EQ(0, ids.serialize(data).code());

    RoaringIds mapped;
    EXPECT_EQ(0, mapped.init_frozen_view(reinterpret_cast<const uint8_t*>(data), size, 1000).code());
    EXPECT_TRUE(mapped.empty());
    EXPECT_EQ(0u, mapped.count());
    EXPECT_TRUE(mapped.begin().eof());
    EXPECT_FALSE(mapped.contains(1000));
    EXPECT_EQ(0u, mapped.lower_bound_index(1000));
}

TEST(roaring_ids, frozen_view_rejects_null_buffer) {
    RoaringIds mapped;

    const Ret ret = mapped.init_frozen_view(nullptr, 1, 0);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("RoaringIds::init_frozen_view: data pointer is null", ret.message());
}

TEST(roaring_ids, frozen_view_rejects_unaligned_buffer) {
    RoaringIds ids;
    EXPECT_EQ(0, ids.init_writable(0).code());
    EXPECT_EQ(0, ids.add(1).code());

    const size_t size = ids.serialized_size_bytes();
    std::vector<uint8_t> storage(size + 32u);
    char* data = aligned_32_data(storage);
    EXPECT_EQ(0, ids.serialize(data).code());

    RoaringIds mapped;
    const auto ret = mapped.init_frozen_view(reinterpret_cast<const uint8_t*>(data + 1), size, 0);
    EXPECT_NE(0, ret.code());
}

TEST(roaring_ids, frozen_view_rejects_truncated_buffer) {
    RoaringIds ids;
    EXPECT_EQ(0, ids.init_writable(0).code());
    for (uint32_t value : {1u, 2u, 3u, 1000u, 70000u}) {
        EXPECT_EQ(0, ids.add(value).code());
    }
    ids.compact();

    const size_t size = ids.serialized_size_bytes();
    ASSERT_GT(size, 1u);
    std::vector<uint8_t> storage(size + 31u);
    char* data = aligned_32_data(storage);
    EXPECT_EQ(0, ids.serialize(data).code());

    RoaringIds mapped;
    const Ret ret = mapped.init_frozen_view(
        reinterpret_cast<const uint8_t*>(data), size - 1u, 0);

    EXPECT_NE(0, ret.code());
}

TEST(roaring_ids, frozen_view_rejects_malformed_buffer) {
    std::vector<uint8_t> storage(64u + 31u, 0xff);
    char* data = aligned_32_data(storage);

    RoaringIds mapped;
    const Ret ret = mapped.init_frozen_view(reinterpret_cast<const uint8_t*>(data), 64u, 0);

    EXPECT_NE(0, ret.code());
}

TEST(roaring_ids, serialize_round_trip_preserves_values) {
    RoaringIds ids;
    EXPECT_EQ(0, ids.init_writable(1000).code());
    EXPECT_EQ(0, ids.add(1000).code());
    EXPECT_EQ(0, ids.add(1003).code());
    EXPECT_EQ(0, ids.add(1010).code());
    ids.compact();

    const size_t size = ids.serialized_size_bytes();
    ASSERT_GT(size, 0u);
    std::vector<uint8_t> aligned_storage(size + 31u);
    char* data = aligned_32_data(aligned_storage);
    ASSERT_EQ(0, ids.serialize(data).code());

    RoaringIds mapped;
    EXPECT_EQ(0, mapped.init_frozen_view(
        reinterpret_cast<const uint8_t*>(data), size, 1000).code());
    EXPECT_EQ(3u, mapped.count());
    EXPECT_EQ(1000u, mapped.id(0));
    EXPECT_EQ(1003u, mapped.id(1));
    EXPECT_EQ(1010u, mapped.id(2));
}

TEST(roaring_ids, stores_offsets_relative_to_base) {
    RoaringIds ids;
    EXPECT_EQ(0, ids.init_writable(1000).code());
    EXPECT_EQ(0, ids.add(1000).code());
    EXPECT_EQ(0, ids.add(1005).code());
    EXPECT_EQ(0, ids.add(1010).code());

    EXPECT_EQ(3u, ids.count());
    EXPECT_EQ(1000u, ids.id(0));
    EXPECT_EQ(1005u, ids.id(1));
    EXPECT_EQ(1010u, ids.id(2));
    EXPECT_EQ(0u, ids.lower_bound_index(999));
    EXPECT_EQ(0u, ids.lower_bound_index(1000));
    EXPECT_EQ(1u, ids.lower_bound_index(1001));
    EXPECT_EQ(2u, ids.lower_bound_index(1010));
    EXPECT_EQ(3u, ids.lower_bound_index(2000));
}

TEST(roaring_ids, rejects_ids_outside_uint32_offset_range) {
    RoaringIds ids;
    EXPECT_EQ(0, ids.init_writable(1000).code());

    EXPECT_NE(0, ids.add(999).code());
    EXPECT_NE(0, ids.add(1000ull + std::numeric_limits<uint32_t>::max() + 1ull).code());
}

TEST(roaring_ids, add_rejects_frozen_read_only_instance) {
    RoaringIds ids;
    EXPECT_EQ(0, ids.init_writable(0).code());
    EXPECT_EQ(0, ids.add(10).code());
    ids.compact();

    const size_t size = ids.serialized_size_bytes();
    std::vector<uint8_t> storage(size + 31u);
    char* data = aligned_32_data(storage);
    EXPECT_EQ(0, ids.serialize(data).code());

    RoaringIds mapped;
    EXPECT_EQ(0, mapped.init_frozen_view(reinterpret_cast<const uint8_t*>(data), size, 0).code());

    const Ret ret = mapped.add(20);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("RoaringIds::add: bitmap is read-only", ret.message());
    EXPECT_EQ(1u, mapped.count());
    EXPECT_TRUE(mapped.contains(10));
    EXPECT_FALSE(mapped.contains(20));
}

TEST(roaring_ids, large_set_round_trip_spans_multiple_containers) {
    constexpr uint64_t base = 1000;
    constexpr size_t values_count = 10000;
    constexpr uint64_t stride = 97;

    RoaringIds ids;
    EXPECT_EQ(0, ids.init_writable(base).code());
    for (size_t i = 0; i < values_count; ++i) {
        EXPECT_EQ(0, ids.add(base + i * stride).code());
    }
    ids.compact();

    EXPECT_EQ(values_count, ids.count());
    EXPECT_EQ(base, ids.id(0));
    EXPECT_EQ(base + 65536u / stride * stride, ids.id(65536u / stride));
    EXPECT_EQ(base + (values_count - 1u) * stride, ids.id(values_count - 1u));
    EXPECT_EQ(1u, ids.lower_bound_index(base + 1u));

    size_t index = 0;
    for (auto it = ids.begin(); !it.eof(); it.next(), ++index) {
        EXPECT_EQ(index, it.index());
        EXPECT_EQ(base + index * stride, it.id());
    }
    EXPECT_EQ(values_count, index);

    const size_t size = ids.serialized_size_bytes();
    std::vector<uint8_t> storage(size + 31u);
    char* data = aligned_32_data(storage);
    EXPECT_EQ(0, ids.serialize(data).code());

    RoaringIds mapped;
    EXPECT_EQ(0, mapped.init_frozen_view(reinterpret_cast<const uint8_t*>(data), size, base).code());
    EXPECT_EQ(values_count, mapped.count());
    EXPECT_EQ(base, mapped.id(0));
    EXPECT_EQ(base + (values_count - 1u) * stride, mapped.id(values_count - 1u));
}

TEST(roaring_ids, frozen_round_trip_applies_new_base) {
    RoaringIds ids;
    EXPECT_EQ(0, ids.init_writable(1000).code());
    EXPECT_EQ(0, ids.add(1000).code());
    EXPECT_EQ(0, ids.add(1002).code());
    EXPECT_EQ(0, ids.add(1004).code());

    const size_t size = ids.serialized_size_bytes();
    std::vector<uint8_t> storage(size + 31u);
    char* data = aligned_32_data(storage);
    EXPECT_EQ(0, ids.serialize(data).code());

    RoaringIds mapped;
    EXPECT_EQ(0, mapped.init_frozen_view(reinterpret_cast<const uint8_t*>(data), size, 5000).code());
    EXPECT_EQ(5000u, mapped.id(0));
    EXPECT_EQ(5002u, mapped.id(1));
    EXPECT_EQ(2u, mapped.lower_bound_index(5004));
}

// ---------- init_writable_copy ----------

TEST(roaring_ids, init_writable_copy_clones_from_frozen_view) {
    RoaringIds source;
    EXPECT_EQ(0, source.init_writable(1000).code());
    for (uint64_t v : {1000ull, 1003ull, 1010ull, 70000ull + 1000ull}) {
        EXPECT_EQ(0, source.add(v).code());
    }
    source.compact();

    const size_t size = source.serialized_size_bytes();
    std::vector<uint8_t> storage(size + 31u);
    char* data = aligned_32_data(storage);
    ASSERT_EQ(0, source.serialize(data).code());

    RoaringIds frozen;
    ASSERT_EQ(0, frozen.init_frozen_view(
        reinterpret_cast<const uint8_t*>(data), size, 1000).code());

    RoaringIds clone;
    ASSERT_EQ(0, clone.init_writable_copy(frozen, 1000).code());

    // Clone has the same membership as the frozen source.
    EXPECT_EQ(frozen.count(), clone.count());
    EXPECT_TRUE(clone.contains(1000));
    EXPECT_TRUE(clone.contains(1003));
    EXPECT_TRUE(clone.contains(1010));
    EXPECT_TRUE(clone.contains(70000u + 1000u));

    // Clone is writable: a new id sticks; frozen is unchanged.
    EXPECT_EQ(0, clone.add(1234).code());
    EXPECT_TRUE(clone.contains(1234));
    EXPECT_FALSE(frozen.contains(1234));
}

TEST(roaring_ids, init_writable_copy_from_uninitialized_yields_empty_writable) {
    RoaringIds empty_source;  // never initialized -> bitmap() == nullptr

    RoaringIds clone;
    ASSERT_EQ(0, clone.init_writable_copy(empty_source, 500).code());
    EXPECT_EQ(0u, clone.count());

    // Confirm clone is writable, not read-only.
    EXPECT_EQ(0, clone.add(500).code());
    EXPECT_TRUE(clone.contains(500));
}

TEST(roaring_ids, init_writable_copy_rejects_base_mismatch) {
    RoaringIds source;
    ASSERT_EQ(0, source.init_writable(1000).code());
    ASSERT_EQ(0, source.add(1005).code());

    RoaringIds clone;
    const Ret ret = clone.init_writable_copy(source, 2000);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("RoaringIds::init_writable_copy: base mismatch", ret.message());
    EXPECT_EQ(0u, clone.count());  // clone untouched on failure
}

// ---------- union_in_place ----------

TEST(roaring_ids, union_in_place_merges_disjoint_and_overlapping_sets) {
    RoaringIds a;
    ASSERT_EQ(0, a.init_writable(0).code());
    for (uint64_t v : {1ull, 5ull, 10ull}) {
        ASSERT_EQ(0, a.add(v).code());
    }

    RoaringIds b;
    ASSERT_EQ(0, b.init_writable(0).code());
    for (uint64_t v : {5ull, 7ull, 20ull}) {
        ASSERT_EQ(0, b.add(v).code());
    }

    ASSERT_EQ(0, a.union_in_place(b).code());

    // {1,5,10} ∪ {5,7,20} = {1,5,7,10,20}
    EXPECT_EQ(5u, a.count());
    EXPECT_TRUE(a.contains(1));
    EXPECT_TRUE(a.contains(5));
    EXPECT_TRUE(a.contains(7));
    EXPECT_TRUE(a.contains(10));
    EXPECT_TRUE(a.contains(20));
    // b is unchanged.
    EXPECT_EQ(3u, b.count());
}

TEST(roaring_ids, union_in_place_with_empty_other_is_noop) {
    RoaringIds a;
    ASSERT_EQ(0, a.init_writable(0).code());
    ASSERT_EQ(0, a.add(42).code());

    // Other is uninitialized (bitmap() == nullptr).
    RoaringIds empty_other;
    ASSERT_EQ(0, a.union_in_place(empty_other).code());
    EXPECT_EQ(1u, a.count());
    EXPECT_TRUE(a.contains(42));

    // Other is initialized but holds nothing.
    RoaringIds initialized_empty;
    ASSERT_EQ(0, initialized_empty.init_writable(0).code());
    ASSERT_EQ(0, a.union_in_place(initialized_empty).code());
    EXPECT_EQ(1u, a.count());
}

TEST(roaring_ids, union_in_place_with_self_is_noop) {
    RoaringIds a;
    ASSERT_EQ(0, a.init_writable(0).code());
    ASSERT_EQ(0, a.add(1).code());
    ASSERT_EQ(0, a.add(2).code());

    ASSERT_EQ(0, a.union_in_place(a).code());

    EXPECT_EQ(2u, a.count());
    EXPECT_TRUE(a.contains(1));
    EXPECT_TRUE(a.contains(2));
}

TEST(roaring_ids, union_in_place_rejects_uninitialized_target) {
    RoaringIds uninit;
    RoaringIds other;
    ASSERT_EQ(0, other.init_writable(0).code());

    const Ret ret = uninit.union_in_place(other);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("RoaringIds::union_in_place: bitmap is not initialized", ret.message());
}

TEST(roaring_ids, union_in_place_rejects_read_only_target) {
    RoaringIds writable;
    ASSERT_EQ(0, writable.init_writable(0).code());
    ASSERT_EQ(0, writable.add(1).code());
    writable.compact();

    const size_t size = writable.serialized_size_bytes();
    std::vector<uint8_t> storage(size + 31u);
    char* data = aligned_32_data(storage);
    ASSERT_EQ(0, writable.serialize(data).code());

    RoaringIds frozen;
    ASSERT_EQ(0, frozen.init_frozen_view(
        reinterpret_cast<const uint8_t*>(data), size, 0).code());

    RoaringIds other;
    ASSERT_EQ(0, other.init_writable(0).code());
    ASSERT_EQ(0, other.add(2).code());

    const Ret ret = frozen.union_in_place(other);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("RoaringIds::union_in_place: bitmap is read-only", ret.message());
    EXPECT_FALSE(frozen.contains(2));
}

TEST(roaring_ids, union_in_place_rejects_base_mismatch) {
    RoaringIds a;
    ASSERT_EQ(0, a.init_writable(1000).code());
    ASSERT_EQ(0, a.add(1005).code());

    RoaringIds b;
    ASSERT_EQ(0, b.init_writable(2000).code());
    ASSERT_EQ(0, b.add(2005).code());

    const Ret ret = a.union_in_place(b);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("RoaringIds::union_in_place: base mismatch", ret.message());
    EXPECT_EQ(1u, a.count());  // a untouched
}

// ---------- andnot_in_place ----------

TEST(roaring_ids, andnot_in_place_removes_overlapping_ids) {
    RoaringIds a;
    ASSERT_EQ(0, a.init_writable(0).code());
    for (uint64_t v : {1ull, 5ull, 10ull, 15ull}) {
        ASSERT_EQ(0, a.add(v).code());
    }

    RoaringIds b;
    ASSERT_EQ(0, b.init_writable(0).code());
    for (uint64_t v : {5ull, 10ull, 999ull}) {
        ASSERT_EQ(0, b.add(v).code());
    }

    ASSERT_EQ(0, a.andnot_in_place(b).code());

    // {1,5,10,15} - {5,10,999} = {1,15}
    EXPECT_EQ(2u, a.count());
    EXPECT_TRUE(a.contains(1));
    EXPECT_FALSE(a.contains(5));
    EXPECT_FALSE(a.contains(10));
    EXPECT_TRUE(a.contains(15));
    // b is unchanged.
    EXPECT_EQ(3u, b.count());
}

TEST(roaring_ids, andnot_in_place_with_empty_other_is_noop) {
    RoaringIds a;
    ASSERT_EQ(0, a.init_writable(0).code());
    ASSERT_EQ(0, a.add(42).code());

    RoaringIds uninit;
    ASSERT_EQ(0, a.andnot_in_place(uninit).code());
    EXPECT_EQ(1u, a.count());
    EXPECT_TRUE(a.contains(42));

    RoaringIds initialized_empty;
    ASSERT_EQ(0, initialized_empty.init_writable(0).code());
    ASSERT_EQ(0, a.andnot_in_place(initialized_empty).code());
    EXPECT_EQ(1u, a.count());
    EXPECT_TRUE(a.contains(42));
}

TEST(roaring_ids, andnot_in_place_self_is_rejected) {
    RoaringIds a;
    ASSERT_EQ(0, a.init_writable(0).code());
    ASSERT_EQ(0, a.add(1).code());
    ASSERT_EQ(0, a.add(2).code());

    const Ret ret = a.andnot_in_place(a);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("RoaringIds::andnot_in_place: self-difference is not allowed",
              ret.message());
    // a is left intact.
    EXPECT_EQ(2u, a.count());
    EXPECT_TRUE(a.contains(1));
    EXPECT_TRUE(a.contains(2));
}

TEST(roaring_ids, andnot_in_place_rejects_uninitialized_target) {
    RoaringIds uninit;
    RoaringIds other;
    ASSERT_EQ(0, other.init_writable(0).code());

    const Ret ret = uninit.andnot_in_place(other);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("RoaringIds::andnot_in_place: bitmap is not initialized",
              ret.message());
}

TEST(roaring_ids, andnot_in_place_rejects_read_only_target) {
    RoaringIds writable;
    ASSERT_EQ(0, writable.init_writable(0).code());
    ASSERT_EQ(0, writable.add(1).code());
    ASSERT_EQ(0, writable.add(2).code());
    writable.compact();

    const size_t size = writable.serialized_size_bytes();
    std::vector<uint8_t> storage(size + 31u);
    char* data = aligned_32_data(storage);
    ASSERT_EQ(0, writable.serialize(data).code());

    RoaringIds frozen;
    ASSERT_EQ(0, frozen.init_frozen_view(
        reinterpret_cast<const uint8_t*>(data), size, 0).code());

    RoaringIds other;
    ASSERT_EQ(0, other.init_writable(0).code());
    ASSERT_EQ(0, other.add(1).code());

    const Ret ret = frozen.andnot_in_place(other);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("RoaringIds::andnot_in_place: bitmap is read-only", ret.message());
    EXPECT_TRUE(frozen.contains(1));  // unchanged
}

TEST(roaring_ids, andnot_in_place_rejects_base_mismatch) {
    RoaringIds a;
    ASSERT_EQ(0, a.init_writable(1000).code());
    ASSERT_EQ(0, a.add(1005).code());

    RoaringIds b;
    ASSERT_EQ(0, b.init_writable(2000).code());
    ASSERT_EQ(0, b.add(2005).code());

    const Ret ret = a.andnot_in_place(b);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("RoaringIds::andnot_in_place: base mismatch", ret.message());
    EXPECT_EQ(1u, a.count());
}

} // namespace
} // namespace sketch2
