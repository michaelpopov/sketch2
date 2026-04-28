// Unit tests for RoaringIds and RoaringIdsBuilder.

#include "roaring_ids.h"

#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace sketch2 {
namespace {

char* aligned_32_data(std::vector<uint8_t>& buffer) {
    const uintptr_t raw = reinterpret_cast<uintptr_t>(buffer.data());
    const uintptr_t aligned = (raw + 31u) & ~uintptr_t{31u};
    return reinterpret_cast<char*>(aligned);
}

RoaringIds build_ids(uint64_t base, std::initializer_list<uint64_t> values) {
    RoaringIdsBuilder builder;
    EXPECT_EQ(0, builder.init(base).code());
    for (uint64_t value : values) {
        EXPECT_EQ(0, builder.add(value).code());
    }
    return std::move(builder).build();
}

void init_frozen_round_trip(const RoaringIds& ids,
        uint64_t base,
        std::vector<uint8_t>* storage,
        RoaringIds* mapped) {
    const size_t size = ids.serialized_size_bytes();
    storage->assign(size + 31u, 0);
    char* data = aligned_32_data(*storage);
    EXPECT_EQ(0, ids.serialize(data).code());

    EXPECT_EQ(0, mapped->init_frozen_view(reinterpret_cast<const uint8_t*>(data), size, base).code());
}

TEST(roaring_ids_builder, add_and_build_ids_in_sorted_order) {
    RoaringIdsBuilder builder;
    EXPECT_EQ(0, builder.init(0).code());
    EXPECT_TRUE(builder.empty());
    EXPECT_EQ(0, builder.add(30).code());
    EXPECT_EQ(0, builder.add(10).code());
    EXPECT_EQ(0, builder.add(20).code());
    EXPECT_EQ(0, builder.add(20).code());
    EXPECT_EQ(3u, builder.count());

    RoaringIds ids = std::move(builder).build();

    EXPECT_EQ(3u, ids.count());
    EXPECT_EQ(10u, ids.id(0));
    EXPECT_EQ(20u, ids.id(1));
    EXPECT_EQ(30u, ids.id(2));
    EXPECT_THROW(ids.id(3), std::out_of_range);
}

TEST(roaring_ids_builder, load_collapses_sorted_values) {
    const uint64_t values[] = {1000, 1001, 1002, 1010, 1020, 1021};
    RoaringIdsBuilder builder;
    ASSERT_EQ(0, builder.init(1000).code());
    ASSERT_EQ(0, builder.load(values, std::size(values)).code());

    RoaringIds ids = std::move(builder).build();

    EXPECT_EQ(std::size(values), ids.count());
    EXPECT_TRUE(ids.contains(1000));
    EXPECT_TRUE(ids.contains(1002));
    EXPECT_FALSE(ids.contains(1003));
    EXPECT_TRUE(ids.contains(1021));
}

TEST(roaring_ids_builder, init_buffered_rejects_zero_buffer_size) {
    RoaringIdsBuilder builder;
    const Ret ret = builder.init_buffered(0, 0);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("RoaringIdsBuilder::init_buffered: buffer size must be greater than 0",
        ret.message());
}

TEST(roaring_ids_builder, buffered_add_flushes_when_full_and_build_flushes_remainder) {
    RoaringIdsBuilder builder;
    ASSERT_EQ(0, builder.init_buffered(1000, 3).code());

    ASSERT_EQ(0, builder.add(1002).code());
    ASSERT_EQ(0, builder.add(1000).code());
    EXPECT_EQ(2u, builder.count());
    ASSERT_EQ(0, builder.add(1001).code());
    EXPECT_EQ(3u, builder.count());
    ASSERT_EQ(0, builder.add(1010).code());
    EXPECT_EQ(4u, builder.count());

    RoaringIds ids = std::move(builder).build();

    EXPECT_EQ(4u, ids.count());
    EXPECT_EQ(1000u, ids.id(0));
    EXPECT_EQ(1001u, ids.id(1));
    EXPECT_EQ(1002u, ids.id(2));
    EXPECT_EQ(1010u, ids.id(3));
}

TEST(roaring_ids_builder, load_flushes_pending_buffer_before_loading_sorted_values) {
    const uint64_t values[] = {1010, 1011, 1012};
    RoaringIdsBuilder builder;
    ASSERT_EQ(0, builder.init_buffered(1000, 4).code());
    ASSERT_EQ(0, builder.add(1002).code());
    ASSERT_EQ(0, builder.add(1000).code());
    ASSERT_EQ(0, builder.load(values, std::size(values)).code());

    RoaringIds ids = std::move(builder).build();

    EXPECT_EQ(5u, ids.count());
    EXPECT_TRUE(ids.contains(1000));
    EXPECT_TRUE(ids.contains(1002));
    EXPECT_TRUE(ids.contains(1010));
    EXPECT_TRUE(ids.contains(1011));
    EXPECT_TRUE(ids.contains(1012));
}

TEST(roaring_ids_builder, rejects_ids_outside_uint32_offset_range) {
    RoaringIdsBuilder builder;
    EXPECT_EQ(0, builder.init(1000).code());

    EXPECT_NE(0, builder.add(999).code());
    EXPECT_NE(0, builder.add(1000ull + std::numeric_limits<uint32_t>::max() + 1ull).code());
}

TEST(roaring_ids_builder, init_copy_clones_read_only_ids) {
    RoaringIds source = build_ids(1000, {1000, 1003, 1010, 71000});
    std::vector<uint8_t> storage;
    RoaringIds frozen;
    init_frozen_round_trip(source, 1000, &storage, &frozen);

    RoaringIdsBuilder builder;
    ASSERT_EQ(0, builder.init_copy(frozen, 1000).code());
    ASSERT_EQ(0, builder.add(1234).code());
    RoaringIds clone = std::move(builder).build();

    EXPECT_TRUE(clone.contains(1000));
    EXPECT_TRUE(clone.contains(1010));
    EXPECT_TRUE(clone.contains(71000));
    EXPECT_TRUE(clone.contains(1234));
    EXPECT_FALSE(frozen.contains(1234));
}

TEST(roaring_ids_builder, init_copy_from_uninitialized_yields_empty_writable) {
    RoaringIds empty_source;
    RoaringIdsBuilder builder;
    ASSERT_EQ(0, builder.init_copy(empty_source, 500).code());
    EXPECT_EQ(0u, builder.count());
    EXPECT_EQ(0, builder.add(500).code());

    RoaringIds ids = std::move(builder).build();
    EXPECT_TRUE(ids.contains(500));
}

TEST(roaring_ids_builder, init_copy_resets_buffered_mode) {
    RoaringIds source = build_ids(0, {1});
    RoaringIdsBuilder builder;
    ASSERT_EQ(0, builder.init_buffered(0, 3).code());
    ASSERT_EQ(0, builder.add(30).code());
    ASSERT_EQ(0, builder.init_copy(source, 0).code());

    ASSERT_EQ(0, builder.add(20).code());
    RoaringIds ids = std::move(builder).build();

    EXPECT_EQ(2u, ids.count());
    EXPECT_TRUE(ids.contains(1));
    EXPECT_TRUE(ids.contains(20));
    EXPECT_FALSE(ids.contains(30));
}

TEST(roaring_ids_builder, init_copy_rejects_base_mismatch) {
    RoaringIds source = build_ids(1000, {1005});

    RoaringIdsBuilder builder;
    const Ret ret = builder.init_copy(source, 2000);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("RoaringIdsBuilder::init_copy: base mismatch", ret.message());
    EXPECT_EQ(0u, builder.count());
}

TEST(roaring_ids_builder, union_in_place_merges_read_only_sets) {
    RoaringIds a = build_ids(0, {1, 5, 10});
    RoaringIds b = build_ids(0, {5, 7, 20});

    RoaringIdsBuilder builder;
    ASSERT_EQ(0, builder.init_copy(a, 0).code());
    ASSERT_EQ(0, builder.union_in_place(b).code());
    RoaringIds merged = std::move(builder).build();

    EXPECT_EQ(5u, merged.count());
    EXPECT_TRUE(merged.contains(1));
    EXPECT_TRUE(merged.contains(5));
    EXPECT_TRUE(merged.contains(7));
    EXPECT_TRUE(merged.contains(10));
    EXPECT_TRUE(merged.contains(20));
}

TEST(roaring_ids_builder, union_in_place_flushes_pending_buffer_first) {
    RoaringIds other = build_ids(0, {5, 7, 20});
    RoaringIdsBuilder builder;
    ASSERT_EQ(0, builder.init_buffered(0, 4).code());
    ASSERT_EQ(0, builder.add(10).code());
    ASSERT_EQ(0, builder.add(1).code());
    ASSERT_EQ(0, builder.union_in_place(other).code());

    RoaringIds merged = std::move(builder).build();

    EXPECT_EQ(5u, merged.count());
    EXPECT_TRUE(merged.contains(1));
    EXPECT_TRUE(merged.contains(5));
    EXPECT_TRUE(merged.contains(7));
    EXPECT_TRUE(merged.contains(10));
    EXPECT_TRUE(merged.contains(20));
}

TEST(roaring_ids_builder, union_in_place_with_source_snapshot_is_noop) {
    RoaringIds source = build_ids(0, {1, 2, 10});

    RoaringIdsBuilder builder;
    ASSERT_EQ(0, builder.init_copy(source, 0).code());
    ASSERT_EQ(0, builder.union_in_place(source).code());
    RoaringIds merged = std::move(builder).build();

    EXPECT_EQ(3u, merged.count());
    EXPECT_TRUE(merged.contains(1));
    EXPECT_TRUE(merged.contains(2));
    EXPECT_TRUE(merged.contains(10));
}

TEST(roaring_ids_builder, andnot_in_place_removes_read_only_set) {
    RoaringIds a = build_ids(0, {1, 5, 10, 15});
    RoaringIds b = build_ids(0, {5, 10, 999});

    RoaringIdsBuilder builder;
    ASSERT_EQ(0, builder.init_copy(a, 0).code());
    ASSERT_EQ(0, builder.andnot_in_place(b).code());
    RoaringIds diff = std::move(builder).build();

    EXPECT_EQ(2u, diff.count());
    EXPECT_TRUE(diff.contains(1));
    EXPECT_FALSE(diff.contains(5));
    EXPECT_FALSE(diff.contains(10));
    EXPECT_TRUE(diff.contains(15));
}

TEST(roaring_ids_builder, andnot_in_place_with_source_snapshot_empties_builder) {
    RoaringIds source = build_ids(0, {1, 2, 10});

    RoaringIdsBuilder builder;
    ASSERT_EQ(0, builder.init_copy(source, 0).code());
    ASSERT_EQ(0, builder.andnot_in_place(source).code());
    RoaringIds diff = std::move(builder).build();

    EXPECT_TRUE(diff.empty());
    EXPECT_EQ(0u, diff.count());
}

TEST(roaring_ids_builder, algebra_rejects_base_mismatch) {
    RoaringIds other = build_ids(2000, {2005});
    RoaringIdsBuilder builder;
    ASSERT_EQ(0, builder.init(1000).code());
    ASSERT_EQ(0, builder.add(1005).code());

    EXPECT_EQ("RoaringIdsBuilder::union_in_place: base mismatch",
        builder.union_in_place(other).message());
    EXPECT_EQ("RoaringIdsBuilder::andnot_in_place: base mismatch",
        builder.andnot_in_place(other).message());
}

TEST(roaring_ids, lower_bound_index_matches_sorted_semantics) {
    RoaringIds ids = build_ids(0, {10, 20, 40, 50});

    EXPECT_EQ(0u, ids.lower_bound_index(1));
    EXPECT_EQ(0u, ids.lower_bound_index(10));
    EXPECT_EQ(1u, ids.lower_bound_index(11));
    EXPECT_EQ(2u, ids.lower_bound_index(40));
    EXPECT_EQ(4u, ids.lower_bound_index(100));
}

TEST(roaring_ids, find_index_matches_exact_membership) {
    RoaringIds ids = build_ids(1000, {1000, 1010, 1020});

    size_t index = RoaringIds::npos;
    EXPECT_TRUE(ids.find_index(1000, &index));
    EXPECT_EQ(0u, index);
    EXPECT_TRUE(ids.find_index(1020, &index));
    EXPECT_EQ(2u, index);

    EXPECT_FALSE(ids.find_index(999, &index));
    EXPECT_EQ(RoaringIds::npos, index);
    EXPECT_FALSE(ids.find_index(1015, &index));
    EXPECT_EQ(RoaringIds::npos, index);
}

TEST(roaring_ids, lookup_on_default_constructed_state_reports_absent) {
    RoaringIds ids;
    size_t index = 123;

    EXPECT_TRUE(ids.empty());
    EXPECT_EQ(0u, ids.count());
    EXPECT_FALSE(ids.contains(0));
    EXPECT_FALSE(ids.contains(1000));
    EXPECT_FALSE(ids.find_index(0, &index));
    EXPECT_EQ(RoaringIds::npos, index);
    index = 123;
    EXPECT_FALSE(ids.find_index(1000, &index));
    EXPECT_EQ(RoaringIds::npos, index);
}

TEST(roaring_ids, contains_matches_base_adjusted_membership) {
    RoaringIds ids = build_ids(1000, {1000, 1005});

    EXPECT_FALSE(ids.contains(999));
    EXPECT_TRUE(ids.contains(1000));
    EXPECT_FALSE(ids.contains(1001));
    EXPECT_TRUE(ids.contains(1005));
    EXPECT_FALSE(ids.contains(1000ull + std::numeric_limits<uint32_t>::max() + 1ull));
}

TEST(roaring_ids, reset_view_drops_container) {
    RoaringIds ids = build_ids(100, {101});
    ASSERT_FALSE(ids.empty());

    ids.reset_view();

    EXPECT_TRUE(ids.empty());
    EXPECT_EQ(0u, ids.count());
    EXPECT_FALSE(ids.contains(101));
    size_t index = 123;
    EXPECT_FALSE(ids.find_index(101, &index));
    EXPECT_EQ(RoaringIds::npos, index);
}

TEST(roaring_ids, iterator_visits_ids_in_order) {
    RoaringIds ids = build_ids(0, {3, 1, 5});

    std::vector<uint64_t> visited;
    for (auto it = ids.begin(); !it.eof(); it.next()) {
        visited.push_back(it.id());
        EXPECT_EQ(visited.size() - 1u, it.index());
    }

    EXPECT_EQ((std::vector<uint64_t>{1, 3, 5}), visited);
}

TEST(roaring_ids, iterator_seek_at_least_updates_id_and_index) {
    RoaringIds ids = build_ids(1000, {1000, 1005, 1010, 2000});

    auto it = ids.begin();
    ASSERT_FALSE(it.eof());
    EXPECT_TRUE(it.seek_at_least(1004));
    EXPECT_FALSE(it.eof());
    EXPECT_EQ(1005u, it.id());
    EXPECT_EQ(1u, it.index());

    EXPECT_TRUE(it.seek_at_least(1005));
    EXPECT_EQ(1005u, it.id());
    EXPECT_EQ(1u, it.index());

    EXPECT_TRUE(it.seek_at_least(1500));
    EXPECT_EQ(2000u, it.id());
    EXPECT_EQ(3u, it.index());

    EXPECT_FALSE(it.seek_at_least(2001));
    EXPECT_TRUE(it.eof());
    EXPECT_THROW(it.index(), std::out_of_range);
}

TEST(roaring_ids, seek_cursor_seek_at_least_updates_id_without_index) {
    RoaringIds ids = build_ids(1000, {1000, 1005, 1010, 2000});

    auto cursor = ids.seek_begin();
    ASSERT_FALSE(cursor.eof());
    EXPECT_TRUE(cursor.seek_at_least(1004));
    EXPECT_FALSE(cursor.eof());
    EXPECT_EQ(1005u, cursor.id());

    EXPECT_TRUE(cursor.seek_at_least(1005));
    EXPECT_EQ(1005u, cursor.id());

    cursor.next();
    EXPECT_EQ(1010u, cursor.id());

    EXPECT_TRUE(cursor.seek_at_least(1500));
    EXPECT_EQ(2000u, cursor.id());

    EXPECT_FALSE(cursor.seek_at_least(2001));
    EXPECT_TRUE(cursor.eof());
    EXPECT_THROW(cursor.id(), std::out_of_range);
}

TEST(roaring_ids, iterator_on_empty_container_is_eof) {
    RoaringIdsBuilder builder;
    ASSERT_EQ(0, builder.init(0).code());
    RoaringIds ids = std::move(builder).build();

    auto it = ids.begin();

    EXPECT_TRUE(it.eof());
    EXPECT_EQ(it, ids.end());
}

TEST(roaring_ids, iterator_access_after_eof_throws) {
    RoaringIds ids = build_ids(0, {1});

    auto it = ids.begin();
    ASSERT_FALSE(it.eof());
    it.next();
    ASSERT_TRUE(it.eof());

    EXPECT_THROW(it.id(), std::out_of_range);
    EXPECT_THROW(it.index(), std::out_of_range);
    EXPECT_THROW(*it, std::out_of_range);
}

TEST(roaring_ids, range_for_visits_ids_in_order) {
    RoaringIds ids = build_ids(100, {103, 101, 105});

    std::vector<uint64_t> visited;
    for (uint64_t id : ids) {
        visited.push_back(id);
    }

    EXPECT_EQ((std::vector<uint64_t>{101, 103, 105}), visited);
}

TEST(roaring_ids, frozen_round_trip_preserves_values) {
    RoaringIds ids = build_ids(0, {1, 2, 3, 1000, 1001, 1002, 70000});
    std::vector<uint8_t> storage;
    RoaringIds mapped;
    init_frozen_round_trip(ids, 0, &storage, &mapped);

    EXPECT_EQ(ids.count(), mapped.count());
    EXPECT_EQ(1u, mapped.id(0));
    EXPECT_EQ(1000u, mapped.id(3));
    EXPECT_EQ(6u, mapped.lower_bound_index(70000));
}

TEST(roaring_ids, empty_frozen_round_trip_preserves_empty_state) {
    RoaringIdsBuilder builder;
    ASSERT_EQ(0, builder.init(1000).code());
    RoaringIds ids = std::move(builder).build();
    std::vector<uint8_t> storage;
    RoaringIds mapped;
    init_frozen_round_trip(ids, 1000, &storage, &mapped);

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
    RoaringIds ids = build_ids(0, {1});
    const size_t size = ids.serialized_size_bytes();
    std::vector<uint8_t> storage(size + 32u);
    char* data = aligned_32_data(storage);
    EXPECT_EQ(0, ids.serialize(data).code());

    RoaringIds mapped;
    const Ret ret = mapped.init_frozen_view(reinterpret_cast<const uint8_t*>(data + 1), size, 0);
    EXPECT_NE(0, ret.code());
}

TEST(roaring_ids, frozen_view_rejects_truncated_buffer) {
    RoaringIds ids = build_ids(0, {1, 2, 3, 1000, 70000});
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

TEST(roaring_ids, stores_offsets_relative_to_base) {
    RoaringIds ids = build_ids(1000, {1000, 1005, 1010});

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

TEST(roaring_ids, large_set_round_trip_spans_multiple_containers) {
    constexpr uint64_t base = 1000;
    constexpr size_t values_count = 10000;
    constexpr uint64_t stride = 97;

    RoaringIdsBuilder builder;
    ASSERT_EQ(0, builder.init(base).code());
    for (size_t i = 0; i < values_count; ++i) {
        ASSERT_EQ(0, builder.add(base + i * stride).code());
    }
    RoaringIds ids = std::move(builder).build();

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

    std::vector<uint8_t> storage;
    RoaringIds mapped;
    init_frozen_round_trip(ids, base, &storage, &mapped);
    EXPECT_EQ(values_count, mapped.count());
    EXPECT_EQ(base, mapped.id(0));
    EXPECT_EQ(base + (values_count - 1u) * stride, mapped.id(values_count - 1u));
}

TEST(roaring_ids, frozen_round_trip_applies_new_base) {
    RoaringIds ids = build_ids(1000, {1000, 1002, 1004});
    std::vector<uint8_t> storage;
    RoaringIds mapped;
    init_frozen_round_trip(ids, 5000, &storage, &mapped);

    EXPECT_EQ(5000u, mapped.id(0));
    EXPECT_EQ(5002u, mapped.id(1));
    EXPECT_EQ(2u, mapped.lower_bound_index(5004));
}

} // namespace
} // namespace sketch2
