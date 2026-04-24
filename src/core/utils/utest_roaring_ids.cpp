// Unit tests for RoaringIds.

#include "roaring_ids.h"

#include <cstdio>
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

TEST(roaring_ids, write_rejects_null_file_handle) {
    RoaringIds ids;
    EXPECT_EQ(0, ids.init_writable(0).code());

    const Ret ret = ids.write(nullptr, "write failed");

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("write failed: file handle is null", ret.message());
}

TEST(roaring_ids, write_round_trip_preserves_values) {
    RoaringIds ids;
    EXPECT_EQ(0, ids.init_writable(1000).code());
    EXPECT_EQ(0, ids.add(1000).code());
    EXPECT_EQ(0, ids.add(1003).code());
    EXPECT_EQ(0, ids.add(1010).code());
    ids.compact();

    FILE* f = tmpfile();
    ASSERT_NE(nullptr, f);
    ASSERT_EQ(0, ids.write(f, "write failed").code());

    ASSERT_EQ(0, fseek(f, 0, SEEK_END));
    const long file_size = ftell(f);
    ASSERT_GT(file_size, 0);
    ASSERT_EQ(0, fseek(f, 0, SEEK_SET));

    std::vector<uint8_t> serialized(static_cast<size_t>(file_size));
    ASSERT_EQ(serialized.size(), fread(serialized.data(), 1, serialized.size(), f));
    ASSERT_EQ(0, fclose(f));

    std::vector<uint8_t> aligned_storage(serialized.size() + 31u);
    char* data = aligned_32_data(aligned_storage);
    std::memcpy(data, serialized.data(), serialized.size());

    RoaringIds mapped;
    EXPECT_EQ(0, mapped.init_frozen_view(
        reinterpret_cast<const uint8_t*>(data), serialized.size(), 1000).code());
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

} // namespace
} // namespace sketch2
