// Unit tests for CompactIdsMisses and CompactIdsBitset.

#include "compact_ids_ext.h"
#include "compact_ids_shared.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <unistd.h>
#include <vector>

#include "utest_tmp_dir.h"

namespace sketch2 {

namespace {

std::vector<uint8_t> read_file_bytes(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return {};
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return {};
    }
    const long size = ftell(f);
    if (size < 0) {
        fclose(f);
        return {};
    }
    rewind(f);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (!bytes.empty()) {
        const size_t read = fread(bytes.data(), 1, bytes.size(), f);
        if (read != bytes.size()) {
            fclose(f);
            return {};
        }
    }
    fclose(f);
    return bytes;
}

std::vector<uint8_t> serialize_ids_to_bytes(const CompactIdsBitset& ids) {
    const std::string path =
        tmp_dir() + "/sketch2_utest_compact_ids_ext_" + std::to_string(getpid()) + ".bin";
    std::remove(path.c_str());

    FILE* f = fopen(path.c_str(), "wb");
    EXPECT_NE(nullptr, f);
    if (f == nullptr) {
        return {};
    }

    EXPECT_EQ(0, ids.write(f, "write failed").code());
    EXPECT_EQ(0, fclose(f));

    std::vector<uint8_t> bytes = read_file_bytes(path);
    std::remove(path.c_str());
    return bytes;
}

CompactIdsHeader read_header_or_die(const std::vector<uint8_t>& serialized) {
    CompactIdsHeader hdr{};
    EXPECT_GE(serialized.size(), sizeof(hdr));
    if (serialized.size() >= sizeof(hdr)) {
        std::memcpy(&hdr, serialized.data(), sizeof(hdr));
    }
    return hdr;
}

std::vector<uint8_t> serialize_auto_ids_to_bytes(const CompactIdsExt& ids) {
    const std::string path =
        tmp_dir() + "/sketch2_utest_compact_ids_ext_auto_" + std::to_string(getpid()) + ".bin";
    std::remove(path.c_str());

    FILE* f = fopen(path.c_str(), "wb");
    EXPECT_NE(nullptr, f);
    if (f == nullptr) {
        return {};
    }

    EXPECT_EQ(0, ids.write(f, "write failed").code());
    EXPECT_EQ(0, fclose(f));

    std::vector<uint8_t> bytes = read_file_bytes(path);
    std::remove(path.c_str());
    return bytes;
}

CompactIdsAccumulator make_accumulator(const std::vector<uint64_t>& ids) {
    CompactIdsAccumulator accumulator;
    if (ids.empty()) {
        accumulator.init(0, 0);
        return accumulator;
    }

    accumulator.init(ids.front(), ids.size());
    for (uint64_t id : ids) {
        accumulator.add(id);
    }
    return accumulator;
}

std::vector<uint8_t> serialize_misses_to_bytes(
    uint64_t base,
    uint32_t count,
    const std::vector<uint32_t>& misses) {
    CompactIdsHeader hdr{};
    hdr.encoding = static_cast<uint8_t>(CompactIdsExtEncoding::Misses32);
    hdr.count = count;
    hdr.aux_data = static_cast<uint32_t>(misses.size());
    hdr.payload_size = static_cast<uint32_t>(misses.size() * sizeof(uint32_t));
    hdr.base = count == 0 ? 0u : base;

    std::vector<uint8_t> serialized(sizeof(hdr) + hdr.payload_size, 0);
    std::memcpy(serialized.data(), &hdr, sizeof(hdr));
    if (!misses.empty()) {
        std::memcpy(serialized.data() + sizeof(hdr), misses.data(), hdr.payload_size);
    }
    return serialized;
}

std::vector<uint8_t> make_unaligned_buffer(const std::vector<uint8_t>& bytes) {
    std::vector<uint8_t> unaligned(bytes.size() + 1, 0);
    std::memcpy(unaligned.data() + 1, bytes.data(), bytes.size());
    return unaligned;
}

} // namespace

TEST(compact_ids_bitset, init_empty_container) {
    CompactIdsBitset ids;

    ASSERT_EQ(0, ids.init(std::vector<uint64_t>{}).code());
    EXPECT_TRUE(ids.empty());
    EXPECT_EQ(0u, ids.count());
    EXPECT_EQ(sizeof(CompactIdsHeader), ids.serialized_size_bytes());
    EXPECT_EQ(0u, ids.lower_bound_index(123));
}

TEST(compact_ids_bitset, init_from_sorted_ids_preserves_values) {
    CompactIdsBitset ids;
    const std::vector<uint64_t> values = {100, 101, 103, 108, 109};

    ASSERT_EQ(0, ids.init(values).code());
    EXPECT_EQ(values.size(), ids.count());

    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(values[i], ids.id(i));
        EXPECT_EQ(values[i], ids.id_unchecked(i));
    }
}

TEST(compact_ids_bitset, lower_bound_index_matches_membership_boundaries) {
    CompactIdsBitset ids;
    ASSERT_EQ(0, ids.init(std::vector<uint64_t>{100, 101, 103, 108, 109}).code());

    EXPECT_EQ(0u, ids.lower_bound_index(99));
    EXPECT_EQ(0u, ids.lower_bound_index(100));
    EXPECT_EQ(2u, ids.lower_bound_index(102));
    EXPECT_EQ(3u, ids.lower_bound_index(104));
    EXPECT_EQ(5u, ids.lower_bound_index(110));
}

TEST(compact_ids_bitset, iterator_visits_ids_in_order) {
    CompactIdsBitset ids;
    const std::vector<uint64_t> values = {100, 101, 103, 108, 109};
    ASSERT_EQ(0, ids.init(values).code());

    auto it = ids.begin();
    for (size_t i = 0; i < values.size(); ++i) {
        ASSERT_FALSE(it.eof());
        EXPECT_EQ(i, it.index());
        EXPECT_EQ(values[i], it.id());
        it.next();
    }

    EXPECT_TRUE(it.eof());
    EXPECT_THROW(it.id(), std::out_of_range);
    EXPECT_THROW(it.index(), std::out_of_range);
}

TEST(compact_ids_bitset, write_and_map_round_trip_preserves_values) {
    CompactIdsBitset ids;
    ASSERT_EQ(0, ids.init(std::vector<uint64_t>{100, 101, 103, 108, 109}).code());

    const std::vector<uint8_t> serialized = serialize_ids_to_bytes(ids);
    const CompactIdsHeader hdr = read_header_or_die(serialized);
    EXPECT_EQ(static_cast<uint8_t>(CompactIdsExtEncoding::Bitset), hdr.encoding);
    EXPECT_EQ(5u, hdr.count);
    EXPECT_EQ(9u, hdr.aux_data);
    EXPECT_EQ(2u, hdr.payload_size);
    EXPECT_EQ(100u, hdr.base);

    CompactIdsBitset mapped;
    size_t consumed = 0;
    ASSERT_EQ(0, mapped.map(serialized.data(), serialized.size(), &consumed).code());
    EXPECT_EQ(serialized.size(), consumed);
    EXPECT_EQ(ids.count(), mapped.count());
    for (size_t i = 0; i < ids.count(); ++i) {
        EXPECT_EQ(ids.id(i), mapped.id(i));
    }
}

TEST(compact_ids_bitset, map_rejects_truncated_payload) {
    CompactIdsHeader hdr{};
    hdr.encoding = static_cast<uint8_t>(CompactIdsExtEncoding::Bitset);
    hdr.count = 3;
    hdr.aux_data = 8;
    hdr.payload_size = 2;
    hdr.base = 100;

    std::vector<uint8_t> serialized(sizeof(hdr) + 1, 0);
    std::memcpy(serialized.data(), &hdr, sizeof(hdr));

    CompactIdsBitset ids;
    const Ret ret = ids.map(serialized.data(), serialized.size(), nullptr);

    ASSERT_NE(0, ret.code());
    EXPECT_EQ("CompactIdsBitset::map: truncated payload", ret.message());
}

TEST(compact_ids_bitset, map_rejects_non_zero_tail_bits) {
    CompactIdsHeader hdr{};
    hdr.encoding = static_cast<uint8_t>(CompactIdsExtEncoding::Bitset);
    hdr.count = 2;
    hdr.aux_data = 4;
    hdr.payload_size = 1;
    hdr.base = 100;

    std::vector<uint8_t> serialized(sizeof(hdr) + 1, 0);
    std::memcpy(serialized.data(), &hdr, sizeof(hdr));
    serialized[sizeof(hdr)] = 0xA1;

    CompactIdsBitset ids;
    const Ret ret = ids.map(serialized.data(), serialized.size(), nullptr);

    ASSERT_NE(0, ret.code());
    EXPECT_EQ("CompactIdsBitset::map: malformed bitset payload tail bits", ret.message());
}

TEST(compact_ids_bitset, map_rejects_count_mismatch) {
    CompactIdsHeader hdr{};
    hdr.encoding = static_cast<uint8_t>(CompactIdsExtEncoding::Bitset);
    hdr.count = 3;
    hdr.aux_data = 4;
    hdr.payload_size = 1;
    hdr.base = 100;

    std::vector<uint8_t> serialized(sizeof(hdr) + 1, 0);
    std::memcpy(serialized.data(), &hdr, sizeof(hdr));
    serialized[sizeof(hdr)] = 0x11;

    CompactIdsBitset ids;
    const Ret ret = ids.map(serialized.data(), serialized.size(), nullptr);

    ASSERT_NE(0, ret.code());
    EXPECT_EQ("CompactIdsBitset::map: bitset count does not match header", ret.message());
}

TEST(compact_ids_bitset, init_rejects_unsorted_ids) {
    CompactIdsBitset ids;

    const Ret ret = ids.init(std::vector<uint64_t>{100, 100});

    ASSERT_NE(0, ret.code());
    EXPECT_EQ("CompactIdsBitset::init: ids must be strictly increasing", ret.message());
}

TEST(compact_ids_bitset, init_rejects_range_larger_than_uint32) {
    CompactIdsBitset ids;

    const Ret ret = ids.init(
        std::vector<uint64_t>{5, 5 + static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1u});

    ASSERT_NE(0, ret.code());
    EXPECT_EQ("CompactIdsBitset::init: id range exceeds uint32_t", ret.message());
}

TEST(compact_ids_accumulator, add_rejects_offsets_larger_than_uint32) {
    CompactIdsAccumulator ids;
    ids.init(5, 2);

    EXPECT_THROW(
        {
            ids.add(5 + static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1u);
        },
        std::out_of_range);
}

TEST(compact_ids_misses, map_rejects_unsorted_misses) {
    CompactIdsMisses ids;
    const std::vector<uint8_t> serialized = serialize_misses_to_bytes(100, 3, {3, 1});

    const Ret ret = ids.map(serialized.data(), serialized.size(), nullptr);

    ASSERT_NE(0, ret.code());
    EXPECT_EQ("CompactIdsMisses::map: misses must be strictly increasing", ret.message());
}

TEST(compact_ids_misses, map_rejects_duplicate_misses) {
    CompactIdsMisses ids;
    const std::vector<uint8_t> serialized = serialize_misses_to_bytes(100, 3, {1, 1});

    const Ret ret = ids.map(serialized.data(), serialized.size(), nullptr);

    ASSERT_NE(0, ret.code());
    EXPECT_EQ("CompactIdsMisses::map: misses must be strictly increasing", ret.message());
}

TEST(compact_ids_misses, map_rejects_boundary_misses) {
    CompactIdsMisses ids;
    const std::vector<uint8_t> leading_miss = serialize_misses_to_bytes(100, 3, {0});
    const Ret leading_ret = ids.map(leading_miss.data(), leading_miss.size(), nullptr);
    ASSERT_NE(0, leading_ret.code());
    EXPECT_EQ("CompactIdsMisses::map: misses must stay within the interior span", leading_ret.message());

    const std::vector<uint8_t> trailing_miss = serialize_misses_to_bytes(100, 3, {3});
    const Ret trailing_ret = ids.map(trailing_miss.data(), trailing_miss.size(), nullptr);
    ASSERT_NE(0, trailing_ret.code());
    EXPECT_EQ("CompactIdsMisses::map: misses must stay within the interior span", trailing_ret.message());
}

TEST(compact_ids_misses, map_rejects_range_larger_than_uint32) {
    CompactIdsMisses ids;
    const std::vector<uint8_t> serialized = serialize_misses_to_bytes(
        5,
        std::numeric_limits<uint32_t>::max(),
        {1, 2});

    const Ret ret = ids.map(serialized.data(), serialized.size(), nullptr);

    ASSERT_NE(0, ret.code());
    EXPECT_EQ("CompactIdsMisses::map: id range exceeds uint32_t", ret.message());
}

TEST(compact_ids_misses, map_rejects_uint64_overflow) {
    CompactIdsMisses ids;
    const std::vector<uint8_t> serialized = serialize_misses_to_bytes(
        std::numeric_limits<uint64_t>::max() - 1u,
        2,
        {1});

    const Ret ret = ids.map(serialized.data(), serialized.size(), nullptr);

    ASSERT_NE(0, ret.code());
    EXPECT_EQ("CompactIdsMisses::map: base plus id span overflows uint64_t", ret.message());
}

TEST(compact_ids_misses, map_accepts_unaligned_misses_payload) {
    CompactIdsMisses ids;
    ASSERT_EQ(0, ids.init(std::vector<uint64_t>{100, 101, 102, 104, 105, 106}).code());

    const std::vector<uint8_t> serialized = serialize_misses_to_bytes(100, 6, {3});
    const std::vector<uint8_t> unaligned = make_unaligned_buffer(serialized);

    CompactIdsMisses mapped;
    size_t consumed = 0;
    ASSERT_EQ(0, mapped.map(unaligned.data() + 1, serialized.size(), &consumed).code());
    EXPECT_EQ(serialized.size(), consumed);
    EXPECT_EQ(ids.count(), mapped.count());
    for (size_t i = 0; i < ids.count(); ++i) {
        EXPECT_EQ(ids.id(i), mapped.id(i));
    }
}

TEST(compact_ids_ext_auto, init_prefers_miss_list_for_sparse_ids) {
    CompactIdsExt ids;
    std::vector<uint64_t> values = {100, 200, 300};
    ASSERT_EQ(0, ids.init(values).code());

    const std::vector<uint8_t> serialized = serialize_auto_ids_to_bytes(ids);
    const CompactIdsHeader hdr = read_header_or_die(serialized);
    EXPECT_EQ(static_cast<uint8_t>(CompactIdsExtEncoding::Offsets32), hdr.encoding);
    EXPECT_EQ(3u, ids.count());
}

TEST(compact_ids_ext_auto, init_from_accumulator_prefers_offsets_for_sparse_ids) {
    CompactIdsExt ids;
    CompactIdsAccumulator accumulator = make_accumulator({100, 200, 300});

    ASSERT_EQ(0, ids.init(accumulator).code());

    const std::vector<uint8_t> serialized = serialize_auto_ids_to_bytes(ids);
    const CompactIdsHeader hdr = read_header_or_die(serialized);
    EXPECT_EQ(static_cast<uint8_t>(CompactIdsExtEncoding::Offsets32), hdr.encoding);
    EXPECT_EQ(3u, ids.count());
    EXPECT_EQ(1u, ids.lower_bound_index(150));
    EXPECT_EQ(2u, ids.lower_bound_index(201));
}

TEST(compact_ids_ext_auto, init_prefers_bitset_for_dense_ids) {
    CompactIdsExt ids;
    std::vector<uint64_t> values = {100, 101, 103, 108, 109};
    ASSERT_EQ(0, ids.init(values).code());

    const std::vector<uint8_t> serialized = serialize_auto_ids_to_bytes(ids);
    const CompactIdsHeader hdr = read_header_or_die(serialized);
    EXPECT_EQ(static_cast<uint8_t>(CompactIdsExtEncoding::Bitset), hdr.encoding);
    EXPECT_EQ(5u, ids.count());
}

TEST(compact_ids_ext_auto, init_from_accumulator_prefers_bitset_for_dense_ids) {
    CompactIdsExt ids;
    CompactIdsAccumulator accumulator = make_accumulator({100, 101, 103, 108, 109});

    ASSERT_EQ(0, ids.init(accumulator).code());

    const std::vector<uint8_t> serialized = serialize_auto_ids_to_bytes(ids);
    const CompactIdsHeader hdr = read_header_or_die(serialized);
    EXPECT_EQ(static_cast<uint8_t>(CompactIdsExtEncoding::Bitset), hdr.encoding);
    EXPECT_EQ(5u, ids.count());
    EXPECT_EQ(3u, ids.lower_bound_index(104));
    EXPECT_EQ(108u, ids.id(3));
}

TEST(compact_ids_ext_auto, init_from_accumulator_prefers_misses_for_dense_interior_gap) {
    CompactIdsExt ids;
    std::vector<uint64_t> values;
    values.reserve(40);
    for (uint64_t id = 100; id <= 140; ++id) {
        if (id != 120) {
            values.push_back(id);
        }
    }
    CompactIdsAccumulator accumulator = make_accumulator(values);

    ASSERT_EQ(0, ids.init(accumulator).code());

    const std::vector<uint8_t> serialized = serialize_auto_ids_to_bytes(ids);
    const CompactIdsHeader hdr = read_header_or_die(serialized);
    EXPECT_EQ(static_cast<uint8_t>(CompactIdsExtEncoding::Misses32), hdr.encoding);
    EXPECT_EQ(values.size(), ids.count());
    EXPECT_EQ(20u, ids.lower_bound_index(120));
    EXPECT_EQ(121u, ids.id(20));
}

TEST(compact_ids_ext_auto, map_dispatches_by_encoding) {
    CompactIdsExt sparse_ids;
    std::vector<uint64_t> sparse_values = {100, 200, 300};
    ASSERT_EQ(0, sparse_ids.init(sparse_values).code());
    const std::vector<uint8_t> sparse_serialized = serialize_auto_ids_to_bytes(sparse_ids);

    CompactIdsExt misses_ids;
    std::vector<uint64_t> misses_values = {100, 101, 102, 104, 105, 106};
    ASSERT_EQ(0, misses_ids.init(misses_values).code());
    const std::vector<uint8_t> misses_serialized = serialize_auto_ids_to_bytes(misses_ids);

    CompactIdsExt dense_ids;
    std::vector<uint64_t> dense_values = {100, 101, 103, 108, 109};
    ASSERT_EQ(0, dense_ids.init(dense_values).code());
    const std::vector<uint8_t> dense_serialized = serialize_auto_ids_to_bytes(dense_ids);

    CompactIdsExt sparse_mapped;
    size_t sparse_consumed = 0;
    ASSERT_EQ(0, sparse_mapped.map(sparse_serialized.data(), sparse_serialized.size(), &sparse_consumed).code());
    EXPECT_EQ(sparse_serialized.size(), sparse_consumed);
    EXPECT_EQ(1u, sparse_mapped.lower_bound_index(150));
    EXPECT_EQ(1u, sparse_mapped.lower_bound_index(200));
    EXPECT_EQ(2u, sparse_mapped.lower_bound_index(201));

    CompactIdsExt misses_mapped;
    size_t misses_consumed = 0;
    ASSERT_EQ(0, misses_mapped.map(misses_serialized.data(), misses_serialized.size(), &misses_consumed).code());
    EXPECT_EQ(misses_serialized.size(), misses_consumed);
    EXPECT_EQ(3u, misses_mapped.lower_bound_index(103));
    EXPECT_EQ(3u, misses_mapped.lower_bound_index(104));
    EXPECT_EQ(3u, misses_mapped.lower_bound_index(103));

    CompactIdsExt dense_mapped;
    size_t dense_consumed = 0;
    ASSERT_EQ(0, dense_mapped.map(dense_serialized.data(), dense_serialized.size(), &dense_consumed).code());
    EXPECT_EQ(dense_serialized.size(), dense_consumed);
    EXPECT_EQ(3u, dense_mapped.lower_bound_index(104));
    EXPECT_EQ(3u, dense_mapped.lower_bound_index(108));
    EXPECT_EQ(3u, dense_mapped.lower_bound_index(104));
}

TEST(compact_ids_ext_auto, map_rejects_unknown_encoding) {
    CompactIdsHeader hdr{};
    hdr.encoding = 99;

    std::vector<uint8_t> serialized(sizeof(hdr), 0);
    std::memcpy(serialized.data(), &hdr, sizeof(hdr));

    CompactIdsExt ids;
    const Ret ret = ids.map(serialized.data(), serialized.size(), nullptr);

    ASSERT_NE(0, ret.code());
    EXPECT_EQ("CompactIdsExt::map: unknown encoding", ret.message());
}

TEST(compact_ids_ext_auto, clear_resets_to_empty_offsets_state) {
    CompactIdsExt ids;
    std::vector<uint64_t> values = {100, 101, 102, 104, 105, 106};
    ASSERT_EQ(0, ids.init(values).code());

    ids.clear();

    EXPECT_TRUE(ids.empty());
    EXPECT_EQ(0u, ids.count());

    const std::vector<uint8_t> serialized = serialize_auto_ids_to_bytes(ids);
    const CompactIdsHeader hdr = read_header_or_die(serialized);
    EXPECT_EQ(static_cast<uint8_t>(CompactIdsExtEncoding::Offsets32), hdr.encoding);
    EXPECT_EQ(0u, hdr.count);
}

} // namespace sketch2
