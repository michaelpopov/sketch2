// Unit tests for the CompactIds utility.

#include "compact_ids.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <string>
#include <limits>
#include <vector>

#include "utest_tmp_dir.h"

namespace sketch2 {

namespace {

struct CompactIdsHeaderForTest {
    uint8_t encoding = 0;
    uint8_t reserved0 = 0;
    uint16_t reserved1 = 0;
    uint32_t count = 0;
    uint32_t max_offset = 0;
    uint32_t payload_size = 0;
    uint64_t base = 0;
};

static_assert(sizeof(CompactIdsHeaderForTest) == 24, "Unexpected test header size");

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

std::vector<uint8_t> serialize_ids_to_bytes(const CompactIds& ids) {
    const std::string path = tmp_dir() + "/sketch2_utest_compact_ids_" + std::to_string(getpid()) + ".bin";
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

std::vector<uint8_t> serialize_offsets_to_bytes(uint64_t base, const std::vector<uint32_t>& offsets) {
    CompactIdsHeaderForTest hdr{};
    hdr.encoding = static_cast<uint8_t>(CompactIdsEncoding::Offsets32);
    hdr.count = static_cast<uint32_t>(offsets.size());
    hdr.max_offset = offsets.empty() ? 0u : offsets.back();
    hdr.payload_size = static_cast<uint32_t>(offsets.size() * sizeof(uint32_t));
    hdr.base = offsets.empty() ? 0u : base;

    std::vector<uint8_t> bytes(sizeof(hdr) + hdr.payload_size, 0);
    std::memcpy(bytes.data(), &hdr, sizeof(hdr));
    if (!offsets.empty()) {
        std::memcpy(bytes.data() + sizeof(hdr), offsets.data(), hdr.payload_size);
    }
    return bytes;
}

std::vector<uint8_t> serialize_bitset_to_bytes(uint64_t base, uint32_t count, uint32_t max_offset,
        const std::vector<uint8_t>& payload) {
    CompactIdsHeaderForTest hdr{};
    hdr.encoding = static_cast<uint8_t>(CompactIdsEncoding::Bitset);
    hdr.count = count;
    hdr.max_offset = max_offset;
    hdr.payload_size = static_cast<uint32_t>(payload.size());
    hdr.base = count == 0 ? 0u : base;

    std::vector<uint8_t> bytes(sizeof(hdr) + payload.size(), 0);
    std::memcpy(bytes.data(), &hdr, sizeof(hdr));
    if (!payload.empty()) {
        std::memcpy(bytes.data() + sizeof(hdr), payload.data(), payload.size());
    }
    return bytes;
}

CompactIds read_ids_from_offsets_or_die(uint64_t base, const std::vector<uint32_t>& offsets) {
    CompactIds ids;
    const std::vector<uint8_t> serialized = serialize_offsets_to_bytes(base, offsets);
    size_t consumed = 0;
    EXPECT_EQ(0, ids.read(serialized.data(), serialized.size(), &consumed).code());
    EXPECT_EQ(serialized.size(), consumed);
    return ids;
}

} // namespace

TEST(compact_ids, init_empty_container) {
    CompactIds ids;

    ASSERT_EQ(0, ids.init(std::vector<uint64_t>{}).code());
    EXPECT_TRUE(ids.empty());
    EXPECT_EQ(0u, ids.count());
    EXPECT_EQ(0u, ids.base());
    EXPECT_EQ(0u, ids.lower_bound_index(123));
}

TEST(compact_ids_builder, append_preserves_values_and_metadata) {
    CompactIdsBuilder ids;
    ids.reserve(4);

    ASSERT_EQ(0, ids.append(20000).code());
    ASSERT_EQ(0, ids.append(20005).code());
    ASSERT_EQ(0, ids.append(20011).code());
    ASSERT_EQ(0, ids.append(29999).code());

    EXPECT_EQ(4u, ids.count());
    EXPECT_EQ(20000u, ids.base());
    EXPECT_EQ(20000u, ids.min_id());
    EXPECT_EQ(29999u, ids.max_id());
    EXPECT_EQ(CompactIdsEncoding::Offsets32, ids.preferred_encoding());
}

TEST(compact_ids_builder, append_rejects_unsorted_ids) {
    CompactIdsBuilder ids;

    ASSERT_EQ(0, ids.append(100).code());
    const Ret ret = ids.append(100);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::init: ids must be strictly increasing", ret.message());
}

TEST(compact_ids_builder, append_rejects_offsets_larger_than_uint32) {
    CompactIdsBuilder ids;

    ASSERT_EQ(0, ids.append(5).code());
    const Ret ret = ids.append(5 + static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1u);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::init: id offset exceeds uint32_t range", ret.message());
}

TEST(compact_ids, init_from_sorted_ids_preserves_values) {
    CompactIds ids;
    const std::vector<uint64_t> values = {20000, 20005, 20011, 29999};

    ASSERT_EQ(0, ids.init(values).code());
    EXPECT_EQ(20000u, ids.base());
    EXPECT_EQ(values.size(), ids.count());
    EXPECT_EQ(20000u, ids.min_id());
    EXPECT_EQ(29999u, ids.max_id());
    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(values[i], ids.id(i));
    }
}

TEST(compact_ids, clear_resets_container) {
    CompactIds ids;
    ASSERT_EQ(0, ids.init(std::vector<uint64_t>{100, 101, 105}).code());

    ids.clear();

    EXPECT_TRUE(ids.empty());
    EXPECT_EQ(0u, ids.count());
    EXPECT_EQ(0u, ids.base());
}

TEST(compact_ids, init_rejects_duplicate_ids) {
    CompactIds ids;
    const std::vector<uint64_t> values = {10, 11, 11};

    const Ret ret = ids.init(values);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::init: ids must be strictly increasing", ret.message());
}

TEST(compact_ids, init_rejects_unsorted_ids) {
    CompactIds ids;
    const std::vector<uint64_t> values = {10, 12, 11};

    const Ret ret = ids.init(values);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::init: ids must be strictly increasing", ret.message());
}

TEST(compact_ids, init_rejects_offsets_larger_than_uint32) {
    CompactIds ids;
    const std::vector<uint64_t> values = {
        5,
        5 + static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1u,
    };

    const Ret ret = ids.init(values);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::init: id offset exceeds uint32_t range", ret.message());
}

TEST(compact_ids, lower_bound_index_and_index_of_match_sorted_semantics) {
    CompactIds ids;
    ASSERT_EQ(0, ids.init(std::vector<uint64_t>{20, 25, 40, 45}).code());

    EXPECT_EQ(0u, ids.lower_bound_index(10));
    EXPECT_EQ(0u, ids.lower_bound_index(20));
    EXPECT_EQ(1u, ids.lower_bound_index(21));
    EXPECT_EQ(2u, ids.lower_bound_index(40));
    EXPECT_EQ(4u, ids.lower_bound_index(100));

    EXPECT_EQ(CompactIds::npos, ids.index_of(10));
    EXPECT_EQ(0u, ids.index_of(20));
    EXPECT_EQ(CompactIds::npos, ids.index_of(21));
    EXPECT_EQ(3u, ids.index_of(45));
    EXPECT_EQ(CompactIds::npos, ids.index_of(100));
}

TEST(compact_ids, lower_bound_index_with_nonzero_first_offset_matches_sorted_semantics) {
    CompactIds ids = read_ids_from_offsets_or_die(100, std::vector<uint32_t>{5, 10});

    EXPECT_EQ(0u, ids.lower_bound_index(100));
    EXPECT_EQ(0u, ids.lower_bound_index(104));
    EXPECT_EQ(0u, ids.lower_bound_index(105));
    EXPECT_EQ(1u, ids.lower_bound_index(106));

    EXPECT_EQ(CompactIds::npos, ids.index_of(100));
    EXPECT_EQ(CompactIds::npos, ids.index_of(104));
    EXPECT_EQ(0u, ids.index_of(105));
    EXPECT_EQ(CompactIds::npos, ids.index_of(106));
}

TEST(compact_ids, contains_reports_membership) {
    CompactIds ids;
    ASSERT_EQ(0, ids.init(std::vector<uint64_t>{20000, 20005, 20011, 29999}).code());

    EXPECT_TRUE(ids.contains(20000));
    EXPECT_TRUE(ids.contains(20005));
    EXPECT_FALSE(ids.contains(20006));
    EXPECT_TRUE(ids.contains(29999));
    EXPECT_FALSE(ids.contains(30000));
}

TEST(compact_ids, init_from_base_and_offsets_preserves_values) {
    const std::vector<uint32_t> offsets = {0, 5, 11, 9999};
    CompactIds ids = read_ids_from_offsets_or_die(20000, offsets);
    EXPECT_EQ(20000u, ids.base());
    EXPECT_EQ(offsets.size(), ids.count());
    EXPECT_EQ(0u, ids.offset(0));
    EXPECT_EQ(5u, ids.offset(1));
    EXPECT_EQ(9999u, ids.max_offset());
    EXPECT_EQ(20000u, ids.id(0));
    EXPECT_EQ(20005u, ids.id(1));
    EXPECT_EQ(29999u, ids.id(3));
}

TEST(compact_ids, min_id_uses_first_offset_for_base_plus_offsets_init) {
    CompactIds ids = read_ids_from_offsets_or_die(100, std::vector<uint32_t>{5, 10});

    EXPECT_EQ(105u, ids.min_id());
    EXPECT_EQ(105u, ids.id(0));
    EXPECT_EQ(110u, ids.id(1));
}

TEST(compact_ids, preferred_encoding_uses_offsets_for_sparse_layouts) {
    CompactIds ids;
    ASSERT_EQ(0, ids.init(std::vector<uint64_t>{0, 1000, 2000}).code());

    EXPECT_EQ(CompactIdsEncoding::Offsets32, ids.preferred_encoding());
    EXPECT_EQ(3u * sizeof(uint32_t), ids.offsets_storage_size_bytes());
    EXPECT_EQ(251u, ids.bitset_storage_size_bytes());
}

TEST(compact_ids, preferred_encoding_uses_bitset_for_dense_layouts) {
    std::vector<uint64_t> values;
    values.reserve(9000);
    for (uint64_t id = 20000; id < 29000; ++id) {
        values.push_back(id);
    }

    CompactIds ids;
    ASSERT_EQ(0, ids.init(values).code());

    EXPECT_EQ(CompactIdsEncoding::Bitset, ids.preferred_encoding());
    EXPECT_EQ(9000u * sizeof(uint32_t), ids.offsets_storage_size_bytes());
    EXPECT_EQ(1125u, ids.bitset_storage_size_bytes());
}

TEST(compact_ids, init_from_base_and_offsets_rejects_duplicate_offsets) {
    CompactIds ids;
    const std::vector<uint8_t> serialized = serialize_offsets_to_bytes(100, std::vector<uint32_t>{0, 5, 5});
    const Ret ret = ids.read(serialized.data(), serialized.size(), nullptr);

    ASSERT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::init: offsets must be strictly increasing", ret.message());
}

TEST(compact_ids, init_from_base_and_offsets_rejects_unsorted_offsets) {
    CompactIds ids;
    const std::vector<uint8_t> serialized = serialize_offsets_to_bytes(100, std::vector<uint32_t>{0, 7, 6});
    const Ret ret = ids.read(serialized.data(), serialized.size(), nullptr);

    ASSERT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::init: offsets must be strictly increasing", ret.message());
}

TEST(compact_ids, init_from_base_and_offsets_rejects_uint64_overflow) {
    CompactIds ids;
    const std::vector<uint8_t> serialized =
        serialize_offsets_to_bytes(std::numeric_limits<uint64_t>::max() - 5u, std::vector<uint32_t>{0, 10});
    const Ret ret = ids.read(serialized.data(), serialized.size(), nullptr);

    ASSERT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::init: base plus offset overflows uint64_t", ret.message());
}

TEST(compact_ids, id_out_of_range_throws) {
    CompactIds ids;
    ASSERT_EQ(0, ids.init(std::vector<uint64_t>{50, 60}).code());

    EXPECT_THROW(ids.id(2), std::out_of_range);
    EXPECT_THROW(ids.id(100), std::out_of_range);
}

TEST(compact_ids, offset_out_of_range_throws) {
    CompactIds ids;
    ASSERT_EQ(0, ids.init(std::vector<uint64_t>{50, 60}).code());

    EXPECT_THROW(ids.offset(2), std::out_of_range);
    EXPECT_THROW(ids.offset(100), std::out_of_range);
}

TEST(compact_ids, min_and_max_throw_for_empty_container) {
    CompactIds ids;

    EXPECT_THROW(ids.min_id(), std::out_of_range);
    EXPECT_THROW(ids.max_id(), std::out_of_range);
    EXPECT_THROW(ids.max_offset(), std::out_of_range);
}

TEST(compact_ids, iterator_walks_ids_in_order) {
    CompactIds ids;
    ASSERT_EQ(0, ids.init(std::vector<uint64_t>{7, 9, 15}).code());

    auto it = ids.begin();
    ASSERT_FALSE(it.eof());
    EXPECT_EQ(0u, it.index());
    EXPECT_EQ(7u, it.id());

    it.next();
    ASSERT_FALSE(it.eof());
    EXPECT_EQ(1u, it.index());
    EXPECT_EQ(9u, it.id());

    it.next();
    ASSERT_FALSE(it.eof());
    EXPECT_EQ(2u, it.index());
    EXPECT_EQ(15u, it.id());

    it.next();
    EXPECT_TRUE(it.eof());
}

TEST(compact_ids, iterator_on_empty_container_is_eof) {
    CompactIds ids;
    ASSERT_EQ(0, ids.init(std::vector<uint64_t>{}).code());

    EXPECT_TRUE(ids.begin().eof());
}

TEST(compact_ids, iterator_access_after_eof_throws) {
    CompactIds ids;
    ASSERT_EQ(0, ids.init(std::vector<uint64_t>{42}).code());

    auto it = ids.begin();
    it.next();

    EXPECT_TRUE(it.eof());
    EXPECT_THROW(it.id(), std::out_of_range);
    EXPECT_THROW(it.index(), std::out_of_range);
}

TEST(compact_ids, serialized_size_matches_preferred_encoding_payload) {
    CompactIds ids;
    ASSERT_EQ(0, ids.init(std::vector<uint64_t>{10, 1000, 2000}).code());

    EXPECT_EQ(CompactIdsEncoding::Offsets32, ids.preferred_encoding());
    EXPECT_EQ(24u + 3u * sizeof(uint32_t), ids.serialized_size_bytes());
}

TEST(compact_ids, write_offsets_encoding_serializes_uint32_offsets_payload) {
    CompactIds ids;
    ASSERT_EQ(0, ids.init(std::vector<uint64_t>{20000, 20005, 20011, 29999}).code());
    ASSERT_EQ(CompactIdsEncoding::Offsets32, ids.preferred_encoding());

    const std::vector<uint8_t> serialized = serialize_ids_to_bytes(ids);
    ASSERT_EQ(sizeof(CompactIdsHeaderForTest) + ids.count() * sizeof(uint32_t), serialized.size());

    CompactIdsHeaderForTest hdr{};
    std::memcpy(&hdr, serialized.data(), sizeof(hdr));
    EXPECT_EQ(static_cast<uint8_t>(CompactIdsEncoding::Offsets32), hdr.encoding);

    std::vector<uint32_t> offsets(ids.count());
    std::memcpy(offsets.data(), serialized.data() + sizeof(hdr), offsets.size() * sizeof(uint32_t));

    EXPECT_EQ((std::vector<uint32_t>{0u, 5u, 11u, 9999u}), offsets);
}

TEST(compact_ids, write_and_read_round_trip_offsets_encoding) {
    CompactIds ids;
    ASSERT_EQ(0, ids.init(std::vector<uint64_t>{10, 1000, 2000}).code());
    ASSERT_EQ(CompactIdsEncoding::Offsets32, ids.preferred_encoding());

    const std::vector<uint8_t> serialized = serialize_ids_to_bytes(ids);
    ASSERT_FALSE(serialized.empty());

    CompactIds restored;
    size_t consumed = 0;
    ASSERT_EQ(0, restored.read(serialized.data(), serialized.size(), &consumed).code());
    ASSERT_EQ(serialized.size(), consumed);

    EXPECT_EQ(ids.base(), restored.base());
    EXPECT_EQ(ids.count(), restored.count());
    for (size_t i = 0; i < ids.count(); ++i) {
        EXPECT_EQ(ids.id(i), restored.id(i));
        EXPECT_EQ(ids.offset(i), restored.offset(i));
    }
}

TEST(compact_ids, write_and_read_round_trip_bitset_encoding) {
    std::vector<uint64_t> values;
    values.reserve(9000);
    for (uint64_t id = 20000; id < 29000; ++id) {
        values.push_back(id);
    }

    CompactIds ids;
    ASSERT_EQ(0, ids.init(values).code());
    ASSERT_EQ(CompactIdsEncoding::Bitset, ids.preferred_encoding());

    const std::vector<uint8_t> serialized = serialize_ids_to_bytes(ids);
    ASSERT_FALSE(serialized.empty());

    CompactIds restored;
    size_t consumed = 0;
    ASSERT_EQ(0, restored.read(serialized.data(), serialized.size(), &consumed).code());
    ASSERT_EQ(serialized.size(), consumed);

    EXPECT_EQ(ids.base(), restored.base());
    EXPECT_EQ(ids.count(), restored.count());
    for (size_t i = 0; i < ids.count(); ++i) {
        EXPECT_EQ(ids.id(i), restored.id(i));
    }
}

TEST(compact_ids, write_and_read_round_trip_empty_container) {
    CompactIds ids;
    ASSERT_EQ(0, ids.init(std::vector<uint64_t>{}).code());
    ASSERT_TRUE(ids.empty());

    const std::vector<uint8_t> serialized = serialize_ids_to_bytes(ids);
    ASSERT_FALSE(serialized.empty());

    CompactIds restored;
    size_t consumed = 0;
    ASSERT_EQ(0, restored.read(serialized.data(), serialized.size(), &consumed).code());
    ASSERT_EQ(serialized.size(), consumed);

    EXPECT_TRUE(restored.empty());
    EXPECT_EQ(0u, restored.count());
    EXPECT_EQ(0u, restored.base());
}

TEST(compact_ids, write_rejects_null_file_handle) {
    CompactIds ids;
    ASSERT_EQ(0, ids.init(std::vector<uint64_t>{1, 2, 3}).code());

    const Ret ret = ids.write(nullptr, "write failed");

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("write failed: file handle is null", ret.message());
}

TEST(compact_ids, read_rejects_null_buffer) {
    CompactIds ids;
    size_t consumed = 123;

    const Ret ret = ids.read(nullptr, 0, &consumed);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::read: data pointer is null", ret.message());
    EXPECT_EQ(0u, consumed);
}

TEST(compact_ids, read_rejects_truncated_header) {
    const uint8_t short_header[8] = {0};

    CompactIds ids;
    size_t consumed = 123;
    const Ret ret = ids.read(short_header, sizeof(short_header), &consumed);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::read: buffer too small to contain header", ret.message());
    EXPECT_EQ(0u, consumed);
}

TEST(compact_ids, read_rejects_unknown_encoding) {
    CompactIdsHeaderForTest hdr{};
    hdr.encoding = 99;
    std::vector<uint8_t> serialized(sizeof(hdr), 0);
    std::memcpy(serialized.data(), &hdr, sizeof(hdr));

    CompactIds ids;
    size_t consumed = 123;
    const Ret ret = ids.read(serialized.data(), serialized.size(), &consumed);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::read: unknown encoding", ret.message());
    EXPECT_EQ(0u, consumed);
}

TEST(compact_ids, read_rejects_malformed_empty_header) {
    CompactIdsHeaderForTest hdr{};
    hdr.encoding = static_cast<uint8_t>(CompactIdsEncoding::Offsets32);
    hdr.count = 0;
    hdr.base = 123; // illegal for empty payload path
    std::vector<uint8_t> serialized(sizeof(hdr), 0);
    std::memcpy(serialized.data(), &hdr, sizeof(hdr));

    CompactIds ids;
    size_t consumed = 123;
    const Ret ret = ids.read(serialized.data(), serialized.size(), &consumed);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::read: malformed empty payload header", ret.message());
    EXPECT_EQ(0u, consumed);
}

TEST(compact_ids, read_rejects_offsets_payload_size_mismatch) {
    CompactIdsHeaderForTest hdr{};
    hdr.encoding = static_cast<uint8_t>(CompactIdsEncoding::Offsets32);
    hdr.count = 2;
    hdr.payload_size = 4; // should be 8
    hdr.base = 100;
    std::vector<uint8_t> serialized(sizeof(hdr) + hdr.payload_size, 0);
    std::memcpy(serialized.data(), &hdr, sizeof(hdr));

    CompactIds ids;
    size_t consumed = 123;
    const Ret ret = ids.read(serialized.data(), serialized.size(), &consumed);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::read: malformed offsets payload size", ret.message());
    EXPECT_EQ(0u, consumed);
}

TEST(compact_ids, read_rejects_bitset_payload_size_mismatch) {
    CompactIdsHeaderForTest hdr{};
    hdr.encoding = static_cast<uint8_t>(CompactIdsEncoding::Bitset);
    hdr.count = 3;
    hdr.max_offset = 31;    // expected bitset payload is 4 bytes
    hdr.payload_size = 3;   // mismatch
    hdr.base = 200;
    std::vector<uint8_t> serialized(sizeof(hdr) + hdr.payload_size, 0);
    std::memcpy(serialized.data(), &hdr, sizeof(hdr));

    CompactIds ids;
    size_t consumed = 123;
    const Ret ret = ids.read(serialized.data(), serialized.size(), &consumed);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::read: malformed bitset payload size", ret.message());
    EXPECT_EQ(0u, consumed);
}

TEST(compact_ids, read_rejects_bitset_payload_with_nonzero_tail_bits) {
    CompactIdsHeaderForTest hdr{};
    hdr.encoding = static_cast<uint8_t>(CompactIdsEncoding::Bitset);
    hdr.count = 1;
    hdr.max_offset = 5;   // valid bits are [0..5], bits 6-7 must be zero
    hdr.payload_size = 1; // one byte payload
    hdr.base = 100;
    std::vector<uint8_t> serialized(sizeof(hdr) + 1, 0);
    std::memcpy(serialized.data(), &hdr, sizeof(hdr));
    const uint8_t malformed_payload = 0x41; // bit 0 set (valid), bit 6 set (invalid tail bit)
    serialized[sizeof(hdr)] = malformed_payload;

    CompactIds ids;
    size_t consumed = 123;
    const Ret ret = ids.read(serialized.data(), serialized.size(), &consumed);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::read: malformed bitset payload tail bits", ret.message());
    EXPECT_EQ(0u, consumed);
}

TEST(compact_ids, read_rejects_non_canonical_sparse_bitset_encoding) {
    // A canonical writer would store this as one uint32 offset, not a 126-byte bitset.
    const std::vector<uint8_t> serialized = serialize_bitset_to_bytes(
        100,
        1,
        1000,
        std::vector<uint8_t>(126, 0));
    std::vector<uint8_t> sparse = serialized;
    sparse[sizeof(CompactIdsHeaderForTest)] = 0x01;

    CompactIds ids;
    size_t consumed = 123;
    const Ret ret = ids.read(sparse.data(), sparse.size(), &consumed);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::read: non-canonical bitset encoding", ret.message());
    EXPECT_EQ(0u, consumed);
}

TEST(compact_ids, read_rejects_truncated_offsets_payload) {
    CompactIdsHeaderForTest hdr{};
    hdr.encoding = static_cast<uint8_t>(CompactIdsEncoding::Offsets32);
    hdr.count = 2;
    hdr.payload_size = 8;
    hdr.base = 100;
    std::vector<uint8_t> truncated(sizeof(hdr) + sizeof(uint32_t), 0);
    std::memcpy(truncated.data(), &hdr, sizeof(hdr));
    const uint32_t one_value = 5; // should have two values
    std::memcpy(truncated.data() + sizeof(hdr), &one_value, sizeof(one_value));

    CompactIds ids;
    size_t consumed = 123;
    const Ret ret = ids.read(truncated.data(), truncated.size(), &consumed);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::read: truncated payload", ret.message());
    EXPECT_EQ(0u, consumed);
}

TEST(compact_ids, read_buffer_round_trip_reports_bytes_consumed) {
    const std::string path = tmp_dir() + "/sketch2_utest_compact_ids_buf_" + std::to_string(getpid()) + ".bin";
    std::remove(path.c_str());

    CompactIds ids;
    ASSERT_EQ(0, ids.init(std::vector<uint64_t>{20000, 20005, 20011, 29999}).code());

    FILE* f = fopen(path.c_str(), "wb");
    ASSERT_NE(nullptr, f);
    ASSERT_EQ(0, ids.write(f, "write failed").code());
    ASSERT_EQ(0, fclose(f));

    const std::vector<uint8_t> serialized = read_file_bytes(path);
    ASSERT_FALSE(serialized.empty());
    std::remove(path.c_str());

    CompactIds restored;
    size_t consumed = 0;
    ASSERT_EQ(0, restored.read(serialized.data(), serialized.size(), &consumed).code());
    EXPECT_EQ(serialized.size(), consumed);
    EXPECT_EQ(ids.base(), restored.base());
    EXPECT_EQ(ids.count(), restored.count());
    for (size_t i = 0; i < ids.count(); ++i) {
        EXPECT_EQ(ids.id(i), restored.id(i));
    }
}

TEST(compact_ids, read_buffer_round_trip_bitset_reports_bytes_consumed) {
    std::vector<uint64_t> values;
    values.reserve(9000);
    for (uint64_t id = 20000; id < 29000; ++id) {
        values.push_back(id);
    }

    CompactIds ids;
    ASSERT_EQ(0, ids.init(values).code());
    ASSERT_EQ(CompactIdsEncoding::Bitset, ids.preferred_encoding());

    const std::string path = tmp_dir() + "/sketch2_utest_compact_ids_buf_bitset_" + std::to_string(getpid()) + ".bin";
    std::remove(path.c_str());
    FILE* f = fopen(path.c_str(), "wb");
    ASSERT_NE(nullptr, f);
    ASSERT_EQ(0, ids.write(f, "write failed").code());
    ASSERT_EQ(0, fclose(f));

    const std::vector<uint8_t> serialized = read_file_bytes(path);
    ASSERT_FALSE(serialized.empty());
    std::remove(path.c_str());

    CompactIds restored;
    size_t consumed = 0;
    ASSERT_EQ(0, restored.read(serialized.data(), serialized.size(), &consumed).code());
    EXPECT_EQ(serialized.size(), consumed);
    EXPECT_EQ(ids.count(), restored.count());
    for (size_t i = 0; i < ids.count(); ++i) {
        EXPECT_EQ(ids.id(i), restored.id(i));
    }
}

TEST(compact_ids, read_buffer_rejects_truncated_payload) {
    CompactIds ids;
    std::vector<uint8_t> truncated(sizeof(CompactIdsHeaderForTest), 0);
    CompactIdsHeaderForTest hdr{};
    hdr.encoding = static_cast<uint8_t>(CompactIdsEncoding::Offsets32);
    hdr.count = 2;
    hdr.payload_size = 8;
    hdr.base = 100;
    std::memcpy(truncated.data(), &hdr, sizeof(hdr));

    size_t consumed = 123;
    const Ret ret = ids.read(truncated.data(), truncated.size(), &consumed);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::read: truncated payload", ret.message());
    EXPECT_EQ(0u, consumed);
}

} // namespace sketch2
