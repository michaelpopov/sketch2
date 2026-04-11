// Unit tests for the CompactIds utility.

#include "compact_ids.h"

#include <gtest/gtest.h>

#include <cstdio>
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

void write_test_header(FILE* f, const CompactIdsHeaderForTest& hdr) {
    ASSERT_NE(nullptr, f);
    ASSERT_EQ(1u, fwrite(&hdr, sizeof(hdr), 1, f));
}

} // namespace

TEST(compact_ids, init_empty_container) {
    CompactIds ids;

    ASSERT_EQ(0, ids.init(static_cast<const uint64_t*>(nullptr), 0).code());
    EXPECT_TRUE(ids.empty());
    EXPECT_EQ(0u, ids.count());
    EXPECT_EQ(0u, ids.base());
    EXPECT_EQ(0u, ids.lower_bound_index(123));
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

TEST(compact_ids, init_rejects_null_pointer_with_nonzero_count) {
    CompactIds ids;

    const Ret ret = ids.init(static_cast<const uint64_t*>(nullptr), 3);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::init: ids pointer is null", ret.message());
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
    CompactIds ids;
    ASSERT_EQ(0, ids.init(100, std::vector<uint32_t>{5, 10}).code());

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
    CompactIds ids;
    const std::vector<uint32_t> offsets = {0, 5, 11, 9999};

    ASSERT_EQ(0, ids.init(20000, offsets).code());
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
    CompactIds ids;
    ASSERT_EQ(0, ids.init(100, std::vector<uint32_t>{5, 10}).code());

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

TEST(compact_ids, init_from_base_and_offsets_rejects_null_pointer_with_nonzero_count) {
    CompactIds ids;

    const Ret ret = ids.init(123, static_cast<const uint32_t*>(nullptr), 2);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::init: offsets pointer is null", ret.message());
}

TEST(compact_ids, init_from_base_and_offsets_rejects_duplicate_offsets) {
    CompactIds ids;
    const std::vector<uint32_t> offsets = {0, 5, 5};

    const Ret ret = ids.init(100, offsets);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::init: offsets must be strictly increasing", ret.message());
}

TEST(compact_ids, init_from_base_and_offsets_rejects_unsorted_offsets) {
    CompactIds ids;
    const std::vector<uint32_t> offsets = {0, 7, 6};

    const Ret ret = ids.init(100, offsets);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::init: offsets must be strictly increasing", ret.message());
}

TEST(compact_ids, init_from_base_and_offsets_rejects_uint64_overflow) {
    CompactIds ids;
    const std::vector<uint32_t> offsets = {0, 10};

    const Ret ret = ids.init(std::numeric_limits<uint64_t>::max() - 5u, offsets);

    EXPECT_NE(0, ret.code());
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

TEST(compact_ids, write_offsets_serializes_uint32_offsets) {
    const std::string path = tmp_dir() + "/sketch2_utest_compact_ids_" + std::to_string(getpid()) + ".bin";
    std::remove(path.c_str());

    CompactIds ids;
    ASSERT_EQ(0, ids.init(std::vector<uint64_t>{20000, 20005, 20011, 29999}).code());

    FILE* f = fopen(path.c_str(), "wb");
    ASSERT_NE(nullptr, f);
    ASSERT_EQ(0, ids.write_offsets(f, "write failed").code());
    ASSERT_EQ(0, fclose(f));

    std::vector<uint32_t> offsets(ids.count());
    f = fopen(path.c_str(), "rb");
    ASSERT_NE(nullptr, f);
    ASSERT_EQ(offsets.size(), fread(offsets.data(), sizeof(uint32_t), offsets.size(), f));
    ASSERT_EQ(0, fclose(f));

    EXPECT_EQ((std::vector<uint32_t>{0u, 5u, 11u, 9999u}), offsets);
    std::remove(path.c_str());
}

TEST(compact_ids, write_offsets_rejects_null_file_handle) {
    CompactIds ids;
    ASSERT_EQ(0, ids.init(std::vector<uint64_t>{1, 2, 3}).code());

    const Ret ret = ids.write_offsets(nullptr, "write failed");

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("write failed: file handle is null", ret.message());
}

TEST(compact_ids, write_and_read_round_trip_offsets_encoding) {
    const std::string path = tmp_dir() + "/sketch2_utest_compact_ids_offsets_" + std::to_string(getpid()) + ".bin";
    std::remove(path.c_str());

    CompactIds ids;
    ASSERT_EQ(0, ids.init(std::vector<uint64_t>{10, 1000, 2000}).code());
    ASSERT_EQ(CompactIdsEncoding::Offsets32, ids.preferred_encoding());

    FILE* f = fopen(path.c_str(), "wb");
    ASSERT_NE(nullptr, f);
    ASSERT_EQ(0, ids.write(f, "write failed").code());
    ASSERT_EQ(0, fclose(f));

    CompactIds restored;
    f = fopen(path.c_str(), "rb");
    ASSERT_NE(nullptr, f);
    ASSERT_EQ(0, restored.read(f).code());
    ASSERT_EQ(0, fclose(f));

    EXPECT_EQ(ids.base(), restored.base());
    EXPECT_EQ(ids.count(), restored.count());
    for (size_t i = 0; i < ids.count(); ++i) {
        EXPECT_EQ(ids.id(i), restored.id(i));
        EXPECT_EQ(ids.offset(i), restored.offset(i));
    }
    std::remove(path.c_str());
}

TEST(compact_ids, write_and_read_round_trip_bitset_encoding) {
    const std::string path = tmp_dir() + "/sketch2_utest_compact_ids_bitset_" + std::to_string(getpid()) + ".bin";
    std::remove(path.c_str());

    std::vector<uint64_t> values;
    values.reserve(9000);
    for (uint64_t id = 20000; id < 29000; ++id) {
        values.push_back(id);
    }

    CompactIds ids;
    ASSERT_EQ(0, ids.init(values).code());
    ASSERT_EQ(CompactIdsEncoding::Bitset, ids.preferred_encoding());

    FILE* f = fopen(path.c_str(), "wb");
    ASSERT_NE(nullptr, f);
    ASSERT_EQ(0, ids.write(f, "write failed").code());
    ASSERT_EQ(0, fclose(f));

    CompactIds restored;
    f = fopen(path.c_str(), "rb");
    ASSERT_NE(nullptr, f);
    ASSERT_EQ(0, restored.read(f).code());
    ASSERT_EQ(0, fclose(f));

    EXPECT_EQ(ids.base(), restored.base());
    EXPECT_EQ(ids.count(), restored.count());
    for (size_t i = 0; i < ids.count(); ++i) {
        EXPECT_EQ(ids.id(i), restored.id(i));
    }
    std::remove(path.c_str());
}

TEST(compact_ids, write_and_read_round_trip_empty_container) {
    const std::string path = tmp_dir() + "/sketch2_utest_compact_ids_empty_" + std::to_string(getpid()) + ".bin";
    std::remove(path.c_str());

    CompactIds ids;
    ASSERT_EQ(0, ids.init(std::vector<uint64_t>{}).code());
    ASSERT_TRUE(ids.empty());

    FILE* f = fopen(path.c_str(), "wb");
    ASSERT_NE(nullptr, f);
    ASSERT_EQ(0, ids.write(f, "write failed").code());
    ASSERT_EQ(0, fclose(f));

    CompactIds restored;
    f = fopen(path.c_str(), "rb");
    ASSERT_NE(nullptr, f);
    ASSERT_EQ(0, restored.read(f).code());
    ASSERT_EQ(0, fclose(f));

    EXPECT_TRUE(restored.empty());
    EXPECT_EQ(0u, restored.count());
    EXPECT_EQ(0u, restored.base());
    std::remove(path.c_str());
}

TEST(compact_ids, write_rejects_null_file_handle) {
    CompactIds ids;
    ASSERT_EQ(0, ids.init(std::vector<uint64_t>{1, 2, 3}).code());

    const Ret ret = ids.write(nullptr, "write failed");

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("write failed: file handle is null", ret.message());
}

TEST(compact_ids, read_rejects_null_file_handle) {
    CompactIds ids;

    const Ret ret = ids.read(nullptr);

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::read: file handle is null", ret.message());
}

TEST(compact_ids, read_rejects_truncated_header) {
    const std::string path = tmp_dir() + "/sketch2_utest_compact_ids_bad_hdr_" + std::to_string(getpid()) + ".bin";
    std::remove(path.c_str());

    FILE* f = fopen(path.c_str(), "wb");
    ASSERT_NE(nullptr, f);
    const uint8_t short_header[8] = {0};
    ASSERT_EQ(sizeof(short_header), fwrite(short_header, 1, sizeof(short_header), f));
    ASSERT_EQ(0, fclose(f));

    CompactIds ids;
    f = fopen(path.c_str(), "rb");
    ASSERT_NE(nullptr, f);
    const Ret ret = ids.read(f);
    ASSERT_EQ(0, fclose(f));

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::read: failed to read header", ret.message());
    std::remove(path.c_str());
}

TEST(compact_ids, read_rejects_unknown_encoding) {
    const std::string path = tmp_dir() + "/sketch2_utest_compact_ids_bad_enc_" + std::to_string(getpid()) + ".bin";
    std::remove(path.c_str());

    FILE* f = fopen(path.c_str(), "wb");
    ASSERT_NE(nullptr, f);
    CompactIdsHeaderForTest hdr{};
    hdr.encoding = 99;
    write_test_header(f, hdr);
    ASSERT_EQ(0, fclose(f));

    CompactIds ids;
    f = fopen(path.c_str(), "rb");
    ASSERT_NE(nullptr, f);
    const Ret ret = ids.read(f);
    ASSERT_EQ(0, fclose(f));

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::read: unknown encoding", ret.message());
    std::remove(path.c_str());
}

TEST(compact_ids, read_rejects_malformed_empty_header) {
    const std::string path = tmp_dir() + "/sketch2_utest_compact_ids_bad_empty_" + std::to_string(getpid()) + ".bin";
    std::remove(path.c_str());

    FILE* f = fopen(path.c_str(), "wb");
    ASSERT_NE(nullptr, f);
    CompactIdsHeaderForTest hdr{};
    hdr.encoding = static_cast<uint8_t>(CompactIdsEncoding::Offsets32);
    hdr.count = 0;
    hdr.base = 123; // illegal for empty payload path
    write_test_header(f, hdr);
    ASSERT_EQ(0, fclose(f));

    CompactIds ids;
    f = fopen(path.c_str(), "rb");
    ASSERT_NE(nullptr, f);
    const Ret ret = ids.read(f);
    ASSERT_EQ(0, fclose(f));

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::read: malformed empty payload header", ret.message());
    std::remove(path.c_str());
}

TEST(compact_ids, read_rejects_offsets_payload_size_mismatch) {
    const std::string path = tmp_dir() + "/sketch2_utest_compact_ids_bad_offsets_size_" + std::to_string(getpid()) + ".bin";
    std::remove(path.c_str());

    FILE* f = fopen(path.c_str(), "wb");
    ASSERT_NE(nullptr, f);
    CompactIdsHeaderForTest hdr{};
    hdr.encoding = static_cast<uint8_t>(CompactIdsEncoding::Offsets32);
    hdr.count = 2;
    hdr.payload_size = 4; // should be 8
    hdr.base = 100;
    write_test_header(f, hdr);
    ASSERT_EQ(0, fclose(f));

    CompactIds ids;
    f = fopen(path.c_str(), "rb");
    ASSERT_NE(nullptr, f);
    const Ret ret = ids.read(f);
    ASSERT_EQ(0, fclose(f));

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::read: malformed offsets payload size", ret.message());
    std::remove(path.c_str());
}

TEST(compact_ids, read_rejects_bitset_payload_size_mismatch) {
    const std::string path = tmp_dir() + "/sketch2_utest_compact_ids_bad_bitset_size_" + std::to_string(getpid()) + ".bin";
    std::remove(path.c_str());

    FILE* f = fopen(path.c_str(), "wb");
    ASSERT_NE(nullptr, f);
    CompactIdsHeaderForTest hdr{};
    hdr.encoding = static_cast<uint8_t>(CompactIdsEncoding::Bitset);
    hdr.count = 3;
    hdr.max_offset = 31;    // expected bitset payload is 4 bytes
    hdr.payload_size = 3;   // mismatch
    hdr.base = 200;
    write_test_header(f, hdr);
    ASSERT_EQ(0, fclose(f));

    CompactIds ids;
    f = fopen(path.c_str(), "rb");
    ASSERT_NE(nullptr, f);
    const Ret ret = ids.read(f);
    ASSERT_EQ(0, fclose(f));

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::read: malformed bitset payload size", ret.message());
    std::remove(path.c_str());
}

TEST(compact_ids, read_rejects_bitset_payload_with_nonzero_tail_bits) {
    const std::string path = tmp_dir() + "/sketch2_utest_compact_ids_bad_bitset_tail_" + std::to_string(getpid()) + ".bin";
    std::remove(path.c_str());

    FILE* f = fopen(path.c_str(), "wb");
    ASSERT_NE(nullptr, f);
    CompactIdsHeaderForTest hdr{};
    hdr.encoding = static_cast<uint8_t>(CompactIdsEncoding::Bitset);
    hdr.count = 1;
    hdr.max_offset = 5;   // valid bits are [0..5], bits 6-7 must be zero
    hdr.payload_size = 1; // one byte payload
    hdr.base = 100;
    write_test_header(f, hdr);
    const uint8_t malformed_payload = 0x41; // bit 0 set (valid), bit 6 set (invalid tail bit)
    ASSERT_EQ(1u, fwrite(&malformed_payload, 1, 1, f));
    ASSERT_EQ(0, fclose(f));

    CompactIds ids;
    f = fopen(path.c_str(), "rb");
    ASSERT_NE(nullptr, f);
    const Ret ret = ids.read(f);
    ASSERT_EQ(0, fclose(f));

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::read: malformed bitset payload tail bits", ret.message());
    std::remove(path.c_str());
}

TEST(compact_ids, read_rejects_truncated_offsets_payload) {
    const std::string path = tmp_dir() + "/sketch2_utest_compact_ids_trunc_offsets_" + std::to_string(getpid()) + ".bin";
    std::remove(path.c_str());

    FILE* f = fopen(path.c_str(), "wb");
    ASSERT_NE(nullptr, f);
    CompactIdsHeaderForTest hdr{};
    hdr.encoding = static_cast<uint8_t>(CompactIdsEncoding::Offsets32);
    hdr.count = 2;
    hdr.payload_size = 8;
    hdr.base = 100;
    write_test_header(f, hdr);
    const uint32_t one_value = 5; // should have two values
    ASSERT_EQ(1u, fwrite(&one_value, sizeof(one_value), 1, f));
    ASSERT_EQ(0, fclose(f));

    CompactIds ids;
    f = fopen(path.c_str(), "rb");
    ASSERT_NE(nullptr, f);
    const Ret ret = ids.read(f);
    ASSERT_EQ(0, fclose(f));

    EXPECT_NE(0, ret.code());
    EXPECT_EQ("CompactIds::read: failed to read offsets payload", ret.message());
    std::remove(path.c_str());
}

} // namespace sketch2
