// Unit tests for chunked sparse bitset filters.

#include "chunked_bits.h"
#include "utest_chunked_bits_helpers.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

namespace sketch2 {
class ChunkedBitsTestPeer {
public:
    static void mark_finish_failed(ChunkedBits* bits, const Ret& ret) {
        bits->finished_ = true;
        bits->cached_serialized_size_ = 0;
        bits->finish_ret_ = ret;
    }
};

namespace {

using test::AlignedBlob;
using test::align_up;

struct ChunkedBitsBlobDirectoryEntry {
    uint64_t chunk_id = 0;
    uint64_t payload_offset = 0;
    uint64_t payload_size = 0;
};

static_assert(sizeof(ChunkedBitsBlobDirectoryEntry) ==
    kChunkedBitsBlobDirectoryEntryBytes);

test::AlignedBlob serialize_ids(const std::vector<uint64_t>& ids) {
    ChunkedBits bits;
    for (uint64_t id : ids) {
        EXPECT_EQ(0, bits.add(id).code());
    }
    EXPECT_EQ(0, bits.finish().code());

    test::AlignedBlob blob = test::make_aligned_blob(bits.serialized_size_bytes());
    EXPECT_EQ(0, bits.serialize(blob.data, blob.size).code());
    return blob;
}

test::AlignedBlob serialize_finished(ChunkedBits&& bits) {
    EXPECT_EQ(0, bits.finish().code());
    test::AlignedBlob blob = test::make_aligned_blob(bits.serialized_size_bytes());
    EXPECT_EQ(0, bits.serialize(blob.data, blob.size).code());
    return blob;
}

void expect_view_contains_in_order(
        const ChunkedBitsView& view, const std::vector<uint64_t>& ids) {
    auto it = view.begin();
    for (uint64_t id : ids) {
        ASSERT_FALSE(it.eof()) << id;
        EXPECT_EQ(id, it.id()) << id;
        it.next();
    }
    EXPECT_TRUE(it.eof());
}

} // namespace

TEST(chunked_bits, rejects_too_many_chunks) {
    ChunkedBits bits;

    for (size_t i = 0; i < kChunkedBitsMaxChunks; ++i) {
        ASSERT_EQ(0, bits.add(static_cast<uint64_t>(i) << kChunkBits).code());
    }

    const Ret ret = bits.add(static_cast<uint64_t>(kChunkedBitsMaxChunks) << kChunkBits);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("ChunkedBits::add: too many chunks", ret.message());
}

TEST(chunked_bits, serialize_before_finish_reports_root_cause) {
    ChunkedBits bits;
    alignas(kChunkedBitsBlobAlignment) uint8_t buffer[kChunkedBitsBlobHeaderBytes] = {};

    const Ret ret = bits.serialize(buffer, sizeof(buffer));
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("ChunkedBits::serialize: finish must be called before serialization",
        ret.message());
}

TEST(chunked_bits, finish_failure_is_stable_and_blocks_serialize) {
    ChunkedBits bits;
    const Ret overflow("ChunkedBits::serialized_size_bytes: payload size overflow");
    ChunkedBitsTestPeer::mark_finish_failed(&bits, overflow);

    EXPECT_EQ(overflow.message(), bits.finish().message());

    alignas(kChunkedBitsBlobAlignment) uint8_t buffer[kChunkedBitsBlobHeaderBytes] = {};
    const Ret serialize_ret = bits.serialize(buffer, 0);
    EXPECT_EQ(overflow.message(), serialize_ret.message());
}

TEST(chunked_bits, serialized_empty_filter_round_trips) {
    AlignedBlob blob = serialize_ids({});

    ASSERT_EQ(kChunkedBitsBlobHeaderBytes, blob.size);
    ChunkedBitsView view;
    ASSERT_EQ(0, view.init_blob(blob.data, blob.size).code());
    auto it = view.begin();
    EXPECT_TRUE(it.eof());
    EXPECT_FALSE(it.seek_at_least(0));
    EXPECT_FALSE(it.consume_if_equal(100));
}

TEST(chunked_bits, serialized_unsorted_multi_chunk_input_round_trips) {
    AlignedBlob blob = serialize_ids({(1ull << kChunkBits) + 5u, 1u, 5u, 1u});

    ChunkedBitsView view;
    ASSERT_EQ(0, view.init_blob(blob.data, blob.size).code());

    auto it = view.begin();
    ASSERT_FALSE(it.eof());
    EXPECT_EQ(1u, it.id());
    EXPECT_TRUE(it.consume_if_equal(1));
    EXPECT_FALSE(it.consume_if_equal(2));
    EXPECT_TRUE(it.consume_if_equal(5));
    EXPECT_FALSE(it.consume_if_equal(1ull << kChunkBits));
    EXPECT_TRUE(it.consume_if_equal((1ull << kChunkBits) + 5u));
    EXPECT_TRUE(it.eof());

    auto seek_it = view.begin();
    EXPECT_TRUE(seek_it.seek_at_least(2));
    EXPECT_EQ(5u, seek_it.id());
    EXPECT_TRUE(seek_it.seek_at_least(1ull << kChunkBits));
    EXPECT_EQ((1ull << kChunkBits) + 5u, seek_it.id());
    EXPECT_FALSE(seek_it.seek_at_least((1ull << kChunkBits) + 6u));
}

TEST(chunked_bits, serialized_payload_offsets_are_aligned) {
    AlignedBlob blob = serialize_ids({1u, (1ull << kChunkBits) + 5u});
    ASSERT_GE(blob.size,
        kChunkedBitsBlobHeaderBytes + 2u * kChunkedBitsBlobDirectoryEntryBytes);

    ChunkedBitsBlobDirectoryEntry first_dir;
    ChunkedBitsBlobDirectoryEntry second_dir;
    std::memcpy(&first_dir, blob.bytes() + kChunkedBitsBlobHeaderBytes, sizeof(first_dir));
    std::memcpy(&second_dir, blob.bytes() + kChunkedBitsBlobHeaderBytes +
        kChunkedBitsBlobDirectoryEntryBytes, sizeof(second_dir));
    EXPECT_EQ(0u, first_dir.payload_offset % kChunkedBitsBlobAlignment);
    EXPECT_EQ(0u, second_dir.payload_offset % kChunkedBitsBlobAlignment);
}

TEST(chunked_bits, view_rejects_misaligned_blob_pointer) {
    AlignedBlob blob = serialize_ids({1u});
    std::vector<uint8_t> misaligned(blob.size + 1u);
    std::memcpy(misaligned.data() + 1u, blob.data, blob.size);

    ChunkedBitsView view;
    const Ret ret = view.init_blob(misaligned.data() + 1u, blob.size);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("ChunkedBitsView::init_blob: blob buffer must be 32-byte aligned", ret.message());
}

TEST(chunked_bits, view_rejects_malformed_header) {
    AlignedBlob blob = serialize_ids({1u});
    blob.bytes()[0] = 0;

    ChunkedBitsView view;
    const Ret ret = view.init_blob(blob.data, blob.size);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("ChunkedBitsView::init_blob: invalid blob magic", ret.message());
}

TEST(chunked_bits, view_rejects_out_of_bounds_payload) {
    AlignedBlob blob = serialize_ids({1u});
    ChunkedBitsBlobDirectoryEntry first_dir;
    std::memcpy(&first_dir, blob.bytes() + kChunkedBitsBlobHeaderBytes, sizeof(first_dir));
    first_dir.payload_offset = static_cast<uint64_t>(
        align_up(blob.size, kChunkedBitsBlobAlignment));
    std::memcpy(blob.bytes() + kChunkedBitsBlobHeaderBytes, &first_dir, sizeof(first_dir));

    ChunkedBitsView view;
    const Ret ret = view.init_blob(blob.data, blob.size);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("ChunkedBitsView::init_blob: payload exceeds blob size", ret.message());
}

TEST(chunked_bits, batch_add_round_trips_single_chunk) {
    ChunkedBits bits;
    const std::vector<uint64_t> ids = {1u, 2u, 3u, 5u, 8u, 13u, 21u};
    ASSERT_EQ(0, bits.add(ids.data(), ids.size()).code());
    AlignedBlob blob = serialize_finished(std::move(bits));

    ChunkedBitsView view;
    ASSERT_EQ(0, view.init_blob(blob.data, blob.size).code());
    expect_view_contains_in_order(view, ids);
}

TEST(chunked_bits, batch_add_round_trips_multi_chunk_with_dense_runs) {
    ChunkedBits bits;
    std::vector<uint64_t> ids;
    // Dense run inside chunk 0 to exercise add_range_closed coalescing.
    for (uint64_t id = 100; id < 200; ++id) {
        ids.push_back(id);
    }
    // Sparse ids in chunks 1 and 2.
    ids.push_back((1ull << kChunkBits) + 7u);
    ids.push_back((1ull << kChunkBits) + 9u);
    ids.push_back((2ull << kChunkBits));
    ids.push_back((2ull << kChunkBits) + 42u);

    ASSERT_EQ(0, bits.add(ids.data(), ids.size()).code());
    AlignedBlob blob = serialize_finished(std::move(bits));

    ChunkedBitsView view;
    ASSERT_EQ(0, view.init_blob(blob.data, blob.size).code());
    expect_view_contains_in_order(view, ids);
}

TEST(chunked_bits, batch_add_tolerates_duplicates) {
    ChunkedBits bits;
    const std::vector<uint64_t> ids = {1u, 1u, 2u, 2u, 2u, 3u};
    ASSERT_EQ(0, bits.add(ids.data(), ids.size()).code());
    AlignedBlob blob = serialize_finished(std::move(bits));

    ChunkedBitsView view;
    ASSERT_EQ(0, view.init_blob(blob.data, blob.size).code());
    expect_view_contains_in_order(view, {1u, 2u, 3u});
}

TEST(chunked_bits, batch_add_rejects_unsorted_input) {
    ChunkedBits bits;
    const uint64_t ids[] = {3u, 1u};
    const Ret ret = bits.add(ids, 2);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("ChunkedBits::add: ids must be sorted in non-decreasing order",
        ret.message());
}

TEST(chunked_bits, batch_add_rejects_unsorted_across_chunk_boundary) {
    ChunkedBits bits;
    const uint64_t ids[] = {(1ull << kChunkBits) + 5u, 4u};
    const Ret ret = bits.add(ids, 2);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("ChunkedBits::add: ids must be sorted in non-decreasing order",
        ret.message());
}

TEST(chunked_bits, batch_add_empty_size_is_a_noop) {
    ChunkedBits bits;
    const uint64_t* ids = nullptr;
    EXPECT_EQ(0, bits.add(ids, 0).code());
    AlignedBlob blob = serialize_finished(std::move(bits));

    ChunkedBitsView view;
    ASSERT_EQ(0, view.init_blob(blob.data, blob.size).code());
    EXPECT_TRUE(view.begin().eof());
}

TEST(chunked_bits, batch_add_rejects_null_pointer_with_nonzero_size) {
    ChunkedBits bits;
    const Ret ret = bits.add(nullptr, 1);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("ChunkedBits::add: ids pointer is null", ret.message());
}

TEST(chunked_bits, batch_add_after_finish_is_rejected) {
    ChunkedBits bits;
    ASSERT_EQ(0, bits.finish().code());
    const uint64_t ids[] = {1u};
    const Ret ret = bits.add(ids, 1);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("ChunkedBits::add: cannot add after finish", ret.message());
}

TEST(chunked_bits, batch_add_interleaves_with_single_add) {
    ChunkedBits bits;
    ASSERT_EQ(0, bits.add(2u).code());
    const uint64_t batch1[] = {5u, 7u, (1ull << kChunkBits) + 3u};
    ASSERT_EQ(0, bits.add(batch1, 3).code());
    ASSERT_EQ(0, bits.add((1ull << kChunkBits) + 4u).code());
    const uint64_t batch2[] = {(2ull << kChunkBits), (2ull << kChunkBits) + 1u};
    ASSERT_EQ(0, bits.add(batch2, 2).code());
    AlignedBlob blob = serialize_finished(std::move(bits));

    ChunkedBitsView view;
    ASSERT_EQ(0, view.init_blob(blob.data, blob.size).code());
    expect_view_contains_in_order(view, {
        2u, 5u, 7u,
        (1ull << kChunkBits) + 3u, (1ull << kChunkBits) + 4u,
        (2ull << kChunkBits), (2ull << kChunkBits) + 1u,
    });
}

TEST(chunked_bits, batch_add_respects_chunk_cap) {
    ChunkedBits bits;
    std::vector<uint64_t> ids;
    ids.reserve(kChunkedBitsMaxChunks + 1);
    for (size_t i = 0; i <= kChunkedBitsMaxChunks; ++i) {
        ids.push_back(static_cast<uint64_t>(i) << kChunkBits);
    }
    const Ret ret = bits.add(ids.data(), ids.size());
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("ChunkedBits::add: too many chunks", ret.message());
}

TEST(chunked_bits, view_rejects_unsorted_directory) {
    AlignedBlob blob = serialize_ids({1u, (1ull << kChunkBits) + 5u});
    ChunkedBitsBlobDirectoryEntry second_dir;
    std::memcpy(&second_dir, blob.bytes() + kChunkedBitsBlobHeaderBytes +
        kChunkedBitsBlobDirectoryEntryBytes, sizeof(second_dir));
    second_dir.chunk_id = 0;
    std::memcpy(blob.bytes() + kChunkedBitsBlobHeaderBytes +
        kChunkedBitsBlobDirectoryEntryBytes, &second_dir, sizeof(second_dir));

    ChunkedBitsView view;
    const Ret ret = view.init_blob(blob.data, blob.size);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("ChunkedBitsView::init_blob: chunk directory is not strictly sorted", ret.message());
}

} // namespace sketch2
