// Unit tests for chunked sparse allowlists.

#include "chunked_bits.h"

#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace sketch2 {
namespace {

struct AlignedBlob {
    void* data = nullptr;
    size_t size = 0;
    size_t allocation_size = 0;

    AlignedBlob() = default;
    AlignedBlob(const AlignedBlob&) = delete;
    AlignedBlob& operator=(const AlignedBlob&) = delete;
    AlignedBlob(AlignedBlob&& other) noexcept
        : data(std::exchange(other.data, nullptr)),
          size(std::exchange(other.size, 0)),
          allocation_size(std::exchange(other.allocation_size, 0)) {}
    AlignedBlob& operator=(AlignedBlob&& other) noexcept {
        if (this != &other) {
            std::free(data);
            data = std::exchange(other.data, nullptr);
            size = std::exchange(other.size, 0);
            allocation_size = std::exchange(other.allocation_size, 0);
        }
        return *this;
    }
    ~AlignedBlob() {
        std::free(data);
    }

    uint8_t* bytes() {
        return static_cast<uint8_t*>(data);
    }
    const uint8_t* bytes() const {
        return static_cast<const uint8_t*>(data);
    }
};

size_t align_up(size_t value, size_t alignment) {
    const size_t mask = alignment - 1u;
    return (value + mask) & ~mask;
}

void write_u64_le(uint8_t* out, uint64_t value) {
    for (size_t i = 0; i < sizeof(value); ++i) {
        out[i] = static_cast<uint8_t>(value >> (i * 8u));
    }
}

uint64_t read_u64_le(const uint8_t* data) {
    uint64_t value = 0;
    for (size_t i = 0; i < sizeof(value); ++i) {
        value |= static_cast<uint64_t>(data[i]) << (i * 8u);
    }
    return value;
}

AlignedBlob serialize_ids(const std::vector<uint64_t>& ids) {
    ChunkedBits bits;
    for (uint64_t id : ids) {
        EXPECT_EQ(0, bits.add(id).code());
    }
    EXPECT_EQ(0, bits.finish().code());

    AlignedBlob blob;
    blob.size = bits.serialized_size_bytes();
    blob.allocation_size = align_up(blob.size, kChunkedBitsBlobAlignment);
    blob.data = std::aligned_alloc(kChunkedBitsBlobAlignment, blob.allocation_size);
    EXPECT_NE(nullptr, blob.data);
    EXPECT_EQ(0, bits.serialize(blob.data, blob.size).code());
    return blob;
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

TEST(chunked_bits, serialized_empty_filter_round_trips) {
    AlignedBlob blob = serialize_ids({});

    ASSERT_EQ(kChunkedBitsBlobHeaderBytes, blob.size);
    ChunkedBitsView view;
    ASSERT_EQ(0, view.init_blob(blob.data, blob.size).code());
    EXPECT_FALSE(view.contains(0));
    EXPECT_FALSE(view.contains(100));
}

TEST(chunked_bits, serialized_unsorted_multi_chunk_input_round_trips) {
    AlignedBlob blob = serialize_ids({(1ull << kChunkBits) + 5u, 1u, 5u, 1u});

    ChunkedBitsView view;
    ASSERT_EQ(0, view.init_blob(blob.data, blob.size).code());
    EXPECT_TRUE(view.contains(1));
    EXPECT_TRUE(view.contains(5));
    EXPECT_TRUE(view.contains((1ull << kChunkBits) + 5u));
    EXPECT_FALSE(view.contains(2));
    EXPECT_FALSE(view.contains(1ull << kChunkBits));
}

TEST(chunked_bits, serialized_payload_offsets_are_aligned) {
    AlignedBlob blob = serialize_ids({1u, (1ull << kChunkBits) + 5u});
    ASSERT_GE(blob.size,
        kChunkedBitsBlobHeaderBytes + 2u * kChunkedBitsBlobDirectoryEntryBytes);

    const uint8_t* first_dir = blob.bytes() + kChunkedBitsBlobHeaderBytes;
    const uint8_t* second_dir = first_dir + kChunkedBitsBlobDirectoryEntryBytes;
    EXPECT_EQ(0u, read_u64_le(first_dir + 8) % kChunkedBitsBlobAlignment);
    EXPECT_EQ(0u, read_u64_le(second_dir + 8) % kChunkedBitsBlobAlignment);
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
    uint8_t* first_dir = blob.bytes() + kChunkedBitsBlobHeaderBytes;
    write_u64_le(first_dir + 8, static_cast<uint64_t>(
        align_up(blob.size, kChunkedBitsBlobAlignment)));

    ChunkedBitsView view;
    const Ret ret = view.init_blob(blob.data, blob.size);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("ChunkedBitsView::init_blob: payload exceeds blob size", ret.message());
}

TEST(chunked_bits, view_rejects_unsorted_directory) {
    AlignedBlob blob = serialize_ids({1u, (1ull << kChunkBits) + 5u});
    uint8_t* second_dir = blob.bytes() + kChunkedBitsBlobHeaderBytes +
        kChunkedBitsBlobDirectoryEntryBytes;
    write_u64_le(second_dir, 0);

    ChunkedBitsView view;
    const Ret ret = view.init_blob(blob.data, blob.size);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("ChunkedBitsView::init_blob: chunk directory is not strictly sorted", ret.message());
}

} // namespace sketch2
