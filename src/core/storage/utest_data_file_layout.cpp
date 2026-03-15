// Unit tests for binary data-file layout helpers.

#include <gtest/gtest.h>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "core/storage/data_file_layout.h"
#include "utest_tmp_dir.h"

using namespace sketch2;

class DataFileLayoutTest : public ::testing::Test {
protected:
    std::string path_;

    void SetUp() override {
        path_ = tmp_dir() + "/sketch2_utest_dfl_" + std::to_string(getpid()) + ".bin";
    }

    void TearDown() override {
        std::remove(path_.c_str());
    }
};

TEST_F(DataFileLayoutTest, MakeDataHeaderSetsExpectedFields) {
    const auto hdr = make_data_header(10, 20, 7, 2, DataType::f32, 64);
    EXPECT_EQ(kMagic, hdr.base.magic);
    EXPECT_EQ(static_cast<uint16_t>(FileType::Data), hdr.base.kind);
    EXPECT_EQ(kVersion, hdr.base.version);
    EXPECT_EQ(10u, hdr.min_id);
    EXPECT_EQ(20u, hdr.max_id);
    EXPECT_EQ(7u, hdr.count);
    EXPECT_EQ(2u, hdr.deleted_count);
    EXPECT_EQ(data_type_to_int(DataType::f32), hdr.type);
    EXPECT_EQ(64u, hdr.dim);
    EXPECT_EQ(0u, hdr.data_offset % kDataAlignment);
    EXPECT_GE(hdr.data_offset, sizeof(DataFileHeader));
    EXPECT_EQ(compute_vector_stride(64u * sizeof(float)), hdr.vector_stride);
    EXPECT_EQ(0u, hdr.vector_stride % kDataAlignment);
    EXPECT_FALSE(data_file_has_cosine_inv_norms(hdr));
}

TEST_F(DataFileLayoutTest, MakeDataHeaderSetsCosineFlagWhenRequested) {
    const auto hdr = make_data_header(10, 20, 7, 2, DataType::f32, 64, true);
    EXPECT_TRUE(data_file_has_cosine_inv_norms(hdr));
    EXPECT_EQ(kDataFileHasCosineInvNorms, hdr.flags);
}

TEST_F(DataFileLayoutTest, ComputeIdsLayoutAlignsIdsOffsetTo8Bytes) {
    const auto hdr = make_data_header(0, 0, 0, 0, DataType::f32, 5);
    const auto layout = compute_ids_layout(hdr, 1);
    EXPECT_EQ(static_cast<size_t>(hdr.vector_stride), layout.vectors_bytes);
    EXPECT_EQ(0u, layout.ids_offset % kIdsAlignment);
    EXPECT_EQ(layout.ids_offset - (static_cast<size_t>(hdr.data_offset) + layout.vectors_bytes), layout.ids_padding);
}

TEST_F(DataFileLayoutTest, ComputeIdsLayoutPlacesCosineSectionBeforeIds) {
    const auto hdr = make_data_header(0, 0, 0, 0, DataType::f32, 5, true);
    const auto layout = compute_ids_layout(hdr, 3);
    EXPECT_EQ(static_cast<size_t>(hdr.data_offset) + layout.vectors_bytes, layout.cosine_inv_norms_offset);
    EXPECT_EQ(3u * sizeof(float), layout.cosine_inv_norms_bytes);
    EXPECT_EQ(0u, layout.ids_offset % kIdsAlignment);
    EXPECT_EQ(layout.ids_offset,
        align_up<size_t>(layout.cosine_inv_norms_offset + layout.cosine_inv_norms_bytes, kIdsAlignment));
}

TEST_F(DataFileLayoutTest, WriteZeroPaddingWritesRequestedZeros) {
    FILE* f = fopen(path_.c_str(), "wb");
    ASSERT_NE(nullptr, f);
    ASSERT_EQ(0, write_zero_padding(f, 13, "pad error").code());
    fclose(f);

    std::ifstream in(path_, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    ASSERT_EQ(13u, bytes.size());
    for (char c : bytes) {
        EXPECT_EQ(0, static_cast<unsigned char>(c));
    }
}

TEST_F(DataFileLayoutTest, WriteHeaderAndDataPaddingProducesAlignedDataOffset) {
    const auto hdr = make_data_header(1, 3, 2, 0, DataType::i16, 4);
    FILE* f = fopen(path_.c_str(), "wb");
    ASSERT_NE(nullptr, f);
    ASSERT_EQ(0, write_header_and_data_padding(f, hdr, "ctx").code());
    fclose(f);

    std::ifstream in(path_, std::ios::binary);
    in.seekg(0, std::ios::end);
    const size_t sz = static_cast<size_t>(in.tellg());
    EXPECT_EQ(static_cast<size_t>(hdr.data_offset), sz);
}

TEST_F(DataFileLayoutTest, RewriteHeaderOverwritesExistingHeader) {
    auto hdr = make_data_header(1, 3, 2, 0, DataType::f32, 4);
    FILE* f = fopen(path_.c_str(), "wb+");
    ASSERT_NE(nullptr, f);
    ASSERT_EQ(0, write_header_and_data_padding(f, hdr, "ctx").code());

    hdr.count = 9;
    hdr.min_id = 100;
    ASSERT_EQ(0, rewrite_header(f, hdr, "ctx").code());
    fclose(f);

    DataFileHeader read{};
    FILE* fr = fopen(path_.c_str(), "rb");
    ASSERT_NE(nullptr, fr);
    ASSERT_EQ(1u, fread(&read, sizeof(read), 1, fr));
    fclose(fr);

    EXPECT_EQ(9u, read.count);
    EXPECT_EQ(100u, read.min_id);
}

TEST_F(DataFileLayoutTest, WriteU64ArrayWritesValuesAndHandlesEmpty) {
    FILE* f = fopen(path_.c_str(), "wb");
    ASSERT_NE(nullptr, f);
    ASSERT_EQ(0, write_u64_array(f, {}, "arr error").code());
    const std::vector<uint64_t> values = {10, 20, 30};
    ASSERT_EQ(0, write_u64_array(f, values, "arr error").code());
    fclose(f);

    std::vector<uint64_t> read(values.size(), 0);
    FILE* fr = fopen(path_.c_str(), "rb");
    ASSERT_NE(nullptr, fr);
    ASSERT_EQ(values.size(), fread(read.data(), sizeof(uint64_t), values.size(), fr));
    fclose(fr);
    EXPECT_EQ(values, read);
}

TEST_F(DataFileLayoutTest, WriteVectorRecordPadsToStride) {
    const auto hdr = make_data_header(1, 1, 1, 0, DataType::f32, 5);
    const std::vector<float> values = {1.f, 2.f, 3.f, 4.f, 5.f};

    FILE* f = fopen(path_.c_str(), "wb");
    ASSERT_NE(nullptr, f);
    ASSERT_EQ(0, write_vector_record(f, reinterpret_cast<const uint8_t*>(values.data()),
        values.size() * sizeof(float), hdr.vector_stride, "ctx").code());
    fclose(f);

    std::ifstream in(path_, std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    ASSERT_EQ(static_cast<size_t>(hdr.vector_stride), bytes.size());
    for (size_t i = values.size() * sizeof(float); i < bytes.size(); ++i) {
        EXPECT_EQ(0u, bytes[i]);
    }
}
