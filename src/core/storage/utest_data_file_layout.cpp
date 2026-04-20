// Unit tests for binary data-file layout helpers.

#include <gtest/gtest.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
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
    const auto hdr = make_data_header(10, 20, 10, 7, 2, DataType::f32, 64);
    const auto record_layout = compute_data_record_layout(DataType::f32, 64, false);
    EXPECT_EQ(kMagic, hdr.base.magic);
    EXPECT_EQ(static_cast<uint16_t>(FileType::Data), hdr.base.kind);
    EXPECT_EQ(kVersion, hdr.base.version);
    EXPECT_EQ(10u, hdr.min_id);
    EXPECT_EQ(20u, hdr.max_id);
    EXPECT_EQ(10u, hdr.min_range_id);
    EXPECT_EQ(7u, hdr.count);
    EXPECT_EQ(2u, hdr.deleted_count);
    EXPECT_EQ(data_type_to_int(DataType::f32), hdr.type);
    EXPECT_EQ(64u, hdr.dim);
    EXPECT_EQ(0u, hdr.data_offset % kDataRegionAlignment);
    EXPECT_GE(hdr.data_offset, sizeof(DataFileHeader));
    EXPECT_EQ(record_layout.stride, hdr.vector_stride);
    EXPECT_EQ(0u, hdr.vector_stride % kDataAlignment);
    EXPECT_FALSE(data_file_has_norms(hdr));
}

TEST_F(DataFileLayoutTest, MakeDataHeaderSetsCosineNormFlagWhenRequested) {
    const auto hdr = make_data_header(
        10, 20, 10, 7, 2, DataType::f32, 64, data_file_norm_flags_for_dist(DistFunc::COS));
    EXPECT_TRUE(data_file_has_norms(hdr));
    EXPECT_TRUE(data_file_has_cosine_inv_norms(hdr));
    EXPECT_EQ(kDataFileHasCosineInvNorms, hdr.flags);
}

TEST_F(DataFileLayoutTest, MakeDataHeaderSetsSquaredNormFlagWhenRequested) {
    const auto hdr = make_data_header(
        10, 20, 10, 7, 2, DataType::f32, 64, data_file_norm_flags_for_dist(DistFunc::L2));
    EXPECT_TRUE(data_file_has_norms(hdr));
    EXPECT_TRUE(data_file_has_squared_norms(hdr));
    EXPECT_EQ(kDataFileHasSquaredNorms, hdr.flags);
}

TEST_F(DataFileLayoutTest, ComputeDataRecordLayoutF32Dim8MatchesIntendedStrideMath) {
    const auto dot_layout = compute_data_record_layout(DataType::f32, 8, false);
    EXPECT_EQ(8u * sizeof(float), dot_layout.vector_size);
    EXPECT_EQ(8u * sizeof(float), dot_layout.norm_offset);
    EXPECT_EQ(32u, dot_layout.stride);

    const auto cos_layout = compute_data_record_layout(DataType::f32, 8, true);
    EXPECT_EQ(8u * sizeof(float), cos_layout.vector_size);
    EXPECT_EQ(8u * sizeof(float), cos_layout.norm_offset);
    EXPECT_EQ(64u, cos_layout.stride);

    const auto l2_layout = compute_data_record_layout(DataType::f32, 8, true);
    EXPECT_EQ(cos_layout.vector_size, l2_layout.vector_size);
    EXPECT_EQ(cos_layout.norm_offset, l2_layout.norm_offset);
    EXPECT_EQ(cos_layout.stride, l2_layout.stride);
}

TEST_F(DataFileLayoutTest, ComputeDataRecordLayoutF32Dim16MatchesIntendedStrideMath) {
    const auto dot_layout = compute_data_record_layout(DataType::f32, 16, false);
    EXPECT_EQ(16u * sizeof(float), dot_layout.vector_size);
    EXPECT_EQ(16u * sizeof(float), dot_layout.norm_offset);
    EXPECT_EQ(64u, dot_layout.stride);

    const auto cos_layout = compute_data_record_layout(DataType::f32, 16, true);
    EXPECT_EQ(16u * sizeof(float), cos_layout.vector_size);
    EXPECT_EQ(16u * sizeof(float), cos_layout.norm_offset);
    EXPECT_EQ(96u, cos_layout.stride);

    const auto l2_layout = compute_data_record_layout(DataType::f32, 16, true);
    EXPECT_EQ(cos_layout.vector_size, l2_layout.vector_size);
    EXPECT_EQ(cos_layout.norm_offset, l2_layout.norm_offset);
    EXPECT_EQ(cos_layout.stride, l2_layout.stride);
}

TEST_F(DataFileLayoutTest, ComputeDataRecordLayoutNonAlignedVectorShapeKeepsNormInline) {
    const auto dot_layout = compute_data_record_layout(DataType::f32, 5, false);
    EXPECT_EQ(5u * sizeof(float), dot_layout.vector_size);
    EXPECT_EQ(5u * sizeof(float), dot_layout.norm_offset);
    EXPECT_EQ(32u, dot_layout.stride);

    const auto cos_layout = compute_data_record_layout(DataType::f32, 5, true);
    EXPECT_EQ(5u * sizeof(float), cos_layout.vector_size);
    EXPECT_EQ(5u * sizeof(float), cos_layout.norm_offset);
    EXPECT_EQ(32u, cos_layout.stride);

    const auto l2_layout = compute_data_record_layout(DataType::f32, 5, true);
    EXPECT_EQ(cos_layout.vector_size, l2_layout.vector_size);
    EXPECT_EQ(cos_layout.norm_offset, l2_layout.norm_offset);
    EXPECT_EQ(cos_layout.stride, l2_layout.stride);
}

TEST_F(DataFileLayoutTest, ComputeDataRecordLayoutAlignsOddSizedI16NormSlot) {
    const auto dot_layout = compute_data_record_layout(DataType::i16, 5, false);
    EXPECT_EQ(10u, dot_layout.vector_size);
    EXPECT_EQ(12u, dot_layout.norm_offset);
    EXPECT_EQ(32u, dot_layout.stride);

    const auto cos_layout = compute_data_record_layout(DataType::i16, 5, true);
    EXPECT_EQ(10u, cos_layout.vector_size);
    EXPECT_EQ(12u, cos_layout.norm_offset);
    EXPECT_EQ(32u, cos_layout.stride);

    const auto l2_layout = compute_data_record_layout(DataType::i16, 5, true);
    EXPECT_EQ(cos_layout.vector_size, l2_layout.vector_size);
    EXPECT_EQ(cos_layout.norm_offset, l2_layout.norm_offset);
    EXPECT_EQ(cos_layout.stride, l2_layout.stride);
}

TEST_F(DataFileLayoutTest, ComputeMetadataLayoutAlignsIdsTrailerOffsetToRegionBoundary) {
    const auto hdr = make_data_header(0, 0, 0, 0, 0, DataType::f32, 5);
    const auto layout = compute_data_metadata_layout(hdr, 1);
    EXPECT_EQ(static_cast<size_t>(hdr.vector_stride), layout.vectors_bytes);
    EXPECT_EQ(0u, layout.ids_trailer_offset % kDataRegionAlignment);
    EXPECT_EQ(layout.ids_trailer_offset - (static_cast<size_t>(hdr.data_offset) + layout.vectors_bytes),
        layout.vectors_padding);
    EXPECT_EQ(0u, layout.ids_trailer_padding);
}

TEST_F(DataFileLayoutTest, ComputeMetadataLayoutKeepsInlineNormsAndPlacesIdsAfterVectors) {
    const auto hdr = make_data_header(
        0, 0, 0, 0, 0, DataType::f32, 5, data_file_norm_flags_for_dist(DistFunc::COS));
    const auto layout = compute_data_metadata_layout(hdr, 3);
    EXPECT_EQ(0u, layout.ids_trailer_offset % kDataRegionAlignment);
    EXPECT_EQ(layout.ids_trailer_offset,
        align_up<size_t>(static_cast<size_t>(hdr.data_offset) + layout.vectors_bytes, kDataRegionAlignment));
}

TEST_F(DataFileLayoutTest, SetDataHeaderLayoutComputesVectorAndIdsSections) {
    auto hdr = make_data_header(
        10, 20, 10, 3, 1, DataType::f32, 8, data_file_norm_flags_for_dist(DistFunc::COS));
    ASSERT_EQ(0, set_data_header_layout(&hdr, 48, 16).code());
    EXPECT_EQ(hdr.data_offset + static_cast<uint64_t>(hdr.count) * hdr.vector_stride, hdr.data_offset + hdr.vectors_bytes);
    EXPECT_EQ(compute_data_region_offset(static_cast<size_t>(hdr.data_offset) + hdr.vectors_bytes), hdr.ids_offset);
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
    const auto hdr = make_data_header(1, 3, 1, 2, 0, DataType::i16, 4);
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
    auto hdr = make_data_header(1, 3, 1, 2, 0, DataType::f32, 4);
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

TEST_F(DataFileLayoutTest, WriteDataRecordPadsToStrideWithoutNorms) {
    const auto layout = compute_data_record_layout(DataType::f32, 5, false);
    const std::vector<float> values = {1.f, 2.f, 3.f, 4.f, 5.f};

    FILE* f = fopen(path_.c_str(), "wb");
    ASSERT_NE(nullptr, f);
    ASSERT_EQ(0, write_data_record(
        f,
        reinterpret_cast<const uint8_t*>(values.data()),
        layout,
        nullptr,
        "ctx").code());
    fclose(f);

    std::ifstream in(path_, std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    ASSERT_EQ(layout.stride, bytes.size());
    for (size_t i = values.size() * sizeof(float); i < bytes.size(); ++i) {
        EXPECT_EQ(0u, bytes[i]);
    }
}

TEST_F(DataFileLayoutTest, WriteDataRecordStoresInlineNormAndPadding) {
    const auto layout = compute_data_record_layout(DataType::i16, 5, true);
    const std::vector<int16_t> values = {1, 2, 3, 4, 5};
    const float norm = 42.5f;

    FILE* f = fopen(path_.c_str(), "wb");
    ASSERT_NE(nullptr, f);
    ASSERT_EQ(0, write_data_record(
        f,
        reinterpret_cast<const uint8_t*>(values.data()),
        layout,
        &norm,
        "ctx").code());
    fclose(f);

    std::ifstream in(path_, std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    ASSERT_EQ(layout.stride, bytes.size());

    float persisted_norm = 0.0f;
    std::memcpy(&persisted_norm, bytes.data() + layout.norm_offset, sizeof(persisted_norm));
    EXPECT_FLOAT_EQ(norm, persisted_norm);

    for (size_t i = layout.vector_size; i < layout.norm_offset; ++i) {
        EXPECT_EQ(0u, bytes[i]);
    }
    for (size_t i = layout.norm_offset + sizeof(float); i < bytes.size(); ++i) {
        EXPECT_EQ(0u, bytes[i]);
    }
}
