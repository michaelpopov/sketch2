// Unit tests for merging base data, deltas, and accumulator output.

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include <vector>
#include "core/compute/norm_utils.h"
#include "core/storage/data_file.h"
#include "core/storage/data_file_layout.h"
#include "core/storage/data_merger.h"
#include "core/storage/data_reader.h"
#include "core/storage/input_reader.h"
#include "core/storage/compact_ids.h"
#include "core/storage/compact_ids_shared.h"
#include "utest_tmp_dir.h"

using namespace sketch2;
namespace fs = std::filesystem;

class DataMergerTest : public ::testing::Test {
protected:
    static constexpr uint16_t kDim = 4;
    std::string base_dir_;
    struct CompactIdsHeaderForTest {
        uint8_t encoding = 0;
        uint8_t reserved0 = 0;
        uint16_t reserved1 = 0;
        uint32_t count = 0;
        uint32_t max_offset = 0;
        uint32_t payload_size = 0;
        uint64_t base = 0;
    };

    static_assert(sizeof(CompactIdsHeaderForTest) == 24, "Unexpected CompactIdsOffsets header size");

    void SetUp() override {
        base_dir_ = tmp_dir() + "/sketch2_utest_dm_" + std::to_string(getpid());
        fs::create_directories(base_dir_);
    }

    void TearDown() override {
        fs::remove_all(base_dir_);
    }

    std::string p(const std::string& name) const {
        return base_dir_ + "/" + name;
    }

    void expect_sorted_unique(const std::vector<uint64_t>& ids) {
        for (size_t i = 1; i < ids.size(); ++i) {
            ASSERT_LT(ids[i - 1], ids[i]);
        }
    }

    void write_f32_file(const std::string& path,
                        FileType kind,
                        const std::vector<std::pair<uint64_t, float>>& active,
                        const std::vector<uint64_t>& deleted,
                        uint16_t dim = kDim,
                        bool has_norms = false,
                        DistFunc stored_norm_dist_func = DistFunc::COS,
                        uint64_t min_range_id = 0) {
        std::vector<uint64_t> active_ids;
        active_ids.reserve(active.size());
        for (const auto& item : active) {
            active_ids.push_back(item.first);
        }
        expect_sorted_unique(active_ids);
        expect_sorted_unique(deleted);

        DataFileHeader hdr = make_data_header(
            active.empty() ? 0 : active.front().first,
            active.empty() ? 0 : active.back().first,
            min_range_id,
            static_cast<uint32_t>(active.size()),
            static_cast<uint32_t>(deleted.size()),
            DataType::f32,
            dim,
            has_norms ? data_file_norm_flags_for_dist(stored_norm_dist_func) : 0u);
        hdr.base.kind = static_cast<uint16_t>(kind);
        CompactIds compact_active_ids;
        ASSERT_EQ(0, compact_active_ids.init(active_ids).code());
        CompactIds compact_deleted_ids;
        std::vector<uint64_t> deleted_ids = deleted;
        ASSERT_EQ(0, compact_deleted_ids.init(deleted_ids).code());
        ASSERT_EQ(0, set_data_header_layout(
            &hdr, compact_active_ids.serialized_size_bytes(), compact_deleted_ids.serialized_size_bytes()).code());

        FILE* f = fopen(path.c_str(), "wb");
        ASSERT_NE(nullptr, f);
        ASSERT_EQ(1u, fwrite(&hdr, sizeof(hdr), 1, f));
        const size_t pad_size = static_cast<size_t>(hdr.data_offset) - sizeof(DataFileHeader);
        if (pad_size > 0) {
            std::vector<uint8_t> pad(pad_size, 0);
            ASSERT_EQ(pad.size(), fwrite(pad.data(), 1, pad.size(), f));
        }

        for (const auto& item : active) {
            std::vector<float> vec(dim, item.second);
            const DataRecordLayout record_layout =
                compute_data_record_layout(DataType::f32, dim, has_norms);
            float norm = 0.0f;
            float* norm_ptr = nullptr;
            if (has_norms) {
                if (stored_norm_dist_func == DistFunc::COS) {
                    norm = inverse_norm(
                        reinterpret_cast<const uint8_t*>(vec.data()), DataType::f32, dim);
                } else if (stored_norm_dist_func == DistFunc::L2) {
                    norm = compute_squared_norm(
                        reinterpret_cast<const uint8_t*>(vec.data()), DataType::f32, dim);
                } else {
                    FAIL() << "write_f32_file: invalid stored norm distance function";
                }
                norm_ptr = &norm;
            }
            ASSERT_EQ(0, write_data_record(
                f,
                reinterpret_cast<const uint8_t*>(vec.data()),
                record_layout,
                norm_ptr,
                "DataMergerTest::write_f32_file").code());
        }
        const DataMetadataLayout metadata_layout = compute_data_metadata_layout(hdr, active.size());
        const size_t ids_pad_size = metadata_layout.vectors_padding;
        if (ids_pad_size > 0) {
            std::vector<uint8_t> pad(ids_pad_size, 0);
            ASSERT_EQ(pad.size(), fwrite(pad.data(), 1, pad.size(), f));
        }
        ASSERT_EQ(0, compact_active_ids.write(f, "DataMergerTest::write_f32_file active ids").code());
        const size_t deleted_ids_pad_size =
            compute_deleted_ids_padding(metadata_layout.ids_trailer_offset, compact_active_ids.serialized_size_bytes());
        if (deleted_ids_pad_size > 0) {
            std::vector<uint8_t> pad(deleted_ids_pad_size, 0);
            ASSERT_EQ(pad.size(), fwrite(pad.data(), 1, pad.size(), f));
        }
        ASSERT_EQ(0, compact_deleted_ids.write(f, "DataMergerTest::write_f32_file deleted ids").code());
        fclose(f);
    }

    void write_i16_file(const std::string& path,
                        FileType kind,
                        const std::vector<std::pair<uint64_t, int16_t>>& active,
                        const std::vector<uint64_t>& deleted,
                        uint16_t dim = kDim,
                        bool has_norms = false,
                        DistFunc stored_norm_dist_func = DistFunc::COS,
                        uint64_t min_range_id = 0) {
        std::vector<uint64_t> active_ids;
        active_ids.reserve(active.size());
        for (const auto& item : active) {
            active_ids.push_back(item.first);
        }
        expect_sorted_unique(active_ids);
        expect_sorted_unique(deleted);

        DataFileHeader hdr = make_data_header(
            active.empty() ? 0 : active.front().first,
            active.empty() ? 0 : active.back().first,
            min_range_id,
            static_cast<uint32_t>(active.size()),
            static_cast<uint32_t>(deleted.size()),
            DataType::i16,
            dim,
            has_norms ? data_file_norm_flags_for_dist(stored_norm_dist_func) : 0u);
        hdr.base.kind = static_cast<uint16_t>(kind);
        CompactIds compact_active_ids;
        ASSERT_EQ(0, compact_active_ids.init(active_ids).code());
        CompactIds compact_deleted_ids;
        std::vector<uint64_t> deleted_ids = deleted;
        ASSERT_EQ(0, compact_deleted_ids.init(deleted_ids).code());
        ASSERT_EQ(0, set_data_header_layout(
            &hdr, compact_active_ids.serialized_size_bytes(), compact_deleted_ids.serialized_size_bytes()).code());

        FILE* f = fopen(path.c_str(), "wb");
        ASSERT_NE(nullptr, f);
        ASSERT_EQ(1u, fwrite(&hdr, sizeof(hdr), 1, f));
        const size_t pad_size = static_cast<size_t>(hdr.data_offset) - sizeof(DataFileHeader);
        if (pad_size > 0) {
            std::vector<uint8_t> pad(pad_size, 0);
            ASSERT_EQ(pad.size(), fwrite(pad.data(), 1, pad.size(), f));
        }

        const DataRecordLayout record_layout =
            compute_data_record_layout(DataType::i16, dim, has_norms);
        for (const auto& item : active) {
            std::vector<int16_t> vec(dim, item.second);
            float norm = 0.0f;
            float* norm_ptr = nullptr;
            if (has_norms) {
                if (stored_norm_dist_func == DistFunc::COS) {
                    norm = inverse_norm(
                        reinterpret_cast<const uint8_t*>(vec.data()), DataType::i16, dim);
                } else if (stored_norm_dist_func == DistFunc::L2) {
                    norm = compute_squared_norm(
                        reinterpret_cast<const uint8_t*>(vec.data()), DataType::i16, dim);
                } else {
                    FAIL() << "write_i16_file: invalid stored norm distance function";
                }
                norm_ptr = &norm;
            }
            ASSERT_EQ(0, write_data_record(
                f,
                reinterpret_cast<const uint8_t*>(vec.data()),
                record_layout,
                norm_ptr,
                "DataMergerTest::write_i16_file").code());
        }
        const DataMetadataLayout metadata_layout = compute_data_metadata_layout(hdr, active.size());
        const size_t ids_pad_size = metadata_layout.vectors_padding;
        if (ids_pad_size > 0) {
            std::vector<uint8_t> pad(ids_pad_size, 0);
            ASSERT_EQ(pad.size(), fwrite(pad.data(), 1, pad.size(), f));
        }
        ASSERT_EQ(0, compact_active_ids.write(f, "DataMergerTest::write_i16_file active ids").code());
        const size_t deleted_ids_pad_size =
            compute_deleted_ids_padding(metadata_layout.ids_trailer_offset, compact_active_ids.serialized_size_bytes());
        if (deleted_ids_pad_size > 0) {
            std::vector<uint8_t> pad(deleted_ids_pad_size, 0);
            ASSERT_EQ(pad.size(), fwrite(pad.data(), 1, pad.size(), f));
        }
        ASSERT_EQ(0, compact_deleted_ids.write(f, "DataMergerTest::write_i16_file deleted ids").code());
        fclose(f);
    }

    DataFileHeader read_header(const std::string& path) {
        DataFileHeader hdr{};
        FILE* f = fopen(path.c_str(), "rb");
        EXPECT_NE(nullptr, f);
        if (!f) {
            return hdr;
        }
        EXPECT_EQ(1u, fread(&hdr, sizeof(hdr), 1, f));
        fclose(f);
        return hdr;
    }

    void expect_inline_norm_layout(const std::string& path) {
        const DataFileHeader hdr = read_header(path);
        EXPECT_TRUE(data_file_has_norms(hdr));
    }

    CompactIdsExtEncoding read_active_ids_encoding(const std::string& path) {
        const DataFileHeader hdr = read_header(path);
        const size_t ids_offset = compute_data_metadata_layout(hdr, hdr.count).ids_trailer_offset;
        FILE* f = fopen(path.c_str(), "rb");
        EXPECT_NE(nullptr, f);
        if (f == nullptr) {
            return CompactIdsExtEncoding::Offsets32;
        }
        EXPECT_EQ(0, fseek(f, static_cast<long>(ids_offset), SEEK_SET));
        CompactIdsHeaderForTest active_hdr{};
        EXPECT_EQ(1u, fread(&active_hdr, sizeof(active_hdr), 1, f));
        fclose(f);
        return static_cast<CompactIdsExtEncoding>(active_hdr.encoding);
    }

    CompactIdsExtEncoding read_deleted_ids_encoding(const std::string& path) {
        const DataFileHeader hdr = read_header(path);
        const size_t ids_offset = compute_data_metadata_layout(hdr, hdr.count).ids_trailer_offset;
        FILE* f = fopen(path.c_str(), "rb");
        EXPECT_NE(nullptr, f);
        if (f == nullptr) {
            return CompactIdsExtEncoding::Offsets32;
        }
        EXPECT_EQ(0, fseek(f, static_cast<long>(ids_offset), SEEK_SET));
        CompactIdsHeaderForTest active_hdr{};
        EXPECT_EQ(1u, fread(&active_hdr, sizeof(active_hdr), 1, f));
        const size_t active_size = sizeof(CompactIdsHeaderForTest) + active_hdr.payload_size;
        const size_t deleted_offset = compute_deleted_ids_offset(ids_offset, active_size);
        EXPECT_EQ(0, fseek(f, static_cast<long>(deleted_offset), SEEK_SET));
        CompactIdsHeaderForTest deleted_hdr{};
        EXPECT_EQ(1u, fread(&deleted_hdr, sizeof(deleted_hdr), 1, f));
        fclose(f);
        return static_cast<CompactIdsExtEncoding>(deleted_hdr.encoding);
    }

    float first_f32(const DataReader& reader, uint64_t id) {
        const auto* p = reinterpret_cast<const float*>(reader.get(id));
        EXPECT_NE(nullptr, p);
        return p ? p[0] : 0.0f;
    }

    void write_input_file(const std::string& path, const std::string& contents) {
        std::ofstream out(path);
        ASSERT_TRUE(out.is_open());
        out << contents;
        ASSERT_FALSE(out.fail());
        out.close();
        ASSERT_FALSE(out.fail());
    }
};

TEST_F(DataMergerTest, MergeDataFileMergesOverrideInsertAndDeletes) {
    const std::string source_path = p("source.data");
    const std::string updater_path = p("updater.data");
    const std::string out_path = p("merged.data");

    write_f32_file(source_path, FileType::Data, {{1, 1.1f}, {3, 3.1f}, {5, 5.1f}}, {});
    write_f32_file(updater_path, FileType::Data, {{2, 20.0f}, {3, 30.0f}, {6, 60.0f}}, {1, 9});

    DataReader source_reader, updater_reader, out_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, updater_reader.init(updater_path).code());

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_data_file(source_reader, updater_reader, out_path).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    EXPECT_EQ(4u, out_reader.count());
    EXPECT_EQ(0u, out_reader.deleted_count());
    EXPECT_EQ(nullptr, out_reader.get(1));
    EXPECT_NE(nullptr, out_reader.get(2));
    EXPECT_NE(nullptr, out_reader.get(3));
    EXPECT_NE(nullptr, out_reader.get(5));
    EXPECT_NE(nullptr, out_reader.get(6));
    EXPECT_FLOAT_EQ(20.0f, first_f32(out_reader, 2));
    EXPECT_FLOAT_EQ(30.0f, first_f32(out_reader, 3));
    EXPECT_FLOAT_EQ(5.1f, first_f32(out_reader, 5));
    EXPECT_FLOAT_EQ(60.0f, first_f32(out_reader, 6));

    const auto hdr = read_header(out_path);
    EXPECT_EQ(static_cast<uint16_t>(FileType::Data), hdr.base.kind);
    EXPECT_EQ(2u, hdr.min_id);
    EXPECT_EQ(6u, hdr.max_id);
    EXPECT_EQ(4u, hdr.count);
}

TEST_F(DataMergerTest, MergeDataFileWithEmptyUpdaterKeepsSource) {
    const std::string source_path = p("source.data");
    const std::string updater_path = p("updater.data");
    const std::string out_path = p("merged.data");

    write_f32_file(source_path, FileType::Data, {{1, 1.1f}, {2, 2.2f}}, {});
    write_f32_file(updater_path, FileType::Data, {}, {});

    DataReader source_reader, updater_reader, out_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, updater_reader.init(updater_path).code());

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_data_file(source_reader, updater_reader, out_path).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    EXPECT_EQ(2u, out_reader.count());
    EXPECT_FLOAT_EQ(1.1f, first_f32(out_reader, 1));
    EXPECT_FLOAT_EQ(2.2f, first_f32(out_reader, 2));
}

TEST_F(DataMergerTest, MergeDataFileFromInputViewMergesAndPreservesCosineValues) {
    const std::string source_path = p("source_cos.data");
    const std::string input_path = p("updates.txt");
    const std::string out_path = p("merged_from_input.data");

    write_f32_file(source_path, FileType::Data, {{1, 3.0f}, {3, 4.0f}, {5, 5.0f}}, {}, kDim, true);
    write_input_file(
        input_path,
        "f32,4\n"
        "1 : []\n"
        "2 : [ 5.0, 5.0, 5.0, 5.0 ]\n"
        "3 : [ 8.0, 8.0, 8.0, 8.0 ]\n"
        "6 : [ 1.0, 2.0, 3.0, 4.0 ]\n");

    DataReader source_reader, out_reader;
    InputReader input_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, input_reader.init(input_path).code());
    InputReaderView view(input_reader, 0, 0);

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_data_file(source_reader, view, out_path, DistFunc::COS).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    expect_inline_norm_layout(out_path);
    ASSERT_EQ(4u, out_reader.count());
    ASSERT_TRUE(out_reader.has_norms());
    EXPECT_EQ(nullptr, out_reader.get(1));
    EXPECT_FLOAT_EQ(5.0f, first_f32(out_reader, 2));
    EXPECT_FLOAT_EQ(8.0f, first_f32(out_reader, 3));
    EXPECT_FLOAT_EQ(5.0f, first_f32(out_reader, 5));
    EXPECT_FLOAT_EQ(1.0f, first_f32(out_reader, 6));
    EXPECT_NEAR(1.0 / (5.0 * std::sqrt(4.0)), static_cast<double>(out_reader.get_norm(0)), 1e-6);
    EXPECT_NEAR(1.0 / (8.0 * std::sqrt(4.0)), static_cast<double>(out_reader.get_norm(1)), 1e-6);
    EXPECT_NEAR(1.0 / (5.0 * std::sqrt(4.0)), static_cast<double>(out_reader.get_norm(2)), 1e-6);
    EXPECT_NEAR(1.0 / std::sqrt(30.0), static_cast<double>(out_reader.get_norm(3)), 1e-6);
}

TEST_F(DataMergerTest, MergeDataFileFromEmptyInputViewKeepsSource) {
    const std::string source_path = p("source.data");
    const std::string input_path = p("updates.txt");
    const std::string out_path = p("merged_empty_view.data");

    write_f32_file(source_path, FileType::Data, {{1, 1.1f}, {3, 3.3f}}, {});
    write_input_file(
        input_path,
        "f32,4\n"
        "10 : [ 10.0, 10.0, 10.0, 10.0 ]\n"
        "11 : []\n");

    DataReader source_reader, out_reader;
    InputReader input_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, input_reader.init(input_path).code());
    InputReaderView empty_view(input_reader, 20, 30);

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_data_file(source_reader, empty_view, out_path, DistFunc::DOT).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    EXPECT_EQ(2u, out_reader.count());
    EXPECT_EQ(0u, out_reader.deleted_count());
    EXPECT_FLOAT_EQ(1.1f, first_f32(out_reader, 1));
    EXPECT_FLOAT_EQ(3.3f, first_f32(out_reader, 3));
}

TEST_F(DataMergerTest, MergeDataFileFromInputViewPreservesL2Norms) {
    const std::string source_path = p("source_l2.data");
    const std::string input_path = p("updates_l2.txt");
    const std::string out_path = p("merged_from_input_l2.data");

    write_f32_file(source_path, FileType::Data, {{1, 3.0f}, {3, 4.0f}, {5, 5.0f}}, {}, kDim, true, DistFunc::L2);
    write_input_file(
        input_path,
        "f32,4\n"
        "1 : []\n"
        "2 : [ 5.0, 5.0, 5.0, 5.0 ]\n"
        "3 : [ 8.0, 8.0, 8.0, 8.0 ]\n"
        "6 : [ 1.0, 2.0, 3.0, 4.0 ]\n");

    DataReader source_reader, out_reader;
    InputReader input_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, input_reader.init(input_path).code());
    InputReaderView view(input_reader, 0, 0);

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_data_file(source_reader, view, out_path, DistFunc::L2).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    expect_inline_norm_layout(out_path);
    ASSERT_EQ(4u, out_reader.count());
    ASSERT_TRUE(out_reader.has_norms());
    EXPECT_EQ(nullptr, out_reader.get(1));
    EXPECT_FLOAT_EQ(5.0f, first_f32(out_reader, 2));
    EXPECT_FLOAT_EQ(8.0f, first_f32(out_reader, 3));
    EXPECT_FLOAT_EQ(5.0f, first_f32(out_reader, 5));
    EXPECT_FLOAT_EQ(1.0f, first_f32(out_reader, 6));
    EXPECT_FLOAT_EQ(100.0f, out_reader.get_norm(0));
    EXPECT_FLOAT_EQ(256.0f, out_reader.get_norm(1));
    EXPECT_FLOAT_EQ(100.0f, out_reader.get_norm(2));
    EXPECT_FLOAT_EQ(30.0f, out_reader.get_norm(3));
}

TEST_F(DataMergerTest, MergeDataFileFromInputViewAddsL2NormsWhenSourceLacksThem) {
    const std::string source_path = p("source_dot_to_l2.data");
    const std::string input_path = p("updates_dot_to_l2.txt");
    const std::string out_path = p("merged_dot_to_l2.data");

    write_f32_file(source_path, FileType::Data, {{1, 3.0f}, {5, 5.0f}}, {});
    write_input_file(
        input_path,
        "f32,4\n"
        "3 : [ 4.0, 4.0, 4.0, 4.0 ]\n");

    DataReader source_reader, out_reader;
    InputReader input_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, input_reader.init(input_path).code());
    InputReaderView view(input_reader, 0, 0);

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_data_file(source_reader, view, out_path, DistFunc::L2).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    expect_inline_norm_layout(out_path);
    ASSERT_TRUE(out_reader.has_norms());
    ASSERT_TRUE(data_file_has_squared_norms(read_header(out_path)));
    EXPECT_FLOAT_EQ(36.0f, out_reader.get_norm(0));
    EXPECT_FLOAT_EQ(64.0f, out_reader.get_norm(1));
    EXPECT_FLOAT_EQ(100.0f, out_reader.get_norm(2));
}

TEST_F(DataMergerTest, MergeDataFileFromInputViewRewritesCosineNormsToL2) {
    const std::string source_path = p("source_cos_to_l2.data");
    const std::string input_path = p("updates_cos_to_l2.txt");
    const std::string out_path = p("merged_cos_to_l2.data");

    write_f32_file(source_path, FileType::Data, {{1, 3.0f}, {5, 5.0f}}, {}, kDim, true, DistFunc::COS);
    write_input_file(
        input_path,
        "f32,4\n"
        "3 : [ 4.0, 4.0, 4.0, 4.0 ]\n");

    DataReader source_reader, out_reader;
    InputReader input_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, input_reader.init(input_path).code());
    InputReaderView view(input_reader, 0, 0);

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_data_file(source_reader, view, out_path, DistFunc::L2).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    expect_inline_norm_layout(out_path);
    ASSERT_TRUE(out_reader.has_norms());
    ASSERT_TRUE(data_file_has_squared_norms(read_header(out_path)));
    EXPECT_FLOAT_EQ(36.0f, out_reader.get_norm(0));
    EXPECT_FLOAT_EQ(64.0f, out_reader.get_norm(1));
    EXPECT_FLOAT_EQ(100.0f, out_reader.get_norm(2));
}

TEST_F(DataMergerTest, MergeDataFileFromInputViewDropsNormsForDot) {
    const std::string source_path = p("source_cos_to_dot.data");
    const std::string input_path = p("updates_cos_to_dot.txt");
    const std::string out_path = p("merged_cos_to_dot.data");

    write_f32_file(source_path, FileType::Data, {{1, 3.0f}, {5, 5.0f}}, {}, kDim, true, DistFunc::COS);
    write_input_file(
        input_path,
        "f32,4\n"
        "3 : [ 4.0, 4.0, 4.0, 4.0 ]\n");

    DataReader source_reader, out_reader;
    InputReader input_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, input_reader.init(input_path).code());
    InputReaderView view(input_reader, 0, 0);

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_data_file(source_reader, view, out_path, DistFunc::DOT).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    ASSERT_FALSE(out_reader.has_norms());
    EXPECT_THROW(static_cast<void>(out_reader.get_norm(0)), std::logic_error);
}

TEST_F(DataMergerTest, MergeDataFileFromInputViewWithOnlyDeletesRemovesMatchingRows) {
    const std::string source_path = p("source.data");
    const std::string input_path = p("deletes.txt");
    const std::string out_path = p("merged_deletes.data");

    write_f32_file(source_path, FileType::Data, {{1, 1.0f}, {2, 2.0f}, {4, 4.0f}}, {});
    write_input_file(
        input_path,
        "f32,4\n"
        "1 : []\n"
        "4 : []\n");

    DataReader source_reader, out_reader;
    InputReader input_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, input_reader.init(input_path).code());
    InputReaderView view(input_reader, 0, 0);

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_data_file(source_reader, view, out_path, DistFunc::DOT).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    EXPECT_EQ(1u, out_reader.count());
    EXPECT_EQ(nullptr, out_reader.get(1));
    EXPECT_FLOAT_EQ(2.0f, first_f32(out_reader, 2));
    EXPECT_EQ(nullptr, out_reader.get(4));
}

TEST_F(DataMergerTest, MergeDataFileFromInputViewWithOnlyInsertsAddsNonOverlappingRows) {
    const std::string source_path = p("source.data");
    const std::string input_path = p("inserts.txt");
    const std::string out_path = p("merged_inserts.data");

    write_f32_file(source_path, FileType::Data, {{5, 5.0f}, {7, 7.0f}}, {});
    write_input_file(
        input_path,
        "f32,4\n"
        "1 : [ 1.0, 1.0, 1.0, 1.0 ]\n"
        "9 : [ 9.0, 9.0, 9.0, 9.0 ]\n");

    DataReader source_reader, out_reader;
    InputReader input_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, input_reader.init(input_path).code());
    InputReaderView view(input_reader, 0, 0);

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_data_file(source_reader, view, out_path, DistFunc::DOT).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    EXPECT_EQ(4u, out_reader.count());
    EXPECT_FLOAT_EQ(1.0f, first_f32(out_reader, 1));
    EXPECT_FLOAT_EQ(5.0f, first_f32(out_reader, 5));
    EXPECT_FLOAT_EQ(7.0f, first_f32(out_reader, 7));
    EXPECT_FLOAT_EQ(9.0f, first_f32(out_reader, 9));
}

TEST_F(DataMergerTest, MergeDataFileFromInputViewWithSpaceSeparatedTextParsesAndMerges) {
    const std::string source_path = p("source_spaces.data");
    const std::string input_path = p("updates_spaces.txt");
    const std::string out_path = p("merged_spaces.data");

    write_f32_file(source_path, FileType::Data, {{2, 2.0f}, {4, 4.0f}}, {});
    write_input_file(
        input_path,
        "f32,4\n"
        "1 : [ 1.0 1.0 1.0 1.0 ]\n"
        "2 : []\n"
        "5 : [ 5.0 5.0 5.0 5.0 ]\n");

    DataReader source_reader, out_reader;
    InputReader input_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, input_reader.init(input_path).code());
    InputReaderView view(input_reader, 0, 0);

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_data_file(source_reader, view, out_path, DistFunc::DOT).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    EXPECT_EQ(3u, out_reader.count());
    EXPECT_FLOAT_EQ(1.0f, first_f32(out_reader, 1));
    EXPECT_EQ(nullptr, out_reader.get(2));
    EXPECT_FLOAT_EQ(4.0f, first_f32(out_reader, 4));
    EXPECT_FLOAT_EQ(5.0f, first_f32(out_reader, 5));
}

TEST_F(DataMergerTest, MergeDataFilePreservesInlineCosineValues) {
    const std::string source_path = p("source_cos.data");
    const std::string updater_path = p("updater_cos.data");
    const std::string out_path = p("merged_cos.data");

    write_f32_file(source_path, FileType::Data, {{1, 3.0f}, {3, 4.0f}}, {}, kDim, true);
    write_f32_file(updater_path, FileType::Data, {{2, 5.0f}, {3, 8.0f}}, {}, kDim, true);

    DataReader source_reader, updater_reader, out_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, updater_reader.init(updater_path).code());

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_data_file(source_reader, updater_reader, out_path).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    expect_inline_norm_layout(out_path);
    ASSERT_TRUE(out_reader.has_norms());
    EXPECT_NEAR(1.0 / (3.0 * std::sqrt(4.0)), static_cast<double>(out_reader.get_norm(0)), 1e-6);
    EXPECT_NEAR(1.0 / (5.0 * std::sqrt(4.0)), static_cast<double>(out_reader.get_norm(1)), 1e-6);
    EXPECT_NEAR(1.0 / (8.0 * std::sqrt(4.0)), static_cast<double>(out_reader.get_norm(2)), 1e-6);
}

TEST_F(DataMergerTest, MergeDataFilePreservesL2NormsAndInlineLayout) {
    const std::string source_path = p("source_l2.data");
    const std::string updater_path = p("updater_l2.data");
    const std::string out_path = p("merged_l2.data");

    write_f32_file(source_path, FileType::Data, {{1, 3.0f}, {3, 4.0f}}, {}, kDim, true, DistFunc::L2);
    write_f32_file(updater_path, FileType::Data, {{2, 5.0f}, {3, 8.0f}}, {}, kDim, true, DistFunc::L2);

    DataReader source_reader, updater_reader, out_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, updater_reader.init(updater_path).code());

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_data_file(source_reader, updater_reader, out_path).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    expect_inline_norm_layout(out_path);
    ASSERT_TRUE(out_reader.has_norms());
    ASSERT_TRUE(data_file_has_squared_norms(read_header(out_path)));
    EXPECT_FLOAT_EQ(36.0f, out_reader.get_norm(0));
    EXPECT_FLOAT_EQ(100.0f, out_reader.get_norm(1));
    EXPECT_FLOAT_EQ(256.0f, out_reader.get_norm(2));
}

TEST_F(DataMergerTest, MergeDataFileAllDeletedProducesEmptyFile) {
    const std::string source_path = p("source.data");
    const std::string updater_path = p("updater.data");
    const std::string out_path = p("merged.data");

    write_f32_file(source_path, FileType::Data, {{10, 1.0f}, {11, 2.0f}}, {});
    write_f32_file(updater_path, FileType::Data, {}, {10, 11});

    DataReader source_reader, updater_reader, out_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, updater_reader.init(updater_path).code());

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_data_file(source_reader, updater_reader, out_path).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    EXPECT_EQ(0u, out_reader.count());
    EXPECT_EQ(0u, out_reader.deleted_count());

    const auto hdr = read_header(out_path);
    EXPECT_EQ(0u, hdr.count);
    EXPECT_EQ(0u, hdr.deleted_count);
    EXPECT_EQ(0u, hdr.min_id);
    EXPECT_EQ(0u, hdr.max_id);
}

TEST_F(DataMergerTest, MergeDataFileRejectsIncompatibleType) {
    const std::string source_path = p("source.data");
    const std::string updater_path = p("updater.data");
    const std::string out_path = p("merged.data");

    write_f32_file(source_path, FileType::Data, {{1, 1.0f}}, {});
    write_i16_file(updater_path, FileType::Data, {{1, 1}}, {});

    DataReader source_reader, updater_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, updater_reader.init(updater_path).code());

    DataMerger merger;
    const auto ret = merger.merge_data_file(source_reader, updater_reader, out_path);
    EXPECT_NE(0, ret.code());
    EXPECT_FALSE(fs::exists(out_path));
}

TEST_F(DataMergerTest, MergeDataFileRejectsIncompatibleDim) {
    const std::string source_path = p("source.data");
    const std::string updater_path = p("updater.data");
    const std::string out_path = p("merged.data");

    write_f32_file(source_path, FileType::Data, {{1, 1.0f}}, {}, 4);
    write_f32_file(updater_path, FileType::Data, {{1, 1.0f}}, {}, 8);

    DataReader source_reader, updater_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, updater_reader.init(updater_path).code());

    DataMerger merger;
    const auto ret = merger.merge_data_file(source_reader, updater_reader, out_path);
    EXPECT_NE(0, ret.code());
    EXPECT_FALSE(fs::exists(out_path));
}

TEST_F(DataMergerTest, MergeDataFileRejectsIncompatibleNormKinds) {
    const std::string source_path = p("source_cos.data");
    const std::string updater_path = p("updater_l2.data");
    const std::string out_path = p("merged_norm_kind.data");

    write_f32_file(source_path, FileType::Data, {{1, 1.0f}}, {}, 4, true, DistFunc::COS);
    write_f32_file(updater_path, FileType::Data, {{2, 2.0f}}, {}, 4, true, DistFunc::L2);

    DataReader source_reader, updater_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, updater_reader.init(updater_path).code());

    DataMerger merger;
    const auto ret = merger.merge_data_file(source_reader, updater_reader, out_path);
    EXPECT_NE(0, ret.code());
    EXPECT_NE(std::string::npos, ret.message().find("incompatible norm layout"));
    EXPECT_FALSE(fs::exists(out_path));
}

TEST_F(DataMergerTest, MergeDataFileRejectsUpdatedIdAlsoDeletedAndCleansOutput) {
    const std::string source_path = p("source.data");
    const std::string updater_path = p("updater.data");
    const std::string out_path = p("merged.data");

    write_f32_file(source_path, FileType::Data, {{1, 1.0f}}, {});
    write_f32_file(updater_path, FileType::Data, {{2, 2.0f}}, {2});

    DataReader source_reader, updater_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, updater_reader.init(updater_path).code());

    DataMerger merger;
    const auto ret = merger.merge_data_file(source_reader, updater_reader, out_path);
    EXPECT_NE(0, ret.code());
    EXPECT_FALSE(fs::exists(out_path));
}

TEST_F(DataMergerTest, WriteI16FileSupportsInlineNorms) {
    const std::string path = p("i16_norms.data");

    write_i16_file(path, FileType::Data, {{7, 3}, {8, 4}}, {}, 4, true, DistFunc::L2);

    DataReader reader;
    ASSERT_EQ(0, reader.init(path).code());
    ASSERT_TRUE(reader.has_norms());
    EXPECT_TRUE(reader.has_matching_stored_norms(DistFunc::L2));
    EXPECT_FALSE(reader.has_matching_stored_norms(DistFunc::COS));
    EXPECT_GT(reader.stride(), reader.size());
    EXPECT_FLOAT_EQ(36.0f, reader.get_norm(0));
    EXPECT_FLOAT_EQ(64.0f, reader.get_norm(1));
}

TEST_F(DataMergerTest, MergeDataFileFromInputViewRejectsIncompatibleDim) {
    const std::string source_path = p("source.data");
    const std::string input_path = p("updates_bad_dim.txt");
    const std::string out_path = p("merged_bad_dim.data");

    write_f32_file(source_path, FileType::Data, {{1, 1.0f}}, {}, 4);
    write_input_file(
        input_path,
        "f32,8\n"
        "2 : [ 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 ]\n");

    DataReader source_reader;
    InputReader input_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, input_reader.init(input_path).code());
    InputReaderView view(input_reader, 0, 0);

    DataMerger merger;
    const auto ret = merger.merge_data_file(source_reader, view, out_path, DistFunc::DOT);
    EXPECT_NE(0, ret.code());
    EXPECT_FALSE(fs::exists(out_path));
}

TEST_F(DataMergerTest, MergeDataFileFromInputViewRejectsActiveIdSpanBeyondCompactIdsRange) {
    const std::string source_path = p("source_wide_span.data");
    const std::string input_path = p("updates_wide_span.txt");
    const std::string out_path = p("merged_wide_span.data");

    write_f32_file(source_path, FileType::Data, {}, {});
    write_input_file(
        input_path,
        "f32,4\n"
        "0 : [ 1.0, 1.0, 1.0, 1.0 ]\n"
        "4294967296 : [ 2.0, 2.0, 2.0, 2.0 ]\n");

    DataReader source_reader;
    InputReader input_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, input_reader.init(input_path).code());
    InputReaderView view(input_reader, 0, 0);

    DataMerger merger;
    const Ret ret = merger.merge_data_file(source_reader, view, out_path, DistFunc::DOT);
    EXPECT_NE(0, ret.code());
    EXPECT_NE(std::string::npos,
              ret.message().find("active ids: data file range exceeds uint32_t"));
    EXPECT_FALSE(fs::exists(out_path));
}

TEST_F(DataMergerTest, MergeDeltaFileFromInputViewRejectsDeletedIdSpanBeyondCompactIdsRange) {
    const std::string source_path = p("source_deleted_wide_span.delta");
    const std::string input_path = p("updates_deleted_wide_span.txt");
    const std::string out_path = p("merged_deleted_wide_span.delta");

    write_f32_file(source_path, FileType::Data, {}, {});
    write_input_file(
        input_path,
        "f32,4\n"
        "0 : []\n"
        "4294967296 : []\n");

    DataReader source_reader;
    InputReader input_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, input_reader.init(input_path).code());
    InputReaderView view(input_reader, 0, 0);

    DataMerger merger;
    const Ret ret = merger.merge_delta_file(source_reader, view, out_path, DistFunc::DOT);
    EXPECT_NE(0, ret.code());
    EXPECT_NE(std::string::npos,
              ret.message().find("deleted ids: id range exceeds uint32_t"));
    EXPECT_FALSE(fs::exists(out_path));
}

TEST_F(DataMergerTest, MergeDataFileDenseActiveIdsUseBitsetEncoding) {
    const std::string source_path = p("source_dense.data");
    const std::string input_path = p("updates_dense.txt");
    const std::string out_path = p("merged_dense.data");

    write_f32_file(source_path, FileType::Data, {}, {});
    write_input_file(input_path, "f32,4\n");
    std::ofstream append(input_path, std::ios::app);
    ASSERT_TRUE(append.is_open());
    for (uint64_t i = 0; i < 9000; ++i) {
        const uint64_t id = 20000 + i * 2;
        append << id << " : [ 1.0, 1.0, 1.0, 1.0 ]\n";
    }
    append.close();

    DataReader source_reader;
    InputReader input_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, input_reader.init(input_path).code());
    InputReaderView view(input_reader, 0, 0);

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_data_file(source_reader, view, out_path, DistFunc::DOT).code());
    EXPECT_EQ(CompactIdsExtEncoding::Bitset, read_active_ids_encoding(out_path));
}

TEST_F(DataMergerTest, MergeDeltaFileDenseDeletedIdsUseBitsetEncoding) {
    const std::string source_path = p("source_dense_deleted.delta");
    const std::string updater_path = p("updater_dense_deleted.delta");
    const std::string out_path = p("merged_dense_deleted.delta");

    std::vector<std::pair<uint64_t, float>> source_active = {{1, 1.0f}};
    std::vector<uint64_t> updater_deleted;
    updater_deleted.reserve(9000);
    for (uint64_t i = 0; i < 9000; ++i) {
        updater_deleted.push_back(20000 + i * 2);
    }

    write_f32_file(source_path, FileType::Data, source_active, {});
    write_f32_file(updater_path, FileType::Data, {}, updater_deleted);

    DataReader source_reader, updater_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, updater_reader.init(updater_path).code());

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_delta_file(source_reader, updater_reader, out_path).code());
    EXPECT_EQ(CompactIdsExtEncoding::Bitset, read_deleted_ids_encoding(out_path));
}

TEST_F(DataMergerTest, MergeDeltaFileMergesRecordsAndDeletes) {
    const std::string source_path = p("source.delta");
    const std::string updater_path = p("updater.delta");
    const std::string out_path = p("merged.delta");

    write_f32_file(source_path, FileType::Data, {{2, 2.0f}, {4, 4.0f}}, {1, 5});
    write_f32_file(updater_path, FileType::Data, {{1, 10.0f}, {3, 30.0f}, {4, 40.0f}, {6, 60.0f}}, {2, 7});

    DataReader source_reader, updater_reader, out_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, updater_reader.init(updater_path).code());

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_delta_file(source_reader, updater_reader, out_path).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    EXPECT_EQ(4u, out_reader.count());
    EXPECT_EQ(3u, out_reader.deleted_count());
    EXPECT_FLOAT_EQ(10.0f, first_f32(out_reader, 1));
    EXPECT_FLOAT_EQ(30.0f, first_f32(out_reader, 3));
    EXPECT_FLOAT_EQ(40.0f, first_f32(out_reader, 4));
    EXPECT_FLOAT_EQ(60.0f, first_f32(out_reader, 6));
    EXPECT_EQ(nullptr, out_reader.get(2));
    EXPECT_EQ(nullptr, out_reader.get(5));
    EXPECT_EQ(nullptr, out_reader.get(7));

    std::vector<uint64_t> deleted;
    for (size_t i = 0; i < out_reader.deleted_count(); ++i) {
        deleted.push_back(out_reader.deleted_id(i));
    }
    EXPECT_EQ((std::vector<uint64_t>{2, 5, 7}), deleted);

    const auto hdr = read_header(out_path);
    EXPECT_EQ(static_cast<uint16_t>(FileType::Data), hdr.base.kind);
    EXPECT_EQ(1u, hdr.min_id);
    EXPECT_EQ(6u, hdr.max_id);
    EXPECT_EQ(4u, hdr.count);
    EXPECT_EQ(3u, hdr.deleted_count);
}

TEST_F(DataMergerTest, MergeDeltaFileCursorEdgeCaseEmptyDeleteStreams) {
    const std::string source_path = p("source_empty_deletes.delta");
    const std::string updater_path = p("updater_empty_deletes.delta");
    const std::string out_path = p("merged_empty_deletes.delta");

    write_f32_file(source_path, FileType::Data, {{10, 10.0f}}, {});
    write_f32_file(updater_path, FileType::Data, {{11, 11.0f}}, {});

    DataReader source_reader, updater_reader, out_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, updater_reader.init(updater_path).code());

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_delta_file(source_reader, updater_reader, out_path).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    EXPECT_EQ(2u, out_reader.count());
    EXPECT_EQ(0u, out_reader.deleted_count());
    EXPECT_FLOAT_EQ(10.0f, first_f32(out_reader, 10));
    EXPECT_FLOAT_EQ(11.0f, first_f32(out_reader, 11));
}

TEST_F(DataMergerTest, MergeDeltaFileCursorEdgeCaseSingleDeletedEntry) {
    const std::string source_path = p("source_single_deleted.delta");
    const std::string updater_path = p("updater_single_deleted.delta");
    const std::string out_path = p("merged_single_deleted.delta");

    write_f32_file(source_path, FileType::Data, {}, {42});
    write_f32_file(updater_path, FileType::Data, {}, {});

    DataReader source_reader, updater_reader, out_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, updater_reader.init(updater_path).code());

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_delta_file(source_reader, updater_reader, out_path).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    EXPECT_EQ(0u, out_reader.count());
    ASSERT_EQ(1u, out_reader.deleted_count());
    EXPECT_EQ(42u, out_reader.deleted_id(0));
}

TEST_F(DataMergerTest, MergeDeltaFileCursorEdgeCaseAllSourceDeletesResurrected) {
    const std::string source_path = p("source_resurrect_all.delta");
    const std::string updater_path = p("updater_resurrect_all.delta");
    const std::string out_path = p("merged_resurrect_all.delta");

    write_f32_file(source_path, FileType::Data, {{8, 8.0f}}, {1, 2, 3});
    write_f32_file(updater_path, FileType::Data, {{1, 10.0f}, {2, 20.0f}, {3, 30.0f}, {9, 90.0f}}, {});

    DataReader source_reader, updater_reader, out_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, updater_reader.init(updater_path).code());

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_delta_file(source_reader, updater_reader, out_path).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    EXPECT_EQ(5u, out_reader.count());
    EXPECT_EQ(0u, out_reader.deleted_count());
    EXPECT_FLOAT_EQ(10.0f, first_f32(out_reader, 1));
    EXPECT_FLOAT_EQ(20.0f, first_f32(out_reader, 2));
    EXPECT_FLOAT_EQ(30.0f, first_f32(out_reader, 3));
    EXPECT_FLOAT_EQ(8.0f, first_f32(out_reader, 8));
    EXPECT_FLOAT_EQ(90.0f, first_f32(out_reader, 9));
}

TEST_F(DataMergerTest, MergeDeltaFileFromInputViewMergesRecordsAndDeletes) {
    const std::string source_path = p("source.delta");
    const std::string input_path = p("updates.txt");
    const std::string out_path = p("merged_from_input.delta");

    write_f32_file(source_path, FileType::Data, {{2, 2.0f}, {4, 4.0f}}, {1, 5});
    write_input_file(
        input_path,
        "f32,4\n"
        "1 : [ 10.0, 10.0, 10.0, 10.0 ]\n"
        "2 : []\n"
        "3 : [ 30.0, 30.0, 30.0, 30.0 ]\n"
        "4 : [ 40.0, 40.0, 40.0, 40.0 ]\n"
        "6 : [ 60.0, 60.0, 60.0, 60.0 ]\n"
        "7 : []\n");

    DataReader source_reader, out_reader;
    InputReader input_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, input_reader.init(input_path).code());
    InputReaderView view(input_reader, 0, 0);

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_delta_file(source_reader, view, out_path, DistFunc::DOT).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    EXPECT_EQ(4u, out_reader.count());
    EXPECT_EQ(3u, out_reader.deleted_count());
    EXPECT_FLOAT_EQ(10.0f, first_f32(out_reader, 1));
    EXPECT_FLOAT_EQ(30.0f, first_f32(out_reader, 3));
    EXPECT_FLOAT_EQ(40.0f, first_f32(out_reader, 4));
    EXPECT_FLOAT_EQ(60.0f, first_f32(out_reader, 6));
    EXPECT_EQ(nullptr, out_reader.get(2));
    EXPECT_EQ(nullptr, out_reader.get(5));
    EXPECT_EQ(nullptr, out_reader.get(7));

    std::vector<uint64_t> deleted;
    for (size_t i = 0; i < out_reader.deleted_count(); ++i) {
        deleted.push_back(out_reader.deleted_id(i));
    }
    EXPECT_EQ((std::vector<uint64_t>{2, 5, 7}), deleted);
}

TEST_F(DataMergerTest, MergeDeltaFileFromInputViewCursorEdgeCaseAllSourceDeletesResurrected) {
    const std::string source_path = p("source_input_resurrect_all.delta");
    const std::string input_path = p("updates_input_resurrect_all.txt");
    const std::string out_path = p("merged_input_resurrect_all.delta");

    write_f32_file(source_path, FileType::Data, {{8, 8.0f}}, {1, 2, 3});
    write_input_file(
        input_path,
        "f32,4\n"
        "1 : [ 10.0, 10.0, 10.0, 10.0 ]\n"
        "2 : [ 20.0, 20.0, 20.0, 20.0 ]\n"
        "3 : [ 30.0, 30.0, 30.0, 30.0 ]\n"
        "9 : [ 90.0, 90.0, 90.0, 90.0 ]\n"
        "10 : []\n");

    DataReader source_reader, out_reader;
    InputReader input_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, input_reader.init(input_path).code());
    InputReaderView view(input_reader, 0, 0);

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_delta_file(source_reader, view, out_path, DistFunc::DOT).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    EXPECT_EQ(5u, out_reader.count());
    ASSERT_EQ(1u, out_reader.deleted_count());
    EXPECT_EQ(10u, out_reader.deleted_id(0));
    EXPECT_FLOAT_EQ(10.0f, first_f32(out_reader, 1));
    EXPECT_FLOAT_EQ(20.0f, first_f32(out_reader, 2));
    EXPECT_FLOAT_EQ(30.0f, first_f32(out_reader, 3));
    EXPECT_FLOAT_EQ(8.0f, first_f32(out_reader, 8));
    EXPECT_FLOAT_EQ(90.0f, first_f32(out_reader, 9));
}

TEST_F(DataMergerTest, MergeDeltaFileFromInputViewPreservesCosineValues) {
    const std::string source_path = p("source_cos.delta");
    const std::string input_path = p("updates_cos.txt");
    const std::string out_path = p("merged_from_input_cos.delta");

    write_f32_file(source_path, FileType::Data, {{2, 2.0f}, {4, 4.0f}}, {1}, kDim, true);
    write_input_file(
        input_path,
        "f32,4\n"
        "3 : [ 3.0, 3.0, 3.0, 3.0 ]\n"
        "4 : [ 1.0, 2.0, 3.0, 4.0 ]\n"
        "5 : []\n");

    DataReader source_reader, out_reader;
    InputReader input_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, input_reader.init(input_path).code());
    InputReaderView view(input_reader, 0, 0);

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_delta_file(source_reader, view, out_path, DistFunc::COS).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    expect_inline_norm_layout(out_path);
    ASSERT_TRUE(out_reader.has_norms());
    ASSERT_EQ(3u, out_reader.count());
    EXPECT_NEAR(1.0 / (2.0 * std::sqrt(4.0)), static_cast<double>(out_reader.get_norm(0)), 1e-6);
    EXPECT_NEAR(1.0 / (3.0 * std::sqrt(4.0)), static_cast<double>(out_reader.get_norm(1)), 1e-6);
    EXPECT_NEAR(1.0 / std::sqrt(30.0), static_cast<double>(out_reader.get_norm(2)), 1e-6);
    EXPECT_EQ((std::vector<uint64_t>{1u, 5u}),
        [&]() {
            std::vector<uint64_t> ids;
            for (size_t i = 0; i < out_reader.deleted_count(); ++i) {
                ids.push_back(out_reader.deleted_id(i));
            }
            return ids;
        }());
}

TEST_F(DataMergerTest, MergeDeltaFileFromInputViewWithOnlyDeletesProducesNoActiveIds) {
    const std::string source_path = p("source.delta");
    const std::string input_path = p("deletes.txt");
    const std::string out_path = p("merged_delete_only.delta");

    write_f32_file(source_path, FileType::Data, {{1, 1.0f}}, {2});
    write_input_file(
        input_path,
        "f32,4\n"
        "1 : []\n"
        "3 : []\n");

    DataReader source_reader, out_reader;
    InputReader input_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, input_reader.init(input_path).code());
    InputReaderView view(input_reader, 0, 0);

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_delta_file(source_reader, view, out_path, DistFunc::DOT).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    EXPECT_EQ(0u, out_reader.count());
    EXPECT_EQ(3u, out_reader.deleted_count());
    EXPECT_EQ(1u, out_reader.deleted_id(0));
    EXPECT_EQ(2u, out_reader.deleted_id(1));
    EXPECT_EQ(3u, out_reader.deleted_id(2));
}

TEST_F(DataMergerTest, MergeDeltaFileFromInputViewPreservesL2Norms) {
    const std::string source_path = p("source_l2.delta");
    const std::string input_path = p("updates_l2.txt");
    const std::string out_path = p("merged_from_input_l2.delta");

    write_f32_file(source_path, FileType::Data, {{2, 2.0f}, {4, 4.0f}}, {1}, kDim, true, DistFunc::L2);
    write_input_file(
        input_path,
        "f32,4\n"
        "3 : [ 3.0, 3.0, 3.0, 3.0 ]\n"
        "4 : [ 1.0, 2.0, 3.0, 4.0 ]\n"
        "5 : []\n");

    DataReader source_reader, out_reader;
    InputReader input_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, input_reader.init(input_path).code());
    InputReaderView view(input_reader, 0, 0);

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_delta_file(source_reader, view, out_path, DistFunc::L2).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    expect_inline_norm_layout(out_path);
    ASSERT_TRUE(out_reader.has_norms());
    ASSERT_EQ(3u, out_reader.count());
    EXPECT_FLOAT_EQ(16.0f, out_reader.get_norm(0));
    EXPECT_FLOAT_EQ(36.0f, out_reader.get_norm(1));
    EXPECT_FLOAT_EQ(30.0f, out_reader.get_norm(2));
    EXPECT_EQ((std::vector<uint64_t>{1u, 5u}),
        [&]() {
            std::vector<uint64_t> ids;
            for (size_t i = 0; i < out_reader.deleted_count(); ++i) {
                ids.push_back(out_reader.deleted_id(i));
            }
            return ids;
        }());
}

TEST_F(DataMergerTest, MergeDeltaFileFromInputViewAddsCosineNormsWhenSourceLacksThem) {
    const std::string source_path = p("source_dot_to_cos.delta");
    const std::string input_path = p("updates_dot_to_cos.txt");
    const std::string out_path = p("merged_dot_to_cos.delta");

    write_f32_file(source_path, FileType::Data, {{2, 2.0f}, {4, 4.0f}}, {1});
    write_input_file(
        input_path,
        "f32,4\n"
        "3 : [ 3.0, 3.0, 3.0, 3.0 ]\n"
        "4 : [ 1.0, 2.0, 3.0, 4.0 ]\n"
        "5 : []\n");

    DataReader source_reader, out_reader;
    InputReader input_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, input_reader.init(input_path).code());
    InputReaderView view(input_reader, 0, 0);

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_delta_file(source_reader, view, out_path, DistFunc::COS).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    expect_inline_norm_layout(out_path);
    ASSERT_TRUE(out_reader.has_norms());
    ASSERT_TRUE(data_file_has_cosine_inv_norms(read_header(out_path)));
    ASSERT_EQ(3u, out_reader.count());
    EXPECT_NEAR(1.0 / (2.0 * std::sqrt(4.0)), static_cast<double>(out_reader.get_norm(0)), 1e-6);
    EXPECT_NEAR(1.0 / (3.0 * std::sqrt(4.0)), static_cast<double>(out_reader.get_norm(1)), 1e-6);
    EXPECT_NEAR(1.0 / std::sqrt(30.0), static_cast<double>(out_reader.get_norm(2)), 1e-6);
}

TEST_F(DataMergerTest, MergeDeltaFileFromInputViewRejectsIncompatibleType) {
    const std::string source_path = p("source.delta");
    const std::string input_path = p("updates_i16.txt");
    const std::string out_path = p("merged_bad_type.delta");

    write_f32_file(source_path, FileType::Data, {{1, 1.0f}}, {});
    write_input_file(
        input_path,
        "i16,4\n"
        "2 : [ 2, 2, 2, 2 ]\n");

    DataReader source_reader;
    InputReader input_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, input_reader.init(input_path).code());
    InputReaderView view(input_reader, 0, 0);

    DataMerger merger;
    const auto ret = merger.merge_delta_file(source_reader, view, out_path, DistFunc::DOT);
    EXPECT_NE(0, ret.code());
    EXPECT_FALSE(fs::exists(out_path));
}

TEST_F(DataMergerTest, MergeDeltaFilePreservesInlineCosineValues) {
    const std::string source_path = p("source_cos.delta");
    const std::string updater_path = p("updater_cos.delta");
    const std::string out_path = p("merged_cos.delta");

    write_f32_file(source_path, FileType::Data, {{2, 2.0f}, {4, 4.0f}}, {1}, kDim, true);
    write_f32_file(updater_path, FileType::Data, {{3, 3.0f}, {4, 8.0f}}, {5}, kDim, true);

    DataReader source_reader, updater_reader, out_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, updater_reader.init(updater_path).code());

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_delta_file(source_reader, updater_reader, out_path).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    expect_inline_norm_layout(out_path);
    ASSERT_TRUE(out_reader.has_norms());
    ASSERT_EQ(3u, out_reader.count());
    EXPECT_NEAR(1.0 / (2.0 * std::sqrt(4.0)), static_cast<double>(out_reader.get_norm(0)), 1e-6);
    EXPECT_NEAR(1.0 / (3.0 * std::sqrt(4.0)), static_cast<double>(out_reader.get_norm(1)), 1e-6);
    EXPECT_NEAR(1.0 / (8.0 * std::sqrt(4.0)), static_cast<double>(out_reader.get_norm(2)), 1e-6);
    EXPECT_EQ((std::vector<uint64_t>{1u, 5u}),
        [&]() {
            std::vector<uint64_t> ids;
            for (size_t i = 0; i < out_reader.deleted_count(); ++i) {
                ids.push_back(out_reader.deleted_id(i));
            }
            return ids;
        }());
}

TEST_F(DataMergerTest, MergeDeltaFilePreservesL2NormsAndInlineLayout) {
    const std::string source_path = p("source_l2.delta");
    const std::string updater_path = p("updater_l2.delta");
    const std::string out_path = p("merged_l2.delta");

    write_f32_file(source_path, FileType::Data, {{2, 2.0f}, {4, 4.0f}}, {1}, kDim, true, DistFunc::L2);
    write_f32_file(updater_path, FileType::Data, {{3, 3.0f}, {4, 8.0f}}, {5}, kDim, true, DistFunc::L2);

    DataReader source_reader, updater_reader, out_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, updater_reader.init(updater_path).code());

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_delta_file(source_reader, updater_reader, out_path).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    expect_inline_norm_layout(out_path);
    ASSERT_TRUE(out_reader.has_norms());
    ASSERT_TRUE(data_file_has_squared_norms(read_header(out_path)));
    ASSERT_EQ(3u, out_reader.count());
    EXPECT_FLOAT_EQ(16.0f, out_reader.get_norm(0));
    EXPECT_FLOAT_EQ(36.0f, out_reader.get_norm(1));
    EXPECT_FLOAT_EQ(256.0f, out_reader.get_norm(2));
    EXPECT_EQ((std::vector<uint64_t>{1u, 5u}),
        [&]() {
            std::vector<uint64_t> ids;
            for (size_t i = 0; i < out_reader.deleted_count(); ++i) {
                ids.push_back(out_reader.deleted_id(i));
            }
            return ids;
        }());
}

TEST_F(DataMergerTest, MergeDeltaFileDeleteOnlyProducesNoActiveIds) {
    const std::string source_path = p("source.delta");
    const std::string updater_path = p("updater.delta");
    const std::string out_path = p("merged.delta");

    write_f32_file(source_path, FileType::Data, {{1, 1.0f}}, {2});
    write_f32_file(updater_path, FileType::Data, {}, {1, 3});

    DataReader source_reader, updater_reader, out_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, updater_reader.init(updater_path).code());

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_delta_file(source_reader, updater_reader, out_path).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    EXPECT_EQ(0u, out_reader.count());
    EXPECT_EQ(3u, out_reader.deleted_count());
    EXPECT_EQ(1u, out_reader.deleted_id(0));
    EXPECT_EQ(2u, out_reader.deleted_id(1));
    EXPECT_EQ(3u, out_reader.deleted_id(2));

    const auto hdr = read_header(out_path);
    EXPECT_EQ(static_cast<uint16_t>(FileType::Data), hdr.base.kind);
    EXPECT_EQ(0u, hdr.min_id);
    EXPECT_EQ(0u, hdr.max_id);
    EXPECT_EQ(0u, hdr.count);
    EXPECT_EQ(3u, hdr.deleted_count);
}

TEST_F(DataMergerTest, MergeDeltaFileResurrectsPreviouslyDeletedId) {
    const std::string source_path = p("source.delta");
    const std::string updater_path = p("updater.delta");
    const std::string out_path = p("merged.delta");

    write_f32_file(source_path, FileType::Data, {}, {42});
    write_f32_file(updater_path, FileType::Data, {{42, 42.5f}}, {});

    DataReader source_reader, updater_reader, out_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, updater_reader.init(updater_path).code());

    DataMerger merger;
    ASSERT_EQ(0, merger.merge_delta_file(source_reader, updater_reader, out_path).code());

    ASSERT_EQ(0, out_reader.init(out_path).code());
    EXPECT_EQ(1u, out_reader.count());
    EXPECT_EQ(0u, out_reader.deleted_count());
    EXPECT_FLOAT_EQ(42.5f, first_f32(out_reader, 42));
}

TEST_F(DataMergerTest, MergeDeltaFileRejectsIncompatibleType) {
    const std::string source_path = p("source.delta");
    const std::string updater_path = p("updater.delta");
    const std::string out_path = p("merged.delta");

    write_f32_file(source_path, FileType::Data, {{1, 1.0f}}, {});
    write_i16_file(updater_path, FileType::Data, {{1, 1}}, {});

    DataReader source_reader, updater_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, updater_reader.init(updater_path).code());

    DataMerger merger;
    const auto ret = merger.merge_delta_file(source_reader, updater_reader, out_path);
    EXPECT_NE(0, ret.code());
    EXPECT_FALSE(fs::exists(out_path));
}

TEST_F(DataMergerTest, MergeDeltaFileRejectsIncompatibleDim) {
    const std::string source_path = p("source.delta");
    const std::string updater_path = p("updater.delta");
    const std::string out_path = p("merged.delta");

    write_f32_file(source_path, FileType::Data, {{1, 1.0f}}, {}, 4);
    write_f32_file(updater_path, FileType::Data, {{1, 1.0f}}, {}, 8);

    DataReader source_reader, updater_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, updater_reader.init(updater_path).code());

    DataMerger merger;
    const auto ret = merger.merge_delta_file(source_reader, updater_reader, out_path);
    EXPECT_NE(0, ret.code());
    EXPECT_FALSE(fs::exists(out_path));
}

TEST_F(DataMergerTest, MergeDeltaFileRejectsUpdatedIdAlsoDeletedAndCleansOutput) {
    const std::string source_path = p("source.delta");
    const std::string updater_path = p("updater.delta");
    const std::string out_path = p("merged.delta");

    write_f32_file(source_path, FileType::Data, {{1, 1.0f}}, {});
    write_f32_file(updater_path, FileType::Data, {{8, 8.0f}}, {8});

    DataReader source_reader, updater_reader;
    ASSERT_EQ(0, source_reader.init(source_path).code());
    ASSERT_EQ(0, updater_reader.init(updater_path).code());

    DataMerger merger;
    const auto ret = merger.merge_delta_file(source_reader, updater_reader, out_path);
    EXPECT_NE(0, ret.code());
    EXPECT_FALSE(fs::exists(out_path));
}
