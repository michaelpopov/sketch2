#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <unistd.h>
#include <vector>
#include "core/storage/data_file.h"
#include "core/storage/data_file_layout.h"
#include "core/storage/data_merger.h"
#include "core/storage/data_reader.h"

using namespace sketch2;
namespace fs = std::filesystem;

class DataMergerTest : public ::testing::Test {
protected:
    static constexpr uint16_t kDim = 4;
    std::string base_dir_;

    void SetUp() override {
        base_dir_ = "/tmp/sketch2_utest_dm_" + std::to_string(getpid());
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
                        bool has_cosine_inv_norms = false) {
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
            static_cast<uint32_t>(active.size()),
            static_cast<uint32_t>(deleted.size()),
            DataType::f32,
            dim,
            has_cosine_inv_norms);
        hdr.base.kind = static_cast<uint16_t>(kind);

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
            ASSERT_EQ(0, write_vector_record(f, reinterpret_cast<const uint8_t*>(vec.data()),
                vec.size() * sizeof(float), hdr.vector_stride, "DataMergerTest::write_f32_file").code());
        }
        if (has_cosine_inv_norms) {
            std::vector<float> cosine_inv_norms;
            cosine_inv_norms.reserve(active.size());
            for (const auto& item : active) {
                std::vector<float> vec(dim, item.second);
                cosine_inv_norms.push_back(compute_cosine_inverse_norm(
                    reinterpret_cast<const uint8_t*>(vec.data()), DataType::f32, dim));
            }
            ASSERT_EQ(0, write_f32_array(f, cosine_inv_norms,
                "DataMergerTest::write_f32_file cosine").code());
        }
        const size_t ids_pad_size = compute_ids_layout(hdr, active.size()).ids_padding;
        if (ids_pad_size > 0) {
            std::vector<uint8_t> pad(ids_pad_size, 0);
            ASSERT_EQ(pad.size(), fwrite(pad.data(), 1, pad.size(), f));
        }
        for (const auto& item : active) {
            const uint64_t id = item.first;
            ASSERT_EQ(1u, fwrite(&id, sizeof(id), 1, f));
        }
        if (!deleted.empty()) {
            ASSERT_EQ(deleted.size(), fwrite(deleted.data(), sizeof(uint64_t), deleted.size(), f));
        }
        fclose(f);
    }

    void write_i16_file(const std::string& path,
                        FileType kind,
                        const std::vector<std::pair<uint64_t, int16_t>>& active,
                        const std::vector<uint64_t>& deleted,
                        uint16_t dim = kDim) {
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
            static_cast<uint32_t>(active.size()),
            static_cast<uint32_t>(deleted.size()),
            DataType::i16,
            dim);
        hdr.base.kind = static_cast<uint16_t>(kind);

        FILE* f = fopen(path.c_str(), "wb");
        ASSERT_NE(nullptr, f);
        ASSERT_EQ(1u, fwrite(&hdr, sizeof(hdr), 1, f));
        const size_t pad_size = static_cast<size_t>(hdr.data_offset) - sizeof(DataFileHeader);
        if (pad_size > 0) {
            std::vector<uint8_t> pad(pad_size, 0);
            ASSERT_EQ(pad.size(), fwrite(pad.data(), 1, pad.size(), f));
        }

        for (const auto& item : active) {
            std::vector<int16_t> vec(dim, item.second);
            ASSERT_EQ(0, write_vector_record(f, reinterpret_cast<const uint8_t*>(vec.data()),
                vec.size() * sizeof(int16_t), hdr.vector_stride, "DataMergerTest::write_i16_file").code());
        }
        const size_t ids_pad_size = compute_ids_layout(hdr, active.size()).ids_padding;
        if (ids_pad_size > 0) {
            std::vector<uint8_t> pad(ids_pad_size, 0);
            ASSERT_EQ(pad.size(), fwrite(pad.data(), 1, pad.size(), f));
        }
        for (const auto& item : active) {
            const uint64_t id = item.first;
            ASSERT_EQ(1u, fwrite(&id, sizeof(id), 1, f));
        }
        if (!deleted.empty()) {
            ASSERT_EQ(deleted.size(), fwrite(deleted.data(), sizeof(uint64_t), deleted.size(), f));
        }
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

    float first_f32(const DataReader& reader, uint64_t id) {
        const auto* p = reinterpret_cast<const float*>(reader.get(id));
        EXPECT_NE(nullptr, p);
        return p ? p[0] : 0.0f;
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

TEST_F(DataMergerTest, MergeDataFilePreservesCosineValuesSection) {
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
    ASSERT_TRUE(out_reader.has_cosine_inv_norms());
    EXPECT_NEAR(1.0 / (3.0 * std::sqrt(4.0)), static_cast<double>(out_reader.cosine_inv_norm(0)), 1e-6);
    EXPECT_NEAR(1.0 / (5.0 * std::sqrt(4.0)), static_cast<double>(out_reader.cosine_inv_norm(1)), 1e-6);
    EXPECT_NEAR(1.0 / (8.0 * std::sqrt(4.0)), static_cast<double>(out_reader.cosine_inv_norm(2)), 1e-6);
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

TEST_F(DataMergerTest, MergeDeltaFilePreservesCosineValuesSection) {
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
    ASSERT_TRUE(out_reader.has_cosine_inv_norms());
    ASSERT_EQ(3u, out_reader.count());
    EXPECT_NEAR(1.0 / (2.0 * std::sqrt(4.0)), static_cast<double>(out_reader.cosine_inv_norm(0)), 1e-6);
    EXPECT_NEAR(1.0 / (3.0 * std::sqrt(4.0)), static_cast<double>(out_reader.cosine_inv_norm(1)), 1e-6);
    EXPECT_NEAR(1.0 / (8.0 * std::sqrt(4.0)), static_cast<double>(out_reader.cosine_inv_norm(2)), 1e-6);
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
