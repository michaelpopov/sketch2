// Unit tests for dataset lifecycle and persistence behavior.

#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include "core/storage/input_generator.h"
#include "core/storage/data_file.h"
#include "core/storage/data_writer.h"
#include "core/storage/dataset_node.h"
#include "core/storage/data_reader.h"
#include "core/utils/singleton.h"
#include "utils/ini_reader.h"
#include "utest_tmp_dir.h"

using namespace sketch2;
namespace fs = std::filesystem;

class DatasetTest : public ::testing::Test {
protected:
    std::string base_dir_;
    std::string input_path_;
    std::string config_path_;

    void SetUp() override {
        base_dir_   = tmp_dir() + "/sketch2_utest_sc_" + std::to_string(getpid());
        input_path_ = base_dir_ + "/input.txt";
        config_path_ = base_dir_ + "/config.ini";
        fs::create_directories(base_dir_);
    }

    void TearDown() override {
        fs::remove_all(base_dir_);
    }

    // Create a named subdirectory under base_dir_ and return its path.
    std::string make_dir(const std::string& name) {
        std::string path = base_dir_ + "/" + name;
        fs::create_directories(path);
        return path;
    }

    GeneratorConfig cfg(size_t count, size_t min_id, DataType type, size_t dim) {
        return {PatternType::Sequential, count, min_id, type, dim, 1000};
    }

    std::string file_path(const std::string& dir, uint64_t file_id, const std::string& ext) {
        return dir + "/" + std::to_string(file_id) + ext;
    }

    void write_input(const std::string& content) {
        std::ofstream f(input_path_);
        f << content;
    }

    void write_config(const std::string& content) {
        std::ofstream f(config_path_);
        f << content;
    }
};

// --- init error cases ---

TEST_F(DatasetTest, InitFailsOnEmptyDirs) {
    DatasetNode sc;
    EXPECT_NE(0, sc.init({}, 100, DataType::f32, 4).code());
}

TEST_F(DatasetTest, InitFailsOnZeroRangeSize) {
    DatasetNode sc;
    EXPECT_NE(0, sc.init({make_dir("d")}, 0, DataType::f32, 4).code());
}

TEST_F(DatasetTest, InitFailsOnDoubleInit) {
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({make_dir("d")}, 100, DataType::f32, 4).code());
    EXPECT_NE(0, sc.init({make_dir("d2")}, 100, DataType::f32, 4).code());
}

TEST_F(DatasetTest, InitFromIniSectionKeysWorks) {
    auto dir0 = make_dir("d0");
    auto dir1 = make_dir("d1");
    write_config(
        std::string("[dataset]\n") +
        "dirs = " + dir0 + ", " + dir1 + "\n"
        "range_size = 10\n"
        "type = f32\n"
        "dist_func = l2\n"
        "dim = 4\n");

    IniReader ini_cfg;
    ASSERT_EQ(0, ini_cfg.init(config_path_).code());
    EXPECT_EQ(4, ini_cfg.get_int("dataset.dim", -1));

    DatasetNode sc;
    const Ret ret = sc.init(config_path_);
    ASSERT_EQ(0, ret.code()) << ret.message();
    EXPECT_EQ(DistFunc::L2, sc.dist_func());
    generate_input_file(input_path_, cfg(30, 0, DataType::f32, 4));
    ASSERT_EQ(0, sc.store(input_path_).code());
    EXPECT_TRUE(fs::exists(dir0 + "/0.data"));
    EXPECT_TRUE(fs::exists(dir1 + "/1.data"));
    EXPECT_TRUE(fs::exists(dir0 + "/2.data"));
}

TEST_F(DatasetTest, StoreProcessesIndependentRangesThroughConfiguredThreadPool) {
    const auto dir0 = make_dir("d_tp0");
    const auto dir1 = make_dir("d_tp1");

    write_config("[thread_pool]\nsize=2\n");
    ASSERT_TRUE(Singleton::apply_config_file(config_path_));
    ASSERT_NE(nullptr, get_singleton().thread_pool());

    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir0, dir1}, 10, DataType::f32, 4).code());
    ASSERT_EQ(0, generate_input_file(input_path_, cfg(30, 0, DataType::f32, 4)).code());
    ASSERT_EQ(0, sc.store(input_path_).code());

    EXPECT_TRUE(fs::exists(dir0 + "/0.data"));
    EXPECT_TRUE(fs::exists(dir1 + "/1.data"));
    EXPECT_TRUE(fs::exists(dir0 + "/2.data"));

    write_config("[thread_pool]\nsize=1\n");
    EXPECT_FALSE(Singleton::apply_config_file(config_path_));
    EXPECT_NE(nullptr, get_singleton().thread_pool());
}

TEST_F(DatasetTest, InitDefaultsDistFuncToL1WhenMissingFromIni) {
    auto dir = make_dir("d_default_dist");
    write_config(
        std::string("[dataset]\n") +
        "dirs = " + dir + "\n"
        "range_size = 10\n"
        "type = f32\n"
        "dim = 4\n");

    DatasetNode sc;
    ASSERT_EQ(0, sc.init(config_path_).code());
    EXPECT_EQ(DistFunc::L1, sc.dist_func());
}

TEST_F(DatasetTest, InitFromIniAcceptsCosDistanceFunction) {
    auto dir = make_dir("d_cos_dist");
    write_config(
        std::string("[dataset]\n") +
        "dirs = " + dir + "\n"
        "range_size = 10\n"
        "type = f32\n"
        "dist_func = cos\n"
        "dim = 4\n");

    DatasetNode sc;
    ASSERT_EQ(0, sc.init(config_path_).code());
    EXPECT_EQ(DistFunc::COS, sc.dist_func());
}

TEST_F(DatasetTest, InitFromMetadataExposesDistanceFunction) {
    DatasetMetadata metadata;
    metadata.dirs = {make_dir("d_metadata_dist")};
    metadata.range_size = 10;
    metadata.type = DataType::f32;
    metadata.dist_func = DistFunc::L2;
    metadata.dim = 4;

    DatasetNode sc;
    ASSERT_EQ(0, sc.init(metadata).code());
    EXPECT_EQ(DistFunc::L2, sc.dist_func());
}

TEST_F(DatasetTest, InitFromIniUsesAccumulatorSizeForLazyAddVector) {
    auto dir = make_dir("d_acc_ini");
    write_config(
        std::string("[dataset]\n") +
        "dirs = " + dir + "\n"
        "range_size = 10\n"
        "type = f32\n"
        "dim = 4\n"
        "accumulator_size = 23\n");

    DatasetNode sc;
    ASSERT_EQ(0, sc.init(config_path_).code());

    const std::array<float, 4> vec {1.0f, 2.0f, 3.0f, 4.0f};
    const Ret ret = sc.add_vector(1, reinterpret_cast<const uint8_t*>(vec.data()));
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("Accumulator: buffer full", ret.message());
}


TEST_F(DatasetTest, InitFromIniFailsOnMissingType) {
    auto dir = make_dir("d");
    write_config(
        std::string("[dataset]\n") +
        "dirs = " + dir + "\n"
        "range_size = 100\n"
        "dims = 4\n");

    DatasetNode sc;
    EXPECT_NE(0, sc.init(config_path_).code());
}

TEST_F(DatasetTest, InitFromIniFailsOnNegativeDim) {
    auto dir = make_dir("d_neg_dim");
    write_config(
        std::string("[dataset]\n") +
        "dirs = " + dir + "\n"
        "range_size = 100\n"
        "type = f32\n"
        "dim = -1\n");

    DatasetNode sc;
    const Ret ret = sc.init(config_path_);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("Dataset: dataset.dim must be >= 0", ret.message());
}

TEST_F(DatasetTest, InitFromIniFailsOnTooLargeDim) {
    auto dir = make_dir("d_big_dim");
    write_config(
        std::string("[dataset]\n") +
        "dirs = " + dir + "\n"
        "range_size = 100\n"
        "type = f32\n"
        "dim = " + std::to_string(kMaxDimension + 1) + "\n");

    DatasetNode sc;
    const Ret ret = sc.init(config_path_);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("Dataset: dim must be in range [4, 4096].", ret.message());
}

TEST_F(DatasetTest, InitFromIniFailsOnNegativeRangeSize) {
    auto dir = make_dir("d_neg_range");
    write_config(
        std::string("[dataset]\n") +
        "dirs = " + dir + "\n"
        "range_size = -1\n"
        "type = f32\n"
        "dim = 4\n");

    DatasetNode sc;
    const Ret ret = sc.init(config_path_);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("Dataset: dataset.range_size must be >= 0", ret.message());
}

TEST_F(DatasetTest, InitFromIniFailsOnNegativeAccumulatorSize) {
    auto dir = make_dir("d_neg_acc");
    write_config(
        std::string("[dataset]\n") +
        "dirs = " + dir + "\n"
        "range_size = 100\n"
        "type = f32\n"
        "dim = 4\n"
        "accumulator_size = -1\n");

    DatasetNode sc;
    const Ret ret = sc.init(config_path_);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("Dataset: dataset.accumulator_size must be >= 0", ret.message());
}

TEST_F(DatasetTest, InitFromIniFailsOnInvalidDistanceFunction) {
    auto dir = make_dir("d_bad_dist");
    write_config(
        std::string("[dataset]\n") +
        "dirs = " + dir + "\n"
        "range_size = 100\n"
        "type = f32\n"
        "dist_func = cosine\n"
        "dim = 4\n");

    DatasetNode sc;
    const Ret ret = sc.init(config_path_);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("Invalid distance function string.", ret.message());
}

// --- load error cases ---

TEST_F(DatasetTest, StoreFailsWhenNotInitialized) {
    DatasetNode sc;
    EXPECT_NE(0, sc.store(input_path_).code());
}

TEST_F(DatasetTest, StoreAccumulatorFailsWhenNotInitialized) {
    DatasetNode sc;
    EXPECT_NE(0, sc.store_accumulator().code());
}

TEST_F(DatasetTest, AddVectorFailsWhenNotInitialized) {
    DatasetNode sc;
    const std::array<float, 4> vec {1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_NE(0, sc.add_vector(1, reinterpret_cast<const uint8_t*>(vec.data())).code());
}

TEST_F(DatasetTest, DeleteVectorFailsWhenNotInitialized) {
    DatasetNode sc;
    EXPECT_NE(0, sc.delete_vector(1).code());
}


TEST_F(DatasetTest, StoreFailsOnBadInputPath) {
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({make_dir("d")}, 100, DataType::f32, 4).code());
    EXPECT_NE(0, sc.store("/nonexistent/dir/input.txt").code());
}

TEST_F(DatasetTest, StoreFailsOnBadOutputDir) {
    generate_input_file(input_path_, cfg(3, 0, DataType::f32, 4));
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({"/nonexistent/output/dir"}, 100, DataType::f32, 4).code());
    EXPECT_NE(0, sc.store(input_path_).code());
}

// --- load success: file placement ---

TEST_F(DatasetTest, StoreCreatesOutputFile) {
    auto dir = make_dir("d");
    generate_input_file(input_path_, cfg(5, 0, DataType::f32, 4));
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 1000, DataType::f32, 4).code());
    ASSERT_EQ(0, sc.store(input_path_).code());
    EXPECT_TRUE(fs::exists(dir + "/0.data"));
}

TEST_F(DatasetTest, CosineDatasetStoresCosineValuesSection) {
    auto dir = make_dir("d_cos_store");
    write_input(
        "f32,4\n"
        "10 : [ 3.0, 3.0, 3.0, 3.0 ]\n");

    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 1000, DataType::f32, 4, kAccumulatorBufferSize, DistFunc::COS).code());
    ASSERT_EQ(0, sc.store(input_path_).code());

    DataReader dr;
    ASSERT_EQ(0, dr.init(dir + "/0.data").code());
    ASSERT_TRUE(dr.has_cosine_inv_norms());
    EXPECT_NEAR(1.0 / 6.0, static_cast<double>(dr.cosine_inv_norm(0)), 1e-6);
}

TEST_F(DatasetTest, CosineDatasetAccumulatorFlushStoresCosineValuesInDeltaFile) {
    auto dir = make_dir("d_cos_acc_delta");
    generate_input_file(input_path_, cfg(5, 0, DataType::f32, 4));

    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 1000, DataType::f32, 4, kAccumulatorBufferSize, DistFunc::COS).code());
    ASSERT_EQ(0, sc.store(input_path_).code());

    const std::array<float, 4> updated {10.0f, 0.0f, 0.0f, 0.0f};
    ASSERT_EQ(0, sc.add_vector(2, reinterpret_cast<const uint8_t*>(updated.data())).code());
    ASSERT_EQ(0, sc.store_accumulator().code());

    DataReader dr;
    ASSERT_EQ(0, dr.init(dir + "/0.delta").code());
    ASSERT_TRUE(dr.has_cosine_inv_norms());
    ASSERT_EQ(1u, dr.count());
    EXPECT_NEAR(0.1, static_cast<double>(dr.cosine_inv_norm(0)), 1e-6);
}

TEST_F(DatasetTest, CosineDatasetRejectsPersistedFilesWithoutStoredInverseNorms) {
    auto dir = make_dir("d_cos_missing_inv_norms");
    write_input(
        "f32,4\n"
        "10 : [ 100.0, 1.0, 0.0, 0.0 ]\n"
        "20 : [ 1.0, 1.0, 0.0, 0.0 ]\n");

    DataWriter writer;
    ASSERT_EQ(0, writer.init(input_path_, file_path(dir, 0, ".data")).code());
    ASSERT_EQ(0, writer.exec().code());

    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 1000, DataType::f32, 4, kAccumulatorBufferSize, DistFunc::COS).code());

    auto [reader, ret] = sc.get(10);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ(nullptr, reader);
    EXPECT_NE(std::string(ret.message()).find("missing stored inverse norms"), std::string::npos);
}

TEST_F(DatasetTest, StoreAccumulatorSucceedsWhenMissing) {
    auto dir = make_dir("d_acc_none");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4).code());
    EXPECT_EQ(0, sc.store_accumulator().code());
}

TEST_F(DatasetTest, StoreAccumulatorCreatesOutputFilesFromAccumulator) {
    auto dir = make_dir("d_acc_store");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 10, DataType::f32, 4).code());

    const std::array<float, 4> vec0 {1.0f, 1.0f, 1.0f, 1.0f};
    const std::array<float, 4> vec1 {11.0f, 11.0f, 11.0f, 11.0f};
    ASSERT_EQ(0, sc.add_vector(1, reinterpret_cast<const uint8_t*>(vec0.data())).code());
    ASSERT_EQ(0, sc.add_vector(12, reinterpret_cast<const uint8_t*>(vec1.data())).code());

    ASSERT_EQ(0, sc.store_accumulator().code());

    EXPECT_TRUE(fs::exists(file_path(dir, 0, ".data")));
    EXPECT_TRUE(fs::exists(file_path(dir, 1, ".data")));

    DataReader dr0;
    DataReader dr1;
    ASSERT_EQ(0, dr0.init(file_path(dir, 0, ".data")).code());
    ASSERT_EQ(0, dr1.init(file_path(dir, 1, ".data")).code());

    const float* v0 = reinterpret_cast<const float*>(dr0.get(1));
    const float* v1 = reinterpret_cast<const float*>(dr1.get(12));
    ASSERT_NE(nullptr, v0);
    ASSERT_NE(nullptr, v1);
    EXPECT_NEAR(1.0f, v0[0], 1e-5f);
    EXPECT_NEAR(11.0f, v1[0], 1e-5f);
}

TEST_F(DatasetTest, StoreAccumulatorLeavesNoTempWhenCreatingData) {
    auto dir = make_dir("d_acc_staging_data");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 10, DataType::f32, 4).code());

    const std::array<float, 4> vec {2.0f, 2.0f, 2.0f, 2.0f};
    ASSERT_EQ(0, sc.add_vector(0, reinterpret_cast<const uint8_t*>(vec.data())).code());
    ASSERT_EQ(0, sc.store_accumulator().code());

    EXPECT_TRUE(fs::exists(file_path(dir, 0, ".data")));
    EXPECT_FALSE(fs::exists(file_path(dir, 0, ".temp")));
}

TEST_F(DatasetTest, StoreAccumulatorLeavesNoTempWhenCreatingDelta) {
    auto dir = make_dir("d_acc_staging_delta");
    DatasetNode sc;
    DatasetMetadata metadata;
    metadata.dirs = {dir};
    metadata.range_size = 100;
    metadata.type = DataType::f32;
    metadata.dim = 4;
    metadata.data_merge_ratio = 2;
    ASSERT_EQ(0, sc.init(metadata).code());

    const std::array<float, 4> base_vec {3.0f, 3.0f, 3.0f, 3.0f};
    ASSERT_EQ(0, sc.add_vector(0, reinterpret_cast<const uint8_t*>(base_vec.data())).code());
    ASSERT_EQ(0, sc.add_vector(1, reinterpret_cast<const uint8_t*>(base_vec.data())).code());
    ASSERT_EQ(0, sc.add_vector(2, reinterpret_cast<const uint8_t*>(base_vec.data())).code());
    ASSERT_EQ(0, sc.add_vector(3, reinterpret_cast<const uint8_t*>(base_vec.data())).code());
    ASSERT_EQ(0, sc.store_accumulator().code());

    const std::array<float, 4> delta_vec {5.0f, 5.0f, 5.0f, 5.0f};
    ASSERT_EQ(0, sc.add_vector(4, reinterpret_cast<const uint8_t*>(delta_vec.data())).code());
    ASSERT_EQ(0, sc.store_accumulator().code());

    EXPECT_TRUE(fs::exists(file_path(dir, 0, ".delta")));
    EXPECT_FALSE(fs::exists(file_path(dir, 0, ".temp")));
}

TEST_F(DatasetTest, StoreAccumulatorIgnoresDeleteForMissingRange) {
    auto dir = make_dir("d_acc_missing_delete");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 10, DataType::f32, 4).code());

    ASSERT_EQ(0, sc.delete_vector(25).code());
    ASSERT_EQ(0, sc.store_accumulator().code());

    EXPECT_FALSE(fs::exists(file_path(dir, 2, ".data")));
    EXPECT_FALSE(fs::exists(file_path(dir, 2, ".delta")));
}

TEST_F(DatasetTest, StoreAccumulatorClearsAccumulatorAfterSuccess) {
    auto dir = make_dir("d_acc_clear");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::i16, 4, 40).code());

    const std::array<int16_t, 4> vec0 {1, 2, 3, 4};
    const std::array<int16_t, 4> vec1 {5, 6, 7, 8};
    ASSERT_EQ(0, sc.add_vector(1, reinterpret_cast<const uint8_t*>(vec0.data())).code());
    ASSERT_EQ(0, sc.store_accumulator().code());

    EXPECT_EQ(0, sc.add_vector(2, reinterpret_cast<const uint8_t*>(vec1.data())).code());
    ASSERT_EQ(0, sc.store_accumulator().code());

    DataReader dr;
    ASSERT_EQ(0, dr.init(file_path(dir, 0, ".data")).code());
    EXPECT_EQ(2u, dr.count());
    ASSERT_NE(nullptr, dr.get(1));
    ASSERT_NE(nullptr, dr.get(2));
}

TEST_F(DatasetTest, StoreAccumulatorReplaysWalAfterReopen) {
    auto dir = make_dir("d_acc_replay");
    const std::array<float, 4> vec {3.0f, 4.0f, 5.0f, 6.0f};

    {
        DatasetNode sc;
        ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4).code());
        ASSERT_EQ(0, sc.add_vector(17, reinterpret_cast<const uint8_t*>(vec.data())).code());
    }

    const std::string wal_path = dir + "/sketch2.accumulator.wal";
    ASSERT_TRUE(fs::exists(wal_path));

    DatasetNode reopened;
    ASSERT_EQ(0, reopened.init({dir}, 100, DataType::f32, 4).code());
    ASSERT_EQ(0, reopened.store_accumulator().code());

    DataReader dr;
    ASSERT_EQ(0, dr.init(file_path(dir, 0, ".data")).code());
    const float* stored = reinterpret_cast<const float*>(dr.get(17));
    ASSERT_NE(nullptr, stored);
    EXPECT_NEAR(3.0f, stored[0], 1e-5f);
    EXPECT_EQ(static_cast<uintmax_t>(sizeof(WalFileHeader)), fs::file_size(wal_path));
}

TEST_F(DatasetTest, DestructorFlushesPendingAccumulatorUpdates) {
    auto dir = make_dir("d_acc_destructor_flush");
    const std::array<float, 4> vec {3.0f, 4.0f, 5.0f, 6.0f};

    {
        DatasetNode sc;
        ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4).code());
        ASSERT_EQ(0, sc.add_vector(17, reinterpret_cast<const uint8_t*>(vec.data())).code());
    }

    DataReader dr;
    ASSERT_EQ(0, dr.init(file_path(dir, 0, ".data")).code());
    const float* stored = reinterpret_cast<const float*>(dr.get(17));
    ASSERT_NE(nullptr, stored);
    EXPECT_NEAR(3.0f, stored[0], 1e-5f);
}

TEST_F(DatasetTest, AddVectorCreatesAccumulatorLazily) {
    auto dir = make_dir("d_acc_add");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4).code());

    const std::array<float, 4> vec {1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_EQ(0, sc.add_vector(1, reinterpret_cast<const uint8_t*>(vec.data())).code());
}

TEST_F(DatasetTest, AddVectorFlushesAccumulatorWhenFull) {
    auto dir = make_dir("d_acc_add_flush");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4, 40).code());

    const std::array<float, 4> vec0 {1.0f, 2.0f, 3.0f, 4.0f};
    const std::array<float, 4> vec1 {5.0f, 6.0f, 7.0f, 8.0f};
    ASSERT_EQ(0, sc.add_vector(1, reinterpret_cast<const uint8_t*>(vec0.data())).code());
    ASSERT_EQ(0, sc.add_vector(2, reinterpret_cast<const uint8_t*>(vec1.data())).code());
    ASSERT_EQ(0, sc.store_accumulator().code());

    DataReader dr;
    ASSERT_EQ(0, dr.init(file_path(dir, 0, ".data")).code());
    EXPECT_EQ(2u, dr.count());
    ASSERT_NE(nullptr, dr.get(1));
    ASSERT_NE(nullptr, dr.get(2));
}

TEST_F(DatasetTest, AddVectorNullDoesNotFlushPendingAccumulatorData) {
    auto dir = make_dir("d_acc_add_null");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4, 40).code());

    const std::array<float, 4> vec {1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT_EQ(0, sc.add_vector(1, reinterpret_cast<const uint8_t*>(vec.data())).code());

    const Ret ret = sc.add_vector(2, nullptr);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("Dataset: invalid data argument", ret.message());

    EXPECT_FALSE(fs::exists(file_path(dir, 0, ".data")));
    ASSERT_EQ(0, sc.store_accumulator().code());

    DataReader dr;
    ASSERT_EQ(0, dr.init(file_path(dir, 0, ".data")).code());
    EXPECT_EQ(1u, dr.count());
    ASSERT_NE(nullptr, dr.get(1));
    EXPECT_EQ(nullptr, dr.get(2));
}

TEST_F(DatasetTest, AddVectorUsesConfiguredAccumulatorSize) {
    auto dir = make_dir("d_acc_add_small");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4, 23).code());

    const std::array<float, 4> vec {1.0f, 2.0f, 3.0f, 4.0f};
    const Ret ret = sc.add_vector(1, reinterpret_cast<const uint8_t*>(vec.data()));
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("Accumulator: buffer full", ret.message());
}

TEST_F(DatasetTest, DeleteVectorCreatesAccumulatorLazily) {
    auto dir = make_dir("d_acc_del");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4).code());

    EXPECT_EQ(0, sc.delete_vector(1).code());
}

TEST_F(DatasetTest, DeleteVectorFlushesAccumulatorWhenFull) {
    auto dir = make_dir("d_acc_del_flush");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4, 40).code());

    const std::array<float, 4> vec {1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT_EQ(0, sc.add_vector(1, reinterpret_cast<const uint8_t*>(vec.data())).code());
    ASSERT_EQ(0, sc.delete_vector(2).code());
    ASSERT_EQ(0, sc.store_accumulator().code());

    DataReader dr;
    ASSERT_EQ(0, dr.init(file_path(dir, 0, ".data")).code());
    EXPECT_EQ(1u, dr.count());
    ASSERT_NE(nullptr, dr.get(1));

    auto [reader, ret] = sc.get(2);
    ASSERT_EQ(0, ret.code()) << ret.message();
    ASSERT_NE(nullptr, reader);
    EXPECT_EQ(nullptr, reader->get(2));
}

TEST_F(DatasetTest, DeleteVectorUsesConfiguredAccumulatorSize) {
    auto dir = make_dir("d_acc_del_small");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4, 7).code());

    const Ret ret = sc.delete_vector(1);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("Accumulator: buffer full", ret.message());
}

TEST_F(DatasetTest, StoreFileIdFromMinId) {
    // ids [2005..2009], range_size=1000 -> file_id=2 -> "2.data"
    auto dir = make_dir("d");
    generate_input_file(input_path_, cfg(5, 2005, DataType::f32, 4));
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 1000, DataType::f32, 4).code());
    ASSERT_EQ(0, sc.store(input_path_).code());
    EXPECT_TRUE(fs::exists(dir + "/2.data"));
    EXPECT_FALSE(fs::exists(dir + "/0.data"));
}

TEST_F(DatasetTest, StoreMultipleRangesCreateMultipleFiles) {
    // ids [5..19], range_size=10 -> file 0 (ids 5-9), file 1 (ids 10-19)
    auto dir = make_dir("d");
    generate_input_file(input_path_, cfg(15, 5, DataType::f32, 4));
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 10, DataType::f32, 4).code());
    ASSERT_EQ(0, sc.store(input_path_).code());
    EXPECT_TRUE(fs::exists(dir + "/0.data"));
    EXPECT_TRUE(fs::exists(dir + "/1.data"));
}

TEST_F(DatasetTest, StoreMultipleDirsRoutesByFileId) {
    // ids [0..29], range_size=10 -> file_id 0,1,2
    // 2 dirs: file_id%2 -> dir0 gets 0.data, 2.data; dir1 gets 1.data
    auto dir0 = make_dir("d0");
    auto dir1 = make_dir("d1");
    generate_input_file(input_path_, cfg(30, 0, DataType::f32, 4));
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir0, dir1}, 10, DataType::f32, 4).code());
    ASSERT_EQ(0, sc.store(input_path_).code());
    EXPECT_TRUE(fs::exists(dir0 + "/0.data"));
    EXPECT_TRUE(fs::exists(dir1 + "/1.data"));
    EXPECT_TRUE(fs::exists(dir0 + "/2.data"));
}

// --- load success: output file content ---

TEST_F(DatasetTest, StoreFileCount) {
    auto dir = make_dir("d");
    generate_input_file(input_path_, cfg(7, 0, DataType::f32, 4));
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 1000, DataType::f32, 4).code());
    ASSERT_EQ(0, sc.store(input_path_).code());
    DataReader dr;
    ASSERT_EQ(0, dr.init(dir + "/0.data").code());
    EXPECT_EQ(7u, dr.count());
}

TEST_F(DatasetTest, StoreFileType) {
    auto dir = make_dir("d");
    generate_input_file(input_path_, cfg(3, 0, DataType::f32, 4));
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 1000, DataType::f32, 4).code());
    ASSERT_EQ(0, sc.store(input_path_).code());
    DataReader dr;
    ASSERT_EQ(0, dr.init(dir + "/0.data").code());
    EXPECT_EQ(DataType::f32, dr.type());
}

TEST_F(DatasetTest, StoreFileDim) {
    auto dir = make_dir("d");
    generate_input_file(input_path_, cfg(3, 0, DataType::f32, 64));
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 1000, DataType::f32, 64).code());
    const Ret ret = sc.store(input_path_);
    ASSERT_EQ(0, ret.code()) << ret.message();
    DataReader dr;
    ASSERT_EQ(0, dr.init(dir + "/0.data").code());
    EXPECT_EQ(64u, dr.dim());
}

TEST_F(DatasetTest, StoreFileDataValues) {
    // generator writes id+0.1 for each dimension
    auto dir = make_dir("d");
    generate_input_file(input_path_, cfg(5, 10, DataType::f32, 4));
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 1000, DataType::f32, 4).code());
    ASSERT_EQ(0, sc.store(input_path_).code());
    DataReader dr;
    ASSERT_EQ(0, dr.init(dir + "/0.data").code());
    for (uint64_t id = 10; id < 15; ++id) {
        const float* v = reinterpret_cast<const float*>(dr.get(id));
        ASSERT_NE(nullptr, v) << "id " << id << " not found";
        float expected = static_cast<float>(id) + 0.1f;
        for (size_t d = 0; d < 4; ++d) {
            EXPECT_NEAR(expected, v[d], 1e-4f) << "id=" << id << " dim=" << d;
        }
    }
}

TEST_F(DatasetTest, StoreMultipleRangesCorrectCounts) {
    // ids [5..19], range_size=10
    // file 0: ids 5-9  -> 5 vectors
    // file 1: ids 10-19 -> 10 vectors
    auto dir = make_dir("d");
    generate_input_file(input_path_, cfg(15, 5, DataType::f32, 4));
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 10, DataType::f32, 4).code());
    ASSERT_EQ(0, sc.store(input_path_).code());

    DataReader dr0, dr1;
    ASSERT_EQ(0, dr0.init(dir + "/0.data").code());
    ASSERT_EQ(0, dr1.init(dir + "/1.data").code());
    EXPECT_EQ(5u,  dr0.count());
    EXPECT_EQ(10u, dr1.count());
}

TEST_F(DatasetTest, StoreSkipsMissingMiddleRanges) {
    auto dir = make_dir("d");
    {
        std::ofstream f(input_path_);
        f << "f32,4\n";
        f << "0 : [ 0.10, 0.10, 0.10, 0.10 ]\n";
        f << "20 : [ 20.10, 20.10, 20.10, 20.10 ]\n";
    }

    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 10, DataType::f32, 4).code());
    ASSERT_EQ(0, sc.store(input_path_).code());

    EXPECT_TRUE(fs::exists(dir + "/0.data"));
    EXPECT_FALSE(fs::exists(dir + "/1.data"));
    EXPECT_TRUE(fs::exists(dir + "/2.data"));
}

// --- merge and delta lifecycle ---

TEST_F(DatasetTest, StoreHeaderOnlyInputCreatesNoFiles) {
    auto dir = make_dir("d");
    write_input("f32,4\n");

    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4).code());
    ASSERT_EQ(0, sc.store(input_path_).code());

    EXPECT_FALSE(fs::exists(file_path(dir, 0, ".data")));
    EXPECT_FALSE(fs::exists(file_path(dir, 0, ".delta")));
}

TEST_F(DatasetTest, SecondSmallLoadCreatesDeltaFile) {
    auto dir = make_dir("d");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4).code());

    generate_input_file(input_path_, cfg(20, 0, DataType::f32, 4));
    ASSERT_EQ(0, sc.store(input_path_).code());
    ASSERT_TRUE(fs::exists(file_path(dir, 0, ".data")));
    ASSERT_FALSE(fs::exists(file_path(dir, 0, ".delta")));

    generate_input_file(input_path_, cfg(1, 50, DataType::f32, 4));
    ASSERT_EQ(0, sc.store(input_path_).code());

    EXPECT_TRUE(fs::exists(file_path(dir, 0, ".data")));
    EXPECT_TRUE(fs::exists(file_path(dir, 0, ".delta")));

    DataReader data_reader, delta_reader;
    ASSERT_EQ(0, data_reader.init(file_path(dir, 0, ".data")).code());
    ASSERT_EQ(0, delta_reader.init(file_path(dir, 0, ".delta")).code());
    EXPECT_EQ(20u, data_reader.count());
    EXPECT_EQ(1u, delta_reader.count());
    EXPECT_NE(nullptr, delta_reader.get(50));
}

TEST_F(DatasetTest, SecondLargeLoadMergesIntoDataWithoutDelta) {
    auto dir = make_dir("d");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4).code());

    generate_input_file(input_path_, cfg(5, 0, DataType::f32, 4));
    ASSERT_EQ(0, sc.store(input_path_).code());

    generate_input_file(input_path_, cfg(3, 5, DataType::f32, 4));
    ASSERT_EQ(0, sc.store(input_path_).code());

    EXPECT_TRUE(fs::exists(file_path(dir, 0, ".data")));
    EXPECT_FALSE(fs::exists(file_path(dir, 0, ".delta")));

    DataReader merged;
    ASSERT_EQ(0, merged.init(file_path(dir, 0, ".data")).code());
    EXPECT_EQ(8u, merged.count());
    EXPECT_NE(nullptr, merged.get(0));
    EXPECT_NE(nullptr, merged.get(7));
}

TEST_F(DatasetTest, ExistingDeltaGetsMergedWithNewSmallUpdateAndStaysDelta) {
    auto dir = make_dir("d");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4).code());

    generate_input_file(input_path_, cfg(20, 0, DataType::f32, 4));
    ASSERT_EQ(0, sc.store(input_path_).code());

    generate_input_file(input_path_, cfg(1, 30, DataType::f32, 4));
    ASSERT_EQ(0, sc.store(input_path_).code());
    ASSERT_TRUE(fs::exists(file_path(dir, 0, ".delta")));

    generate_input_file(input_path_, cfg(1, 40, DataType::f32, 4));
    ASSERT_EQ(0, sc.store(input_path_).code());

    EXPECT_TRUE(fs::exists(file_path(dir, 0, ".data")));
    EXPECT_TRUE(fs::exists(file_path(dir, 0, ".delta")));

    DataReader data_reader, delta_reader;
    ASSERT_EQ(0, data_reader.init(file_path(dir, 0, ".data")).code());
    ASSERT_EQ(0, delta_reader.init(file_path(dir, 0, ".delta")).code());
    EXPECT_EQ(20u, data_reader.count());
    EXPECT_EQ(2u, delta_reader.count());
    EXPECT_NE(nullptr, delta_reader.get(30));
    EXPECT_NE(nullptr, delta_reader.get(40));
}

TEST_F(DatasetTest, LargeDeltaEventuallyMergesIntoDataAndRemovesDeltaFile) {
    auto dir = make_dir("d");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4).code());

    generate_input_file(input_path_, cfg(10, 0, DataType::f32, 4));
    ASSERT_EQ(0, sc.store(input_path_).code());

    generate_input_file(input_path_, cfg(1, 80, DataType::f32, 4));
    ASSERT_EQ(0, sc.store(input_path_).code());
    ASSERT_TRUE(fs::exists(file_path(dir, 0, ".delta")));

    generate_input_file(input_path_, cfg(5, 50, DataType::f32, 4));
    ASSERT_EQ(0, sc.store(input_path_).code());

    EXPECT_TRUE(fs::exists(file_path(dir, 0, ".data")));
    EXPECT_FALSE(fs::exists(file_path(dir, 0, ".delta")));

    DataReader data_reader;
    ASSERT_EQ(0, data_reader.init(file_path(dir, 0, ".data")).code());
    EXPECT_EQ(16u, data_reader.count());
    EXPECT_NE(nullptr, data_reader.get(80));
    EXPECT_NE(nullptr, data_reader.get(52));
}

TEST_F(DatasetTest, DeleteFromDeltaIsAppliedAfterDataDeltaMerge) {
    auto dir = make_dir("d");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4).code());

    generate_input_file(input_path_, cfg(10, 0, DataType::f32, 4));
    ASSERT_EQ(0, sc.store(input_path_).code());

    write_input(
        "f32,4\n"
        "3 : []\n");
    ASSERT_EQ(0, sc.store(input_path_).code());
    ASSERT_TRUE(fs::exists(file_path(dir, 0, ".delta")));

    generate_input_file(input_path_, cfg(5, 50, DataType::f32, 4));
    ASSERT_EQ(0, sc.store(input_path_).code());

    EXPECT_FALSE(fs::exists(file_path(dir, 0, ".delta")));
    DataReader data_reader;
    ASSERT_EQ(0, data_reader.init(file_path(dir, 0, ".data")).code());
    EXPECT_EQ(14u, data_reader.count()); // 10 original - 1 deleted + 5 inserted
    EXPECT_EQ(nullptr, data_reader.get(3));
    EXPECT_NE(nullptr, data_reader.get(50));
}

TEST_F(DatasetTest, DeltaCreatedOnlyForTouchedRange) {
    auto dir = make_dir("d");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 10, DataType::f32, 4).code());

    generate_input_file(input_path_, cfg(20, 0, DataType::f32, 4)); // files 0 and 1
    ASSERT_EQ(0, sc.store(input_path_).code());
    ASSERT_TRUE(fs::exists(file_path(dir, 0, ".data")));
    ASSERT_TRUE(fs::exists(file_path(dir, 1, ".data")));

    generate_input_file(input_path_, cfg(1, 2, DataType::f32, 4)); // touches only file 0
    ASSERT_EQ(0, sc.store(input_path_).code());

    EXPECT_TRUE(fs::exists(file_path(dir, 0, ".delta")));
    EXPECT_FALSE(fs::exists(file_path(dir, 1, ".delta")));
}

TEST_F(DatasetTest, MergeCombinesExistingDeltaIntoDataFile) {
    auto dir = make_dir("d_merge");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4).code());

    generate_input_file(input_path_, cfg(20, 0, DataType::f32, 4));
    ASSERT_EQ(0, sc.store(input_path_).code());

    write_input("f32,4\n5 : [ 99.0, 99.0, 99.0, 99.0 ]\n");
    ASSERT_EQ(0, sc.store(input_path_).code());
    ASSERT_TRUE(fs::exists(file_path(dir, 0, ".delta")));

    ASSERT_EQ(0, sc.merge().code());

    EXPECT_TRUE(fs::exists(file_path(dir, 0, ".data")));
    EXPECT_FALSE(fs::exists(file_path(dir, 0, ".delta")));

    DataReader data_reader;
    ASSERT_EQ(0, data_reader.init(file_path(dir, 0, ".data")).code());
    const float* v = reinterpret_cast<const float*>(data_reader.get(5));
    ASSERT_NE(nullptr, v);
    EXPECT_NEAR(99.0f, v[0], 1e-5f);
}

TEST_F(DatasetTest, MergeCombinesAccumulatorDeltaIntoDataFile) {
    auto dir = make_dir("d_merge_acc");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4).code());

    generate_input_file(input_path_, cfg(20, 0, DataType::f32, 4));
    ASSERT_EQ(0, sc.store(input_path_).code());

    const std::array<float, 4> vec {77.0f, 77.0f, 77.0f, 77.0f};
    ASSERT_EQ(0, sc.add_vector(5, reinterpret_cast<const uint8_t*>(vec.data())).code());
    ASSERT_EQ(0, sc.store_accumulator().code());
    ASSERT_TRUE(fs::exists(file_path(dir, 0, ".delta")));

    ASSERT_EQ(0, sc.merge().code());

    EXPECT_TRUE(fs::exists(file_path(dir, 0, ".data")));
    EXPECT_FALSE(fs::exists(file_path(dir, 0, ".delta")));

    DataReader data_reader;
    ASSERT_EQ(0, data_reader.init(file_path(dir, 0, ".data")).code());
    const float* v = reinterpret_cast<const float*>(data_reader.get(5));
    ASSERT_NE(nullptr, v);
    EXPECT_NEAR(77.0f, v[0], 1e-5f);
}

TEST_F(DatasetTest, MergeProcessesAllRangesWithDeltaFiles) {
    auto dir = make_dir("d_merge_all");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 10, DataType::f32, 4).code());

    generate_input_file(input_path_, cfg(20, 0, DataType::f32, 4));
    ASSERT_EQ(0, sc.store(input_path_).code());

    write_input(
        "f32,4\n"
        "2 : [ 20.0, 20.0, 20.0, 20.0 ]\n"
        "15 : [ 150.0, 150.0, 150.0, 150.0 ]\n");
    ASSERT_EQ(0, sc.store(input_path_).code());
    ASSERT_TRUE(fs::exists(file_path(dir, 0, ".delta")));
    ASSERT_TRUE(fs::exists(file_path(dir, 1, ".delta")));

    ASSERT_EQ(0, sc.merge().code());

    EXPECT_FALSE(fs::exists(file_path(dir, 0, ".delta")));
    EXPECT_FALSE(fs::exists(file_path(dir, 1, ".delta")));

    DataReader data0;
    DataReader data1;
    ASSERT_EQ(0, data0.init(file_path(dir, 0, ".data")).code());
    ASSERT_EQ(0, data1.init(file_path(dir, 1, ".data")).code());

    const float* v0 = reinterpret_cast<const float*>(data0.get(2));
    const float* v1 = reinterpret_cast<const float*>(data1.get(15));
    ASSERT_NE(nullptr, v0);
    ASSERT_NE(nullptr, v1);
    EXPECT_NEAR(20.0f, v0[0], 1e-5f);
    EXPECT_NEAR(150.0f, v1[0], 1e-5f);
}

// --- DatasetRangeReader::next() ---

TEST_F(DatasetTest, DatasetRangeReaderNextReturnsNullWhenExhausted) {
    auto dir = make_dir("d");
    generate_input_file(input_path_, cfg(5, 0, DataType::f32, 4));
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 1000, DataType::f32, 4).code());
    ASSERT_EQ(0, sc.store(input_path_).code());

    auto drs = sc.reader();

    // Consume the one data file.
    auto [r1, ret1] = drs->next();
    EXPECT_EQ(0, ret1.code());
    EXPECT_NE(nullptr, r1);

    // Past the end: null reader with success code.
    auto [r2, ret2] = drs->next();
    EXPECT_EQ(0, ret2.code());
    EXPECT_EQ(nullptr, r2);

    // Repeated calls past the end also succeed.
    auto [r3, ret3] = drs->next();
    EXPECT_EQ(0, ret3.code());
    EXPECT_EQ(nullptr, r3);
}

TEST_F(DatasetTest, DatasetRangeReaderNextOnEmptyDatasetReturnsNull) {
    auto dir = make_dir("empty");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 1000, DataType::f32, 4).code());
    // No store() call — no files in dir.
    auto drs = sc.reader();
    auto [reader, ret] = drs->next();
    EXPECT_EQ(0, ret.code());
    EXPECT_EQ(nullptr, reader);
}

TEST_F(DatasetTest, DatasetGetOnEmptyDatasetReturnsNull) {
    auto dir = make_dir("empty_get");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 1000, DataType::f32, 4).code());

    auto [reader, ret] = sc.get(42);
    EXPECT_EQ(0, ret.code());
    EXPECT_EQ(nullptr, reader);
}

TEST_F(DatasetTest, DatasetRangeReaderNextIteratesAllFiles) {
    auto dir = make_dir("d");
    // ids 0..29 with range_size=10 → 3 data files: 0.data, 1.data, 2.data.
    generate_input_file(input_path_, cfg(30, 0, DataType::f32, 4));
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 10, DataType::f32, 4).code());
    ASSERT_EQ(0, sc.store(input_path_).code());

    auto drs = sc.reader();
    size_t file_count = 0;
    while (true) {
        auto [reader, ret] = drs->next();
        ASSERT_EQ(0, ret.code());
        if (!reader) break;
        ++file_count;
    }
    EXPECT_EQ(3u, file_count);
}

TEST_F(DatasetTest, DatasetRangeReaderNextReturnsDeltaFileAlongsideData) {
    auto dir = make_dir("d");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4).code());

    generate_input_file(input_path_, cfg(20, 0, DataType::f32, 4));
    ASSERT_EQ(0, sc.store(input_path_).code());

    // Small update: modify id=5 (already in the data file) — triggers delta creation.
    // Inserting a brand-new id would only land in the delta and be invisible via
    // DataReader::get(), which searches the data file's id array.
    write_input("f32,4\n5 : [ 99.0, 99.0, 99.0, 99.0 ]\n");
    ASSERT_EQ(0, sc.store(input_path_).code());
    ASSERT_TRUE(fs::exists(file_path(dir, 0, ".delta")));

    // The merged reader must return the updated value for id=5.
    auto drs = sc.reader();
    auto [reader, ret] = drs->next();
    ASSERT_EQ(0, ret.code());
    ASSERT_NE(nullptr, reader);
    const float* v = reinterpret_cast<const float*>(reader->get(5));
    ASSERT_NE(nullptr, v);
    EXPECT_NEAR(99.0f, v[0], 1e-5f);
}

TEST_F(DatasetTest, DatasetRangeReaderGetReturnsReaderForContainingRange) {
    auto dir = make_dir("d_get");
    generate_input_file(input_path_, cfg(30, 0, DataType::f32, 4)); // files 0,1,2 for range_size=10
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 10, DataType::f32, 4).code());
    ASSERT_EQ(0, sc.store(input_path_).code());

    auto drs = sc.reader();
    auto [reader, ret] = drs->get(17);
    ASSERT_EQ(0, ret.code()) << ret.message();
    ASSERT_NE(nullptr, reader);
    ASSERT_NE(nullptr, reader->get(17));
    ASSERT_EQ(nullptr, reader->get(29));
}

TEST_F(DatasetTest, DatasetRangeReaderGetReturnsNullWhenRangeHasNoFile) {
    auto dir = make_dir("d_get_sparse");
    write_input(
        "f32,4\n"
        "0 : [ 0.1, 0.1, 0.1, 0.1 ]\n"
        "20 : [ 20.1, 20.1, 20.1, 20.1 ]\n");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 10, DataType::f32, 4).code());
    ASSERT_EQ(0, sc.store(input_path_).code());

    auto drs = sc.reader();
    auto [reader, ret] = drs->get(15); // file_id=1 is missing
    ASSERT_EQ(0, ret.code()) << ret.message();
    EXPECT_EQ(nullptr, reader);
}

TEST_F(DatasetTest, DatasetGetReturnsReaderWithDeltaApplied) {
    auto dir = make_dir("d_ds_get");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4).code());

    generate_input_file(input_path_, cfg(20, 0, DataType::f32, 4));
    ASSERT_EQ(0, sc.store(input_path_).code());

    write_input("f32,4\n5 : [ 99.0, 99.0, 99.0, 99.0 ]\n");
    ASSERT_EQ(0, sc.store(input_path_).code());
    ASSERT_TRUE(fs::exists(file_path(dir, 0, ".delta")));

    auto [reader, ret] = sc.get(5);
    ASSERT_EQ(0, ret.code()) << ret.message();
    ASSERT_NE(nullptr, reader);
    const float* v = reinterpret_cast<const float*>(reader->get(5));
    ASSERT_NE(nullptr, v);
    EXPECT_NEAR(99.0f, v[0], 1e-5f);

    auto [missing_reader, missing_ret] = sc.get(5000);
    ASSERT_EQ(0, missing_ret.code()) << missing_ret.message();
    EXPECT_EQ(nullptr, missing_reader);
}

TEST_F(DatasetTest, DatasetGetVectorLoadsPendingWalAfterReopen) {
    auto dir = make_dir("d_ds_get_vector_reopen");
    const std::array<float, 4> updated {500.0f, 500.0f, 500.0f, 500.0f};

    {
        DatasetNode owner;
        ASSERT_EQ(0, owner.init({dir}, 100, DataType::f32, 4).code());
        generate_input_file(input_path_, cfg(5, 0, DataType::f32, 4));
        ASSERT_EQ(0, owner.store(input_path_).code());
        ASSERT_EQ(0, owner.add_vector(2, reinterpret_cast<const uint8_t*>(updated.data())).code());
        ASSERT_EQ(0, owner.delete_vector(3).code());
    }

    DatasetNode reopened;
    ASSERT_EQ(0, reopened.init({dir}, 100, DataType::f32, 4).code());

    auto [vec_data, ret] = reopened.get_vector(2);
    ASSERT_EQ(0, ret.code()) << ret.message();
    ASSERT_NE(nullptr, vec_data);
    const float* values = reinterpret_cast<const float*>(vec_data);
    EXPECT_FLOAT_EQ(500.0f, values[0]);

    auto [deleted_data, deleted_ret] = reopened.get_vector(3);
    ASSERT_EQ(0, deleted_ret.code()) << deleted_ret.message();
    EXPECT_EQ(nullptr, deleted_data);
}

TEST_F(DatasetTest, DatasetGetCachesOpenedReaders) {
    auto dir = make_dir("d_ds_cache");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4).code());

    generate_input_file(input_path_, cfg(20, 0, DataType::f32, 4));
    ASSERT_EQ(0, sc.store(input_path_).code());

    auto [reader0, ret0] = sc.get(5);
    ASSERT_EQ(0, ret0.code()) << ret0.message();
    ASSERT_NE(nullptr, reader0);

    auto [reader1, ret1] = sc.get(7);
    ASSERT_EQ(0, ret1.code()) << ret1.message();
    ASSERT_NE(nullptr, reader1);
    EXPECT_EQ(reader0.get(), reader1.get());
}

TEST_F(DatasetTest, DatasetRangeReaderSharesDatasetRangeReaderCache) {
    auto dir = make_dir("d_ds_reader_cache");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 10, DataType::f32, 4).code());

    generate_input_file(input_path_, cfg(20, 0, DataType::f32, 4));
    ASSERT_EQ(0, sc.store(input_path_).code());

    auto drs = sc.reader();
    auto [reader0, ret0] = drs->get(5);
    ASSERT_EQ(0, ret0.code()) << ret0.message();
    ASSERT_NE(nullptr, reader0);

    auto [reader1, ret1] = sc.get(7);
    ASSERT_EQ(0, ret1.code()) << ret1.message();
    ASSERT_NE(nullptr, reader1);
    EXPECT_EQ(reader0.get(), reader1.get());
}

TEST_F(DatasetTest, DatasetInvalidatesCachesAfterStore) {
    auto dir = make_dir("d_ds_invalidate_store");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4).code());

    generate_input_file(input_path_, cfg(20, 0, DataType::f32, 4));
    ASSERT_EQ(0, sc.store(input_path_).code());

    auto [reader0, ret0] = sc.get(5);
    ASSERT_EQ(0, ret0.code()) << ret0.message();
    ASSERT_NE(nullptr, reader0);
    const void* old_ptr = reader0.get();

    write_input("f32,4\n5 : [ 99.0, 99.0, 99.0, 99.0 ]\n");
    ASSERT_EQ(0, sc.store(input_path_).code());

    auto [reader1, ret1] = sc.get(5);
    ASSERT_EQ(0, ret1.code()) << ret1.message();
    ASSERT_NE(nullptr, reader1);
    EXPECT_NE(old_ptr, reader1.get());

    const float* v = reinterpret_cast<const float*>(reader1->get(5));
    ASSERT_NE(nullptr, v);
    EXPECT_NEAR(99.0f, v[0], 1e-5f);
}

TEST_F(DatasetTest, DatasetInvalidatesItemCacheAfterStoreCreatesNewRange) {
    auto dir = make_dir("d_ds_invalidate_items");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 10, DataType::f32, 4).code());

    generate_input_file(input_path_, cfg(5, 0, DataType::f32, 4));
    ASSERT_EQ(0, sc.store(input_path_).code());

    auto [missing_reader, missing_ret] = sc.get(15);
    ASSERT_EQ(0, missing_ret.code()) << missing_ret.message();
    EXPECT_EQ(nullptr, missing_reader);

    write_input("f32,4\n15 : [ 15.1, 15.1, 15.1, 15.1 ]\n");
    ASSERT_EQ(0, sc.store(input_path_).code());

    auto [reader, ret] = sc.get(15);
    ASSERT_EQ(0, ret.code()) << ret.message();
    ASSERT_NE(nullptr, reader);
    ASSERT_NE(nullptr, reader->get(15));
}

TEST_F(DatasetTest, GuestModeAllowsQueries) {
    auto dir = make_dir("d_guest_query");

    DatasetNode owner;
    ASSERT_EQ(0, owner.init({dir}, 100, DataType::f32, 4).code());
    generate_input_file(input_path_, cfg(5, 0, DataType::f32, 4));
    ASSERT_EQ(0, owner.store(input_path_).code());

    DatasetReader guest;
    ASSERT_EQ(0, guest.init({dir}, 100, DataType::f32, 4).code());


    auto [reader, ret] = guest.get(2);
    ASSERT_EQ(0, ret.code()) << ret.message();
    ASSERT_NE(nullptr, reader);
    ASSERT_NE(nullptr, reader->get(2));

    auto drs = guest.reader();
    auto [next_reader, next_ret] = drs->next();
    ASSERT_EQ(0, next_ret.code()) << next_ret.message();
    ASSERT_NE(nullptr, next_reader);
    ASSERT_NE(nullptr, next_reader->get(2));
}

TEST_F(DatasetTest, WriterInitSucceedsAgainAfterPreviousOwnerIsDestroyed) {
    auto dir = make_dir("d_owner_release");
    generate_input_file(input_path_, cfg(3, 0, DataType::f32, 4));
    const std::string lock_path = dir + "/" + kOwnerLockFileName;

    {
        DatasetNode owner;
        ASSERT_EQ(0, owner.init({dir}, 100, DataType::f32, 4).code());
        ASSERT_EQ(0, owner.store(input_path_).code());
        EXPECT_FALSE(Singleton::instance().check_file_path(lock_path));
    }

    EXPECT_TRUE(Singleton::instance().check_file_path(lock_path));
    EXPECT_TRUE(Singleton::instance().release_file_path(lock_path));

    DatasetNode reopened;
    ASSERT_EQ(0, reopened.init({dir}, 100, DataType::f32, 4).code());
    const Ret reopened_ret = reopened.store(input_path_);
    EXPECT_EQ(0, reopened_ret.code()) << reopened_ret.message();
}


// --- UpdateNotifier integration ---

TEST_F(DatasetTest, StoreCreatesNotifierFile) {
    auto dir = make_dir("d_notifier_store");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4).code());

    generate_input_file(input_path_, cfg(5, 0, DataType::f32, 4));
    ASSERT_EQ(0, sc.store(input_path_).code());

    EXPECT_TRUE(fs::exists(dir + "/" + kOwnerLockFileName));
}

TEST_F(DatasetTest, StoreAccumulatorIncrementsNotifierCounter) {
    auto dir = make_dir("d_notifier_acc");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4).code());

    const std::array<float, 4> vec {1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT_EQ(0, sc.add_vector(1, reinterpret_cast<const uint8_t*>(vec.data())).code());
    ASSERT_EQ(0, sc.store_accumulator().code());

    ASSERT_EQ(0, sc.add_vector(2, reinterpret_cast<const uint8_t*>(vec.data())).code());
    ASSERT_EQ(0, sc.store_accumulator().code());

    // Read the raw counter from the lock file — should be 2 after two store_accumulator() calls.
    const std::string lock_path = dir + "/" + kOwnerLockFileName;
    int fd = open(lock_path.c_str(), O_RDONLY);
    ASSERT_GE(fd, 0);
    uint64_t counter = 0;
    ASSERT_EQ(static_cast<ssize_t>(sizeof(counter)), pread(fd, &counter, sizeof(counter), 0));
    close(fd);
    EXPECT_EQ(2u, counter);
}

TEST_F(DatasetTest, GuestDetectsOwnerStoreAndFlushesCache) {
    auto dir = make_dir("d_notifier_guest");

    // Owner stores initial data.
    DatasetNode owner;
    ASSERT_EQ(0, owner.init({dir}, 100, DataType::f32, 4).code());
    generate_input_file(input_path_, cfg(5, 0, DataType::f32, 4));
    ASSERT_EQ(0, owner.store(input_path_).code());

    // Guest reads data — populates its cache.
    DatasetReader guest;
    ASSERT_EQ(0, guest.init({dir}, 100, DataType::f32, 4).code());


    auto [reader0, ret0] = guest.get(2);
    ASSERT_EQ(0, ret0.code()) << ret0.message();
    ASSERT_NE(nullptr, reader0);
    const void* old_ptr = reader0.get();

    // Owner writes new data — bumps the notifier counter.
    write_input("f32,4\n2 : [ 99.0, 99.0, 99.0, 99.0 ]\n");
    ASSERT_EQ(0, owner.store(input_path_).code());

    // Guest's next read should see the update and return a fresh reader.
    auto [reader1, ret1] = guest.get(2);
    ASSERT_EQ(0, ret1.code()) << ret1.message();
    ASSERT_NE(nullptr, reader1);
    EXPECT_NE(old_ptr, reader1.get());

    const float* v = reinterpret_cast<const float*>(reader1->get(2));
    ASSERT_NE(nullptr, v);
    EXPECT_NEAR(99.0f, v[0], 1e-5f);
}

TEST_F(DatasetTest, GuestCacheStaysValidWhenNoUpdate) {
    auto dir = make_dir("d_notifier_no_update");

    DatasetNode owner;
    ASSERT_EQ(0, owner.init({dir}, 100, DataType::f32, 4).code());
    generate_input_file(input_path_, cfg(5, 0, DataType::f32, 4));
    ASSERT_EQ(0, owner.store(input_path_).code());

    DatasetReader guest;
    ASSERT_EQ(0, guest.init({dir}, 100, DataType::f32, 4).code());


    auto [reader0, ret0] = guest.get(2);
    ASSERT_EQ(0, ret0.code()) << ret0.message();
    ASSERT_NE(nullptr, reader0);
    const void* ptr0 = reader0.get();

    // No owner writes between reads — guest should reuse cached reader.
    auto [reader1, ret1] = guest.get(2);
    ASSERT_EQ(0, ret1.code()) << ret1.message();
    ASSERT_NE(nullptr, reader1);
    EXPECT_EQ(ptr0, reader1.get());
}

TEST_F(DatasetTest, MergeIncrementsNotifierCounter) {
    auto dir = make_dir("d_notifier_merge");
    DatasetNode sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4).code());

    // Store base data, then a small update to create a delta file.
    generate_input_file(input_path_, cfg(50, 0, DataType::f32, 4));
    ASSERT_EQ(0, sc.store(input_path_).code());
    write_input("f32,4\n0 : [ 99.0, 99.0, 99.0, 99.0 ]\n");
    ASSERT_EQ(0, sc.store(input_path_).code());
    ASSERT_TRUE(fs::exists(file_path(dir, 0, ".delta")));

    ASSERT_EQ(0, sc.merge().code());

    // Counter should be 3: two store() calls + one merge().
    const std::string lock_path = dir + "/" + kOwnerLockFileName;
    int fd = open(lock_path.c_str(), O_RDONLY);
    ASSERT_GE(fd, 0);
    uint64_t counter = 0;
    ASSERT_EQ(static_cast<ssize_t>(sizeof(counter)), pread(fd, &counter, sizeof(counter), 0));
    close(fd);
    EXPECT_EQ(3u, counter);
}


// Simulates the race where a writer merges a delta file between the time a
// guest reads its items cache and the time it opens the file.  The retry
// in get_cached_reader_ should invalidate the cache and succeed.
TEST_F(DatasetTest, GetRetriesWhenDeltaFileDeletedByWriter) {
    auto dir = make_dir("d_stale_retry");

    // Owner: store data, add a vector, flush to create a delta file.
    DatasetNode owner;
    ASSERT_EQ(0, owner.init({dir}, 100, DataType::f32, 4).code());
    generate_input_file(input_path_, cfg(5, 0, DataType::f32, 4));
    ASSERT_EQ(0, owner.store(input_path_).code());

    const std::array<float, 4> updated {99.0f, 99.0f, 99.0f, 99.0f};
    ASSERT_EQ(0, owner.add_vector(2, reinterpret_cast<const uint8_t*>(updated.data())).code());
    ASSERT_EQ(0, owner.store_accumulator().code());
    ASSERT_TRUE(fs::exists(dir + "/0.delta"));

    // Guest: populate items cache (sees 0.data + 0.delta).
    DatasetReader guest;
    ASSERT_EQ(0, guest.init({dir}, 100, DataType::f32, 4).code());

    auto [miss, miss_ret] = guest.get(999); // miss — but items cache is now populated
    ASSERT_EQ(0, miss_ret.code());

    // Simulate writer merge: delete the delta file without updating the
    // notifier, so the guest's items cache stays stale.
    fs::remove(dir + "/0.delta");
    ASSERT_FALSE(fs::exists(dir + "/0.delta"));

    // Guest tries to open a reader for file_id=0.  The first attempt uses the
    // stale delta path and fails; the retry rescans the directory (no delta)
    // and succeeds with the data file alone.
    auto [reader, ret] = guest.get(2);
    ASSERT_EQ(0, ret.code()) << ret.message();
    ASSERT_NE(nullptr, reader);
    ASSERT_NE(nullptr, reader->get(2));
}

TEST_F(DatasetTest, GetFailsWhenDataFileDeletedAndRetryAlsoFails) {
    auto dir = make_dir("d_stale_fail");

    DatasetNode owner;
    ASSERT_EQ(0, owner.init({dir}, 100, DataType::f32, 4).code());
    generate_input_file(input_path_, cfg(5, 0, DataType::f32, 4));
    ASSERT_EQ(0, owner.store(input_path_).code());

    DatasetReader guest;
    ASSERT_EQ(0, guest.init({dir}, 100, DataType::f32, 4).code());

    auto [miss, miss_ret] = guest.get(999);
    ASSERT_EQ(0, miss_ret.code());

    // Delete the data file.  The first open fails, the retry rescans the
    // directory and finds no files, so get() returns null (not found).
    fs::remove(dir + "/0.data");

    auto [reader, ret] = guest.get(2);
    ASSERT_EQ(0, ret.code());
    EXPECT_EQ(nullptr, reader);
}
