#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include "core/storage/input_generator.h"
#include "core/storage/data_file.h"
#include "core/storage/dataset.h"
#include "core/storage/data_reader.h"
#include "utils/ini_reader.h"

using namespace sketch2;
namespace fs = std::filesystem;

class DatasetTest : public ::testing::Test {
protected:
    std::string base_dir_;
    std::string input_path_;
    std::string config_path_;

    void SetUp() override {
        base_dir_   = "/tmp/sketch2_utest_sc_" + std::to_string(getpid());
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
    Dataset sc;
    EXPECT_NE(0, sc.init({}, 100, DataType::f32, 4).code());
}

TEST_F(DatasetTest, InitFailsOnZeroRangeSize) {
    Dataset sc;
    EXPECT_NE(0, sc.init({make_dir("d")}, 0, DataType::f32, 4).code());
}

TEST_F(DatasetTest, InitFailsOnDoubleInit) {
    Dataset sc;
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
        "dim = 4\n");

    IniReader ini_cfg;
    ASSERT_EQ(0, ini_cfg.init(config_path_).code());
    EXPECT_EQ(4, ini_cfg.get_int("dataset.dim", -1));

    Dataset sc;
    const Ret ret = sc.init(config_path_);
    ASSERT_EQ(0, ret.code()) << ret.message();
    generate_input_file(input_path_, cfg(30, 0, DataType::f32, 4));
    ASSERT_EQ(0, sc.store(input_path_).code());
    EXPECT_TRUE(fs::exists(dir0 + "/0.data"));
    EXPECT_TRUE(fs::exists(dir1 + "/1.data"));
    EXPECT_TRUE(fs::exists(dir0 + "/2.data"));
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

    Dataset sc;
    ASSERT_EQ(0, sc.init(config_path_).code());

    const std::array<float, 4> vec {1.0f, 2.0f, 3.0f, 4.0f};
    const Ret ret = sc.add_vector(1, reinterpret_cast<const uint8_t*>(vec.data()));
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("Accumulator: buffer full", ret.message());
}

TEST_F(DatasetTest, SetGuestModeRejectsMutationsAfterInitFromMetadata) {
    DatasetMetadata metadata;
    metadata.dirs = {make_dir("d_guest_metadata")};
    metadata.range_size = 10;
    metadata.type = DataType::f32;
    metadata.dim = 4;

    Dataset sc;
    ASSERT_EQ(0, sc.init(metadata).code());
    ASSERT_EQ(0, sc.set_guest_mode().code());

    generate_input_file(input_path_, cfg(1, 0, DataType::f32, 4));
    const Ret ret = sc.store(input_path_);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("Dataset: guest mode is read-only", ret.message());
}

TEST_F(DatasetTest, InitFromIniFailsOnMissingType) {
    auto dir = make_dir("d");
    write_config(
        std::string("[dataset]\n") +
        "dirs = " + dir + "\n"
        "range_size = 100\n"
        "dims = 4\n");

    Dataset sc;
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

    Dataset sc;
    const Ret ret = sc.init(config_path_);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("Dataset: dataset.dim must be >= 0", ret.message());
}

TEST_F(DatasetTest, InitFromIniFailsOnNegativeRangeSize) {
    auto dir = make_dir("d_neg_range");
    write_config(
        std::string("[dataset]\n") +
        "dirs = " + dir + "\n"
        "range_size = -1\n"
        "type = f32\n"
        "dim = 4\n");

    Dataset sc;
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

    Dataset sc;
    const Ret ret = sc.init(config_path_);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("Dataset: dataset.accumulator_size must be >= 0", ret.message());
}

// --- load error cases ---

TEST_F(DatasetTest, StoreFailsWhenNotInitialized) {
    Dataset sc;
    EXPECT_NE(0, sc.store(input_path_).code());
}

TEST_F(DatasetTest, StoreAccumulatorFailsWhenNotInitialized) {
    Dataset sc;
    EXPECT_NE(0, sc.store_accumulator().code());
}

TEST_F(DatasetTest, AddVectorFailsWhenNotInitialized) {
    Dataset sc;
    const std::array<float, 4> vec {1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_NE(0, sc.add_vector(1, reinterpret_cast<const uint8_t*>(vec.data())).code());
}

TEST_F(DatasetTest, DeleteVectorFailsWhenNotInitialized) {
    Dataset sc;
    EXPECT_NE(0, sc.delete_vector(1).code());
}

TEST_F(DatasetTest, GuestModeRejectsMutatingOperations) {
    auto dir = make_dir("d_guest_mut");
    Dataset sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4).code());
    ASSERT_EQ(0, sc.set_guest_mode().code());

    const std::array<float, 4> vec {1.0f, 2.0f, 3.0f, 4.0f};
    generate_input_file(input_path_, cfg(3, 0, DataType::f32, 4));

    const Ret store_ret = sc.store(input_path_);
    EXPECT_NE(0, store_ret.code());
    EXPECT_EQ("Dataset: guest mode is read-only", store_ret.message());

    const Ret add_ret = sc.add_vector(1, reinterpret_cast<const uint8_t*>(vec.data()));
    EXPECT_NE(0, add_ret.code());
    EXPECT_EQ("Dataset: guest mode is read-only", add_ret.message());

    const Ret delete_ret = sc.delete_vector(1);
    EXPECT_NE(0, delete_ret.code());
    EXPECT_EQ("Dataset: guest mode is read-only", delete_ret.message());

    const Ret flush_ret = sc.store_accumulator();
    EXPECT_NE(0, flush_ret.code());
    EXPECT_EQ("Dataset: guest mode is read-only", flush_ret.message());

    const Ret merge_ret = sc.merge();
    EXPECT_NE(0, merge_ret.code());
    EXPECT_EQ("Dataset: guest mode is read-only", merge_ret.message());
}

TEST_F(DatasetTest, StoreFailsOnBadInputPath) {
    Dataset sc;
    ASSERT_EQ(0, sc.init({make_dir("d")}, 100, DataType::f32, 4).code());
    EXPECT_NE(0, sc.store("/nonexistent/dir/input.txt").code());
}

TEST_F(DatasetTest, StoreFailsOnBadOutputDir) {
    generate_input_file(input_path_, cfg(3, 0, DataType::f32, 4));
    Dataset sc;
    ASSERT_EQ(0, sc.init({"/nonexistent/output/dir"}, 100, DataType::f32, 4).code());
    EXPECT_NE(0, sc.store(input_path_).code());
}

// --- load success: file placement ---

TEST_F(DatasetTest, StoreCreatesOutputFile) {
    auto dir = make_dir("d");
    generate_input_file(input_path_, cfg(5, 0, DataType::f32, 4));
    Dataset sc;
    ASSERT_EQ(0, sc.init({dir}, 1000, DataType::f32, 4).code());
    ASSERT_EQ(0, sc.store(input_path_).code());
    EXPECT_TRUE(fs::exists(dir + "/0.data"));
}

TEST_F(DatasetTest, StoreAccumulatorSucceedsWhenMissing) {
    auto dir = make_dir("d_acc_none");
    Dataset sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4).code());
    EXPECT_EQ(0, sc.store_accumulator().code());
}

TEST_F(DatasetTest, StoreAccumulatorCreatesOutputFilesFromAccumulator) {
    auto dir = make_dir("d_acc_store");
    Dataset sc;
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

TEST_F(DatasetTest, StoreAccumulatorIgnoresDeleteForMissingRange) {
    auto dir = make_dir("d_acc_missing_delete");
    Dataset sc;
    ASSERT_EQ(0, sc.init({dir}, 10, DataType::f32, 4).code());

    ASSERT_EQ(0, sc.delete_vector(25).code());
    ASSERT_EQ(0, sc.store_accumulator().code());

    EXPECT_FALSE(fs::exists(file_path(dir, 2, ".data")));
    EXPECT_FALSE(fs::exists(file_path(dir, 2, ".delta")));
}

TEST_F(DatasetTest, StoreAccumulatorClearsAccumulatorAfterSuccess) {
    auto dir = make_dir("d_acc_clear");
    Dataset sc;
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
        Dataset sc;
        ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4).code());
        ASSERT_EQ(0, sc.add_vector(17, reinterpret_cast<const uint8_t*>(vec.data())).code());
    }

    const std::string wal_path = dir + "/sketch2.accumulator.wal";
    ASSERT_TRUE(fs::exists(wal_path));

    Dataset reopened;
    ASSERT_EQ(0, reopened.init({dir}, 100, DataType::f32, 4).code());
    ASSERT_EQ(0, reopened.store_accumulator().code());

    DataReader dr;
    ASSERT_EQ(0, dr.init(file_path(dir, 0, ".data")).code());
    const float* stored = reinterpret_cast<const float*>(dr.get(17));
    ASSERT_NE(nullptr, stored);
    EXPECT_NEAR(3.0f, stored[0], 1e-5f);
    EXPECT_EQ(static_cast<uintmax_t>(sizeof(WalFileHeader)), fs::file_size(wal_path));
}

TEST_F(DatasetTest, AddVectorCreatesAccumulatorLazily) {
    auto dir = make_dir("d_acc_add");
    Dataset sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4).code());

    const std::array<float, 4> vec {1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_EQ(0, sc.add_vector(1, reinterpret_cast<const uint8_t*>(vec.data())).code());
}

TEST_F(DatasetTest, AddVectorFlushesAccumulatorWhenFull) {
    auto dir = make_dir("d_acc_add_flush");
    Dataset sc;
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
    Dataset sc;
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
    Dataset sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4, 23).code());

    const std::array<float, 4> vec {1.0f, 2.0f, 3.0f, 4.0f};
    const Ret ret = sc.add_vector(1, reinterpret_cast<const uint8_t*>(vec.data()));
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("Accumulator: buffer full", ret.message());
}

TEST_F(DatasetTest, DeleteVectorCreatesAccumulatorLazily) {
    auto dir = make_dir("d_acc_del");
    Dataset sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4).code());

    EXPECT_EQ(0, sc.delete_vector(1).code());
}

TEST_F(DatasetTest, DeleteVectorFlushesAccumulatorWhenFull) {
    auto dir = make_dir("d_acc_del_flush");
    Dataset sc;
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
    Dataset sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4, 7).code());

    const Ret ret = sc.delete_vector(1);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("Accumulator: buffer full", ret.message());
}

TEST_F(DatasetTest, StoreFileIdFromMinId) {
    // ids [2005..2009], range_size=1000 -> file_id=2 -> "2.data"
    auto dir = make_dir("d");
    generate_input_file(input_path_, cfg(5, 2005, DataType::f32, 4));
    Dataset sc;
    ASSERT_EQ(0, sc.init({dir}, 1000, DataType::f32, 4).code());
    ASSERT_EQ(0, sc.store(input_path_).code());
    EXPECT_TRUE(fs::exists(dir + "/2.data"));
    EXPECT_FALSE(fs::exists(dir + "/0.data"));
}

TEST_F(DatasetTest, StoreMultipleRangesCreateMultipleFiles) {
    // ids [5..19], range_size=10 -> file 0 (ids 5-9), file 1 (ids 10-19)
    auto dir = make_dir("d");
    generate_input_file(input_path_, cfg(15, 5, DataType::f32, 4));
    Dataset sc;
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
    Dataset sc;
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
    Dataset sc;
    ASSERT_EQ(0, sc.init({dir}, 1000, DataType::f32, 4).code());
    ASSERT_EQ(0, sc.store(input_path_).code());
    DataReader dr;
    ASSERT_EQ(0, dr.init(dir + "/0.data").code());
    EXPECT_EQ(7u, dr.count());
}

TEST_F(DatasetTest, StoreFileType) {
    auto dir = make_dir("d");
    generate_input_file(input_path_, cfg(3, 0, DataType::f32, 4));
    Dataset sc;
    ASSERT_EQ(0, sc.init({dir}, 1000, DataType::f32, 4).code());
    ASSERT_EQ(0, sc.store(input_path_).code());
    DataReader dr;
    ASSERT_EQ(0, dr.init(dir + "/0.data").code());
    EXPECT_EQ(DataType::f32, dr.type());
}

TEST_F(DatasetTest, StoreFileDim) {
    auto dir = make_dir("d");
    generate_input_file(input_path_, cfg(3, 0, DataType::f32, 64));
    Dataset sc;
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
    Dataset sc;
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
    Dataset sc;
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

    Dataset sc;
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

    Dataset sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4).code());
    ASSERT_EQ(0, sc.store(input_path_).code());

    EXPECT_FALSE(fs::exists(file_path(dir, 0, ".data")));
    EXPECT_FALSE(fs::exists(file_path(dir, 0, ".delta")));
}

TEST_F(DatasetTest, SecondSmallLoadCreatesDeltaFile) {
    auto dir = make_dir("d");
    Dataset sc;
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
    Dataset sc;
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
    Dataset sc;
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
    Dataset sc;
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
    Dataset sc;
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
    Dataset sc;
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
    Dataset sc;
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
    Dataset sc;
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
    Dataset sc;
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

// --- DatasetReader::next() ---

TEST_F(DatasetTest, DatasetReaderNextReturnsNullWhenExhausted) {
    auto dir = make_dir("d");
    generate_input_file(input_path_, cfg(5, 0, DataType::f32, 4));
    Dataset sc;
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

TEST_F(DatasetTest, DatasetReaderNextOnEmptyDatasetReturnsNull) {
    auto dir = make_dir("empty");
    Dataset sc;
    ASSERT_EQ(0, sc.init({dir}, 1000, DataType::f32, 4).code());
    // No store() call — no files in dir.
    auto drs = sc.reader();
    auto [reader, ret] = drs->next();
    EXPECT_EQ(0, ret.code());
    EXPECT_EQ(nullptr, reader);
}

TEST_F(DatasetTest, DatasetGetOnEmptyDatasetReturnsNull) {
    auto dir = make_dir("empty_get");
    Dataset sc;
    ASSERT_EQ(0, sc.init({dir}, 1000, DataType::f32, 4).code());

    auto [reader, ret] = sc.get(42);
    EXPECT_EQ(0, ret.code());
    EXPECT_EQ(nullptr, reader);
}

TEST_F(DatasetTest, DatasetReaderNextIteratesAllFiles) {
    auto dir = make_dir("d");
    // ids 0..29 with range_size=10 → 3 data files: 0.data, 1.data, 2.data.
    generate_input_file(input_path_, cfg(30, 0, DataType::f32, 4));
    Dataset sc;
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

TEST_F(DatasetTest, DatasetReaderNextReturnsDeltaFileAlongsideData) {
    auto dir = make_dir("d");
    Dataset sc;
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

TEST_F(DatasetTest, DatasetReaderGetReturnsReaderForContainingRange) {
    auto dir = make_dir("d_get");
    generate_input_file(input_path_, cfg(30, 0, DataType::f32, 4)); // files 0,1,2 for range_size=10
    Dataset sc;
    ASSERT_EQ(0, sc.init({dir}, 10, DataType::f32, 4).code());
    ASSERT_EQ(0, sc.store(input_path_).code());

    auto drs = sc.reader();
    auto [reader, ret] = drs->get(17);
    ASSERT_EQ(0, ret.code()) << ret.message();
    ASSERT_NE(nullptr, reader);
    ASSERT_NE(nullptr, reader->get(17));
    ASSERT_EQ(nullptr, reader->get(29));
}

TEST_F(DatasetTest, DatasetReaderGetReturnsNullWhenRangeHasNoFile) {
    auto dir = make_dir("d_get_sparse");
    write_input(
        "f32,4\n"
        "0 : [ 0.1, 0.1, 0.1, 0.1 ]\n"
        "20 : [ 20.1, 20.1, 20.1, 20.1 ]\n");
    Dataset sc;
    ASSERT_EQ(0, sc.init({dir}, 10, DataType::f32, 4).code());
    ASSERT_EQ(0, sc.store(input_path_).code());

    auto drs = sc.reader();
    auto [reader, ret] = drs->get(15); // file_id=1 is missing
    ASSERT_EQ(0, ret.code()) << ret.message();
    EXPECT_EQ(nullptr, reader);
}

TEST_F(DatasetTest, DatasetGetReturnsReaderWithDeltaApplied) {
    auto dir = make_dir("d_ds_get");
    Dataset sc;
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

TEST_F(DatasetTest, GuestModeAllowsQueries) {
    auto dir = make_dir("d_guest_query");

    Dataset owner;
    ASSERT_EQ(0, owner.init({dir}, 100, DataType::f32, 4).code());
    generate_input_file(input_path_, cfg(5, 0, DataType::f32, 4));
    ASSERT_EQ(0, owner.store(input_path_).code());

    Dataset guest;
    ASSERT_EQ(0, guest.init({dir}, 100, DataType::f32, 4).code());
    ASSERT_EQ(0, guest.set_guest_mode().code());

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

TEST_F(DatasetTest, SetGuestModeFailsWhenAccumulatorIsNotEmpty) {
    auto dir = make_dir("d_guest_pending");
    Dataset sc;
    ASSERT_EQ(0, sc.init({dir}, 100, DataType::f32, 4).code());

    const std::array<float, 4> vec {1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT_EQ(0, sc.add_vector(1, reinterpret_cast<const uint8_t*>(vec.data())).code());

    const Ret ret = sc.set_guest_mode();
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("Dataset: cannot switch to guest mode with non-empty accumulator", ret.message());
}
