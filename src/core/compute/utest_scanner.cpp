// Unit tests for nearest-neighbor scanning over readers and datasets.

#include <gtest/gtest.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <unistd.h>
#include <vector>
#include <fstream>
#include <memory>
#include <filesystem>
#include <experimental/scope>
#include "core/compute/scanner.h"
#include "core/utils/singleton.h"
#include "core/utils/thread_pool.h"
#include "core/storage/input_generator.h"
#include "core/storage/data_writer.h"
#include "core/storage/data_reader.h"
#include "core/storage/dataset_node.h"
#include "utest_tmp_dir.h"

using namespace sketch2;
namespace fs = std::filesystem;

class ScannerTest : public ::testing::Test {
protected:
    std::string input_path_;
    std::string data_path_;
    std::string delta_input_path_;
    std::string delta_path_;
    std::vector<std::string> cleanup_dirs_;
    std::vector<std::string> cleanup_files_;

    void SetUp() override {
        std::string base = tmp_dir() + "/sketch2_utest_sc_" + std::to_string(getpid());
        input_path_ = base + ".txt";
        data_path_  = base + ".bin";
        delta_input_path_ = base + ".delta.txt";
        delta_path_ = base + ".delta.bin";
    }

    void TearDown() override {
        std::remove(input_path_.c_str());
        std::remove(data_path_.c_str());
        std::remove(delta_input_path_.c_str());
        std::remove(delta_path_.c_str());
        for (const std::string& path : cleanup_files_) {
            std::remove(path.c_str());
        }
        for (const std::string& path : cleanup_dirs_) {
            fs::remove_all(path);
        }
    }

    void generate_file(const std::string& in_path, const std::string& out_path, const GeneratorConfig& cfg) {
        generate_input_file(in_path, cfg);
        DataWriter w;
        w.init(in_path, out_path);
        ASSERT_EQ(0, w.exec().code());
    }

    void generate(size_t count, size_t min_id, DataType type, size_t dim) {
        GeneratorConfig cfg{PatternType::Sequential, count, min_id, type, dim, 1000};
        generate_file(input_path_, data_path_, cfg);
    }

    void generate_delta(size_t count, size_t min_id, DataType type, size_t dim, size_t every_n_deleted = 0) {
        GeneratorConfig cfg{PatternType::Sequential, count, min_id, type, dim, 1000, every_n_deleted};
        generate_file(delta_input_path_, delta_path_, cfg);
    }

    void write_delta_raw(const std::string& content) {
        std::ofstream f(delta_input_path_);
        f << content;
        f.close();
        DataWriter w;
        w.init(delta_input_path_, delta_path_);
        ASSERT_EQ(0, w.exec().code());
    }

    void write_input_raw(const std::string& path, const std::string& content) {
        std::ofstream f(path);
        f << content;
        f.close();
    }

    std::unique_ptr<DatasetReader> make_dataset_reader(
            DataType type,
            uint64_t dim,
            DistFunc func,
            const std::vector<std::string>& store_inputs,
            uint64_t range_size = 1000) {
        const std::string dataset_dir = data_path_ + ".dataset_" + std::to_string(cleanup_dirs_.size());
        const std::string config_path = data_path_ + ".dataset_" + std::to_string(cleanup_dirs_.size()) + ".ini";
        cleanup_dirs_.push_back(dataset_dir);
        cleanup_files_.push_back(config_path);
        fs::create_directories(dataset_dir);

        DatasetNode ds;
        EXPECT_EQ(0, ds.init({dataset_dir}, range_size, type, dim, kAccumulatorBufferSize, func).code());
        for (const std::string& input : store_inputs) {
            EXPECT_EQ(0, ds.store(input).code());
        }

        write_input_raw(
            config_path,
            std::string("[dataset]\n") +
            "dirs = " + dataset_dir + "\n"
            "range_size = " + std::to_string(range_size) + "\n"
            "type = " + data_type_to_string(type) + "\n"
            "dist_func = " + dist_func_to_string(func) + "\n"
            "dim = " + std::to_string(dim) + "\n");

        auto reader = std::make_unique<DatasetReader>();
        EXPECT_EQ(0, reader->init(config_path).code());
        return reader;
    }

    std::vector<uint8_t> f32_vec(float val, size_t dim) {
        std::vector<uint8_t> buf(dim * sizeof(float));
        auto* p = reinterpret_cast<float*>(buf.data());
        for (size_t i = 0; i < dim; ++i) p[i] = val;
        return buf;
    }

    std::vector<uint8_t> f32_values(std::initializer_list<float> values) {
        std::vector<uint8_t> buf(values.size() * sizeof(float));
        auto* p = reinterpret_cast<float*>(buf.data());
        size_t i = 0;
        for (float v : values) {
            p[i++] = v;
        }
        return buf;
    }

    std::vector<uint8_t> i16_vec(int16_t val, size_t dim) {
        std::vector<uint8_t> buf(dim * sizeof(int16_t));
        auto* p = reinterpret_cast<int16_t*>(buf.data());
        for (size_t i = 0; i < dim; ++i) p[i] = val;
        return buf;
    }

    std::vector<uint8_t> f16_vec(float val, size_t dim) {
        std::vector<uint8_t> buf(dim * sizeof(uint16_t));
        auto* p = reinterpret_cast<uint16_t*>(buf.data());
        for (size_t i = 0; i < dim; ++i) p[i] = float_to_f16(val);
        return buf;
    }

    static uint16_t float_to_f16(float f) {
        uint32_t x;
        memcpy(&x, &f, sizeof(x));
        uint16_t sign     = static_cast<uint16_t>((x >> 16) & 0x8000);
        int      exp      = static_cast<int>((x >> 23) & 0xFF) - 127 + 15;
        uint32_t mantissa = x & 0x7FFFFFu;
        if (exp <= 0)  return sign;
        if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
        return static_cast<uint16_t>(sign | (exp << 10) | (mantissa >> 13));
    }
};

TEST_F(ScannerTest, FindFailsOnCountZero) {
    generate(3, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::L1, {input_path_});
    Scanner s;
    auto q = f32_vec(0.0f, 4);
    std::vector<uint64_t> result;
    EXPECT_NE(0, s.find(*reader, 0, q.data(), result).code());
}

TEST_F(ScannerTest, FindFailsOnNullQueryPointer) {
    generate(3, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::L1, {input_path_});
    Scanner s;
    std::vector<uint64_t> result;
    EXPECT_NE(0, s.find(*reader, 1, nullptr, result).code());
}

TEST_F(ScannerTest, FindFailsOnUnknownFunction) {
    DatasetReader reader;
    Scanner s;
    auto q = f32_vec(0.0f, 4);
    std::vector<uint64_t> result;
    EXPECT_NE(0, s.find(reader, 1, q.data(), result).code());
}

TEST_F(ScannerTest, FindCountExceedsTotalReturnsCapped) {
    const size_t total = 3;
    generate(total, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::L1, {input_path_});
    Scanner s;
    auto q = f32_vec(0.0f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 100, q.data(), result).code());
    EXPECT_EQ(total, result.size());
}

TEST_F(ScannerTest, FindResultSizeMatchesRequest) {
    generate(5, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::L1, {input_path_});
    Scanner s;
    auto q = f32_vec(3.2f, 4);
    std::vector<uint64_t> result;

    ASSERT_EQ(0, s.find(*reader, 1, q.data(), result).code());
    EXPECT_EQ(1u, result.size());

    ASSERT_EQ(0, s.find(*reader, 3, q.data(), result).code());
    EXPECT_EQ(3u, result.size());

    ASSERT_EQ(0, s.find(*reader, 5, q.data(), result).code());
    EXPECT_EQ(5u, result.size());
}

TEST_F(ScannerTest, FindF32K3ReturnsInOrder) {
    generate(5, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::L1, {input_path_});
    Scanner s;
    auto q = f32_vec(3.2f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(3u, result[0]);
    EXPECT_EQ(4u, result[1]);
    EXPECT_EQ(2u, result[2]);
}

TEST_F(ScannerTest, FindItemsF32ReturnsIdsAndDistancesInOrder) {
    const std::string dataset_dir = data_path_ + ".dataset_l1";
    const std::string config_path = data_path_ + ".dataset_l1.ini";
    std::experimental::scope_exit cleanup([&]() {
        fs::remove_all(dataset_dir);
        std::remove(config_path.c_str());
    });
    fs::create_directories(dataset_dir);

    generate_input_file(input_path_, GeneratorConfig{PatternType::Sequential, 5, 0, DataType::f32, 4, 1000});
    DatasetNode ds;
    ASSERT_EQ(0, ds.init({dataset_dir}, 1000, DataType::f32, 4, kAccumulatorBufferSize, DistFunc::L1).code());
    ASSERT_EQ(0, ds.store(input_path_).code());

    write_input_raw(
        config_path,
        std::string("[dataset]\n") +
        "dirs = " + dataset_dir + "\n"
        "range_size = 1000\n"
        "type = f32\n"
        "dist_func = l1\n"
        "dim = 4\n");

    DatasetReader reader;
    ASSERT_EQ(0, reader.init(config_path).code());
    Scanner s;
    auto q = f32_vec(3.2f, 4);
    std::vector<DistItem> result;
    ASSERT_EQ(0, s.find_items(reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(3u, result[0].id);
    EXPECT_EQ(4u, result[1].id);
    EXPECT_EQ(2u, result[2].id);
    EXPECT_NEAR(0.4, result[0].dist, 1e-5);
    EXPECT_NEAR(3.6, result[1].dist, 1e-5);
    EXPECT_NEAR(4.4, result[2].dist, 1e-5);
}

TEST_F(ScannerTest, FindF32L2K3ReturnsInOrder) {
    generate(5, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::L2, {input_path_});
    Scanner s;
    auto q = f32_vec(3.2f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(3u, result[0]);
    EXPECT_EQ(4u, result[1]);
    EXPECT_EQ(2u, result[2]);
}

TEST_F(ScannerTest, FindF32CosK3ReturnsInOrder) {
    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 100.0, 1.0, 0.0, 0.0 ]\n"
        "20 : [ 1.0, 1.0, 0.0, 0.0 ]\n"
        "30 : [ -1.0, 0.0, 0.0, 0.0 ]\n");
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::COS, {input_path_});
    Scanner s;
    auto q = f32_values({1.0f, 0.0f, 0.0f, 0.0f});
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(10u, result[0]);
    EXPECT_EQ(20u, result[1]);
    EXPECT_EQ(30u, result[2]);
}

TEST_F(ScannerTest, FindF32CosK3ReturnsInOrderWithStoredCosineValues) {
    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 100.0, 1.0, 0.0, 0.0 ]\n"
        "20 : [ 1.0, 1.0, 0.0, 0.0 ]\n"
        "30 : [ -1.0, 0.0, 0.0, 0.0 ]\n");
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::COS, {input_path_});

    Scanner s;
    auto q = f32_values({1.0f, 0.0f, 0.0f, 0.0f});
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 3, q.data(), result).code());
    ASSERT_EQ((std::vector<uint64_t> {10u, 20u, 30u}), result);
}

TEST_F(ScannerTest, FindF32CosStoredCosineValuesHandleZeroVectors) {
    const std::string dataset_dir = data_path_ + ".dataset_cos";
    const std::string config_path = data_path_ + ".dataset_cos.ini";
    std::experimental::scope_exit cleanup([&]() {
        fs::remove_all(dataset_dir);
        std::remove(config_path.c_str());
    });
    fs::create_directories(dataset_dir);

    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 0.0, 0.0, 0.0, 0.0 ]\n"
        "20 : [ 1.0, 0.0, 0.0, 0.0 ]\n");
    DatasetNode ds;
    ASSERT_EQ(0, ds.init({dataset_dir}, 1000, DataType::f32, 4, kAccumulatorBufferSize, DistFunc::COS).code());
    ASSERT_EQ(0, ds.store(input_path_).code());

    write_input_raw(
        config_path,
        std::string("[dataset]\n") +
        "dirs = " + dataset_dir + "\n"
        "range_size = 1000\n"
        "type = f32\n"
        "dist_func = cos\n"
        "dim = 4\n");

    DatasetReader reader;
    ASSERT_EQ(0, reader.init(config_path).code());

    Scanner s;
    auto q = f32_values({0.0f, 0.0f, 0.0f, 0.0f});
    std::vector<DistItem> result;
    ASSERT_EQ(0, s.find_items(reader, 2, q.data(), result).code());
    ASSERT_EQ(2u, result.size());
    EXPECT_EQ(10u, result[0].id);
    EXPECT_DOUBLE_EQ(0.0, result[0].dist);
    EXPECT_EQ(20u, result[1].id);
    EXPECT_DOUBLE_EQ(1.0, result[1].dist);
}

TEST_F(ScannerTest, FindF32CosStoredAndComputedPathsMatchRanking) {
    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 10.0, 0.0, 0.0, 0.0 ]\n"
        "20 : [ 2.0, 1.0, 0.0, 0.0 ]\n"
        "30 : [ 1.0, 2.0, 0.0, 0.0 ]\n"
        "40 : [ 0.0, 1.0, 0.0, 0.0 ]\n"
        "50 : [ -1.0, 0.0, 0.0, 0.0 ]\n");

    auto plain_reader = make_dataset_reader(DataType::f32, 4, DistFunc::COS, {input_path_});
    auto inv_reader = make_dataset_reader(DataType::f32, 4, DistFunc::COS, {input_path_});

    Scanner s;
    auto q = f32_values({1.0f, 0.0f, 0.0f, 0.0f});
    std::vector<uint64_t> plain_result;
    std::vector<uint64_t> inv_result;
    ASSERT_EQ(0, s.find(*plain_reader, 5, q.data(), plain_result).code());
    ASSERT_EQ(0, s.find(*inv_reader, 5, q.data(), inv_result).code());

    ASSERT_EQ((std::vector<uint64_t> {10u, 20u, 30u, 40u, 50u}), plain_result);
    EXPECT_EQ(plain_result, inv_result);
}

TEST_F(ScannerTest, FindI16AllSortedByDistance) {
    generate(3, 0, DataType::i16, 4);
    auto reader = make_dataset_reader(DataType::i16, 4, DistFunc::L1, {input_path_});
    Scanner s;
    auto q = i16_vec(0, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(0u, result[0]);
    EXPECT_EQ(1u, result[1]);
    EXPECT_EQ(2u, result[2]);
}

TEST_F(ScannerTest, FindF16Works) {
    generate(3, 0, DataType::f16, 4);
    auto reader = make_dataset_reader(DataType::f16, 4, DistFunc::L1, {input_path_});
    Scanner s;
    auto q = f16_vec(1.1f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 1, q.data(), result).code());
    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(1u, result[0]);
}

TEST_F(ScannerTest, DeltaSkipsDeletedIds) {
    generate(6, 0, DataType::f32, 4);
    generate_delta(6, 0, DataType::f32, 4, 2); // deleted ids: 2,4

    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::L1, {input_path_, delta_input_path_});

    Scanner s;
    auto q = f32_vec(3.2f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 6, q.data(), result).code());

    for (uint64_t id : result) {
        EXPECT_NE(2u, id);
        EXPECT_NE(4u, id);
    }
}

TEST_F(ScannerTest, DeltaDeletingAllVectorsReturnsEmptyResult) {
    generate(3, 0, DataType::f32, 4);
    write_delta_raw(
        "f32,4\n"
        "0 : []\n"
        "1 : []\n"
        "2 : []\n");

    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::L1, {input_path_, delta_input_path_});

    Scanner s;
    auto q = f32_vec(1.1f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 3, q.data(), result).code());
    EXPECT_TRUE(result.empty());
}

TEST_F(ScannerTest, DeltaUsesUpdatedVectors) {
    generate(4, 10, DataType::f32, 4); // values 10.1, 11.1, 12.1, 13.1
    write_delta_raw(
        "f32,4\n"
        "11 : [ 20.0, 20.0, 20.0, 20.0 ]\n");

    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::L1, {input_path_, delta_input_path_});

    Scanner s;
    auto q = f32_vec(20.0f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 1, q.data(), result).code());
    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(11u, result[0]);
}

TEST_F(ScannerTest, FindDatasetWorks) {
    std::string d0 = tmp_dir() + "/sketch2_utest_sc_ds0_" + std::to_string(getpid());
    std::string d1 = tmp_dir() + "/sketch2_utest_sc_ds1_" + std::to_string(getpid());
    fs::create_directories(d0);
    fs::create_directories(d1);
    std::experimental::scope_exit cleanup([&]() {
        fs::remove_all(d0);
        fs::remove_all(d1);
    });

    DatasetNode ds;
    ASSERT_EQ(0, ds.init({d0, d1}, 10, DataType::f32, 4).code());

    // Generate 30 items: 0..29.
    // They go into 0.data (0..9), 1.data (10..19), 2.data (20..29).
    generate_input_file(input_path_, GeneratorConfig{PatternType::Sequential, 30, 0, DataType::f32, 4, 1000});
    ASSERT_EQ(0, ds.store(input_path_).code());

    Scanner s;
    auto q = f32_vec(15.2f, 4); // nearest to id 15
    std::vector<uint64_t> result;
    const auto ret = s.find(ds, 3, q.data(), result);
    ASSERT_EQ(0, ret.code()) << "\n\nfind failed: " << ret.message() << "\n\n";
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(15u, result[0]);
    EXPECT_EQ(16u, result[1]);
    EXPECT_EQ(14u, result[2]);
}

TEST_F(ScannerTest, FindDatasetItemsReturnsIdsAndDistancesInOrder) {
    std::string d = tmp_dir() + "/sketch2_utest_sc_dsitems_" + std::to_string(getpid());
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    DatasetNode ds;
    ASSERT_EQ(0, ds.init({d}, 100, DataType::f32, 4).code());
    generate_input_file(input_path_, GeneratorConfig{PatternType::Sequential, 30, 0, DataType::f32, 4, 1000});
    ASSERT_EQ(0, ds.store(input_path_).code());

    Scanner s;
    auto q = f32_vec(15.2f, 4);
    std::vector<DistItem> result;
    ASSERT_EQ(0, s.find_items(ds, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(15u, result[0].id);
    EXPECT_EQ(16u, result[1].id);
    EXPECT_EQ(14u, result[2].id);
    EXPECT_NEAR(0.4, result[0].dist, 1e-5);
    EXPECT_NEAR(3.6, result[1].dist, 1e-5);
    EXPECT_NEAR(4.4, result[2].dist, 1e-5);
}

TEST_F(ScannerTest, FindDatasetL2Works) {
    std::string d0 = tmp_dir() + "/sketch2_utest_sc_l2ds0_" + std::to_string(getpid());
    std::string d1 = tmp_dir() + "/sketch2_utest_sc_l2ds1_" + std::to_string(getpid());
    fs::create_directories(d0);
    fs::create_directories(d1);
    std::experimental::scope_exit cleanup([&]() {
        fs::remove_all(d0);
        fs::remove_all(d1);
    });

    DatasetNode ds;
    ASSERT_EQ(0, ds.init({d0, d1}, 10, DataType::f32, 4, kAccumulatorBufferSize, DistFunc::L2).code());
    generate_input_file(input_path_, GeneratorConfig{PatternType::Sequential, 30, 0, DataType::f32, 4, 1000});
    ASSERT_EQ(0, ds.store(input_path_).code());

    Scanner s;
    auto q = f32_vec(15.2f, 4);
    std::vector<uint64_t> result;
    const auto ret = s.find(ds, 3, q.data(), result);
    ASSERT_EQ(0, ret.code()) << "\n\nfind failed: " << ret.message() << "\n\n";
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(15u, result[0]);
    EXPECT_EQ(16u, result[1]);
    EXPECT_EQ(14u, result[2]);
}

TEST_F(ScannerTest, FindDatasetCosWorks) {
    std::string d = tmp_dir() + "/sketch2_utest_sc_cosds_" + std::to_string(getpid());
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    DatasetNode ds;
    ASSERT_EQ(0, ds.init({d}, 100, DataType::f32, 4, kAccumulatorBufferSize, DistFunc::COS).code());
    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 100.0, 1.0, 0.0, 0.0 ]\n"
        "20 : [ 1.0, 1.0, 0.0, 0.0 ]\n"
        "30 : [ -1.0, 0.0, 0.0, 0.0 ]\n");
    ASSERT_EQ(0, ds.store(input_path_).code());

    Scanner s;
    auto q = f32_values({1.0f, 0.0f, 0.0f, 0.0f});
    std::vector<uint64_t> result;
    const auto ret = s.find(ds, 3, q.data(), result);
    ASSERT_EQ(0, ret.code()) << "\n\nfind failed: " << ret.message() << "\n\n";
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(10u, result[0]);
    EXPECT_EQ(20u, result[1]);
    EXPECT_EQ(30u, result[2]);
}

TEST_F(ScannerTest, FindDatasetCosStoredDeltaHandlesZeroVectors) {
    std::string d = tmp_dir() + "/sketch2_utest_sc_cosdelta_zero_" + std::to_string(getpid());
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    DatasetNode ds;
    ASSERT_EQ(0, ds.init({d}, 100, DataType::f32, 4, kAccumulatorBufferSize, DistFunc::COS).code());
    write_input_raw(
        input_path_,
        "f32,4\n"
        "20 : [ 1.0, 0.0, 0.0, 0.0 ]\n"
        "21 : [ 2.0, 0.0, 0.0, 0.0 ]\n"
        "22 : [ 3.0, 0.0, 0.0, 0.0 ]\n"
        "23 : [ 4.0, 0.0, 0.0, 0.0 ]\n"
        "24 : [ 5.0, 0.0, 0.0, 0.0 ]\n");
    ASSERT_EQ(0, ds.store(input_path_).code());

    const auto zero = f32_values({0.0f, 0.0f, 0.0f, 0.0f});
    ASSERT_EQ(0, ds.add_vector(10, zero.data()).code());
    ASSERT_EQ(0, ds.store_accumulator().code());
    ASSERT_TRUE(fs::exists(d + "/0.delta"));

    Scanner s;
    std::vector<DistItem> result;
    ASSERT_EQ(0, s.find_items(ds, 2, zero.data(), result).code());
    ASSERT_EQ(2u, result.size());
    EXPECT_EQ(10u, result[0].id);
    EXPECT_DOUBLE_EQ(0.0, result[0].dist);
    EXPECT_EQ(20u, result[1].id);
    EXPECT_DOUBLE_EQ(1.0, result[1].dist);
}

TEST_F(ScannerTest, FindDatasetCosRejectsFilesMissingStoredInverseNorms) {
    std::string d = tmp_dir() + "/sketch2_utest_sc_cosds_legacy_" + std::to_string(getpid());
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 100.0, 1.0, 0.0, 0.0 ]\n"
        "20 : [ 1.0, 1.0, 0.0, 0.0 ]\n"
        "30 : [ -1.0, 0.0, 0.0, 0.0 ]\n");
    DataWriter writer;
    ASSERT_EQ(0, writer.init(input_path_, d + "/0.data").code());
    ASSERT_EQ(0, writer.exec().code());

    DatasetNode ds;
    ASSERT_EQ(0, ds.init({d}, 100, DataType::f32, 4, kAccumulatorBufferSize, DistFunc::COS).code());

    Scanner s;
    auto q = f32_values({1.0f, 0.0f, 0.0f, 0.0f});
    std::vector<uint64_t> result;
    const Ret ret = s.find(ds, 3, q.data(), result);
    EXPECT_NE(0, ret.code());
    EXPECT_TRUE(result.empty());
    EXPECT_NE(std::string(ret.message()).find("missing stored inverse norms"), std::string::npos);
}

TEST_F(ScannerTest, FindDatasetFailsOnNullQueryPointer) {
    std::string d = tmp_dir() + "/sketch2_utest_sc_dsnull_" + std::to_string(getpid());
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    DatasetNode ds;
    ASSERT_EQ(0, ds.init({d}, 100, DataType::f32, 4).code());
    generate_input_file(input_path_, GeneratorConfig{PatternType::Sequential, 3, 0, DataType::f32, 4, 1000});
    ASSERT_EQ(0, ds.store(input_path_).code());

    Scanner s;
    std::vector<uint64_t> result;
    EXPECT_NE(0, s.find(ds, 1, nullptr, result).code());
}

TEST_F(ScannerTest, FindDatasetFailsOnZeroCount) {
    std::string d = tmp_dir() + "/sketch2_utest_sc_dszero_" + std::to_string(getpid());
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    DatasetNode ds;
    ASSERT_EQ(0, ds.init({d}, 100, DataType::f32, 4).code());
    generate_input_file(input_path_, GeneratorConfig{PatternType::Sequential, 3, 0, DataType::f32, 4, 1000});
    ASSERT_EQ(0, ds.store(input_path_).code());

    Scanner s;
    auto q = f32_vec(1.0f, 4);
    std::vector<uint64_t> result;
    EXPECT_NE(0, s.find(ds, 0, q.data(), result).code());
}

TEST_F(ScannerTest, FindDatasetSkipsDeletedVectorsFromDelta) {
    std::string d = tmp_dir() + "/sketch2_utest_sc_dsdel_" + std::to_string(getpid());
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    DatasetNode ds;
    ASSERT_EQ(0, ds.init({d}, 100, DataType::f32, 4).code());

    // ids 0..4; values i+0.1 for each dimension
    generate_input_file(input_path_, GeneratorConfig{PatternType::Sequential, 5, 0, DataType::f32, 4, 1000});
    ASSERT_EQ(0, ds.store(input_path_).code());

    // Small update: delete id=2. With 5 existing and 1 incoming, ratio is below
    // the merge threshold, so a delta file is created.
    {
        std::ofstream f(input_path_);
        f << "f32,4\n2 : []\n";
    }
    ASSERT_EQ(0, ds.store(input_path_).code());
    ASSERT_TRUE(fs::exists(d + "/0.delta")) << "expected a delta file to exist";

    Scanner s;
    auto q = f32_vec(2.1f, 4); // query closest to deleted id=2
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(ds, 5, q.data(), result).code());

    // Only 4 vectors survive (0,1,3,4); id=2 must not appear.
    EXPECT_EQ(4u, result.size());
    for (uint64_t id : result) {
        EXPECT_NE(2u, id) << "deleted id=2 must not appear in results";
    }
}

TEST_F(ScannerTest, FindDatasetUsesUpdatedVectorFromDelta) {
    std::string d = tmp_dir() + "/sketch2_utest_sc_dsupd_" + std::to_string(getpid());
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    DatasetNode ds;
    ASSERT_EQ(0, ds.init({d}, 100, DataType::f32, 4).code());

    // ids 0..4; values i+0.1 for each dimension
    generate_input_file(input_path_, GeneratorConfig{PatternType::Sequential, 5, 0, DataType::f32, 4, 1000});
    ASSERT_EQ(0, ds.store(input_path_).code());

    // Small update: move id=1 far away. Should create delta.
    {
        std::ofstream f(input_path_);
        f << "f32,4\n1 : [ 500.0, 500.0, 500.0, 500.0 ]\n";
    }
    ASSERT_EQ(0, ds.store(input_path_).code());
    ASSERT_TRUE(fs::exists(d + "/0.delta")) << "expected a delta file to exist";

    Scanner s;
    // Query at 0.0: id=0 ([0.1,...]) is at distance 0.4, id=2 ([2.1,...]) is at 8.4.
    // Querying at 1.1 would be ambiguous in f32 arithmetic (2.1-1.1 < 1.1-0.1 due to
    // rounding), so we use 0.0 to make id=0 the unambiguous nearest neighbour.
    auto q = f32_vec(0.0f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(ds, 1, q.data(), result).code());
    ASSERT_EQ(1u, result.size());
    // id=0 (value 0.1, dist=0.4) beats all others; id=1 at 500 is not the nearest.
    EXPECT_EQ(0u, result[0]);
}

TEST_F(ScannerTest, FindDatasetDeleteFlushedFromAccumulatorStaysHidden) {
    std::string d = tmp_dir() + "/sketch2_utest_sc_dsaccflushdel_" + std::to_string(getpid());
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    DatasetNode ds;
    ASSERT_EQ(0, ds.init({d}, 100, DataType::f32, 4).code());

    generate_input_file(input_path_, GeneratorConfig{PatternType::Sequential, 5, 0, DataType::f32, 4, 1000});
    ASSERT_EQ(0, ds.store(input_path_).code());

    ASSERT_EQ(0, ds.delete_vector(2).code());
    ASSERT_EQ(0, ds.store_accumulator().code());
    ASSERT_TRUE(fs::exists(d + "/0.delta")) << "expected a delta file to exist";

    Scanner s;
    auto q = f32_vec(2.1f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(ds, 5, q.data(), result).code());

    EXPECT_EQ(4u, result.size());
    for (uint64_t id : result) {
        EXPECT_NE(2u, id);
    }
}

TEST_F(ScannerTest, FindDatasetUpdatedVectorAppearsOnlyOnceInResults) {
    std::string d = tmp_dir() + "/sketch2_utest_sc_dsaccdup_" + std::to_string(getpid());
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    DatasetNode ds;
    ASSERT_EQ(0, ds.init({d}, 100, DataType::f32, 4).code());

    write_input_raw(input_path_,
        "f32,4\n"
        "1 : [ 10.0, 10.0, 10.0, 10.0 ]\n"
        "2 : [ 20.0, 20.0, 20.0, 20.0 ]\n"
        "3 : [ 30.0, 30.0, 30.0, 30.0 ]\n");
    ASSERT_EQ(0, ds.store(input_path_).code());

    const auto updated = f32_vec(15.0f, 4);
    ASSERT_EQ(0, ds.add_vector(1, updated.data()).code());

    Scanner s;
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(ds, 3, updated.data(), result).code());
    ASSERT_EQ((std::vector<uint64_t> {1u, 2u, 3u}), result);
}

TEST_F(ScannerTest, FindDatasetSkipsPersistedDeltaVersionWhenAccumulatorUpdatesSameId) {
    std::string d = tmp_dir() + "/sketch2_utest_sc_dsaccdeltadup_" + std::to_string(getpid());
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    DatasetNode ds;
    ASSERT_EQ(0, ds.init({d}, 100, DataType::f32, 4).code());

    write_input_raw(input_path_,
        "f32,4\n"
        "10 : [ 10.0, 10.0, 10.0, 10.0 ]\n"
        "11 : [ 11.0, 11.0, 11.0, 11.0 ]\n"
        "12 : [ 12.0, 12.0, 12.0, 12.0 ]\n"
        "13 : [ 13.0, 13.0, 13.0, 13.0 ]\n");
    ASSERT_EQ(0, ds.store(input_path_).code());

    write_input_raw(input_path_,
        "f32,4\n"
        "12 : [ 100.0, 100.0, 100.0, 100.0 ]\n");
    ASSERT_EQ(0, ds.store(input_path_).code());
    ASSERT_TRUE(fs::exists(d + "/0.delta")) << "expected a delta file to exist";

    const auto updated = f32_vec(500.0f, 4);
    ASSERT_EQ(0, ds.add_vector(12, updated.data()).code());

    Scanner s;
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(ds, 5, updated.data(), result).code());
    ASSERT_EQ((std::vector<uint64_t> {12u, 13u, 11u, 10u}), result);
}

// ---------------------------------------------------------------------------
// Concurrent scan tests
//
// These install a live thread pool so the parallel reader path in
// scan_dataset_heap_custom is exercised. The parallel path activates when
// pool != nullptr && readers.size() >= 2. All tests use range_size=10 with
// enough vectors to produce at least three reader files.
// ---------------------------------------------------------------------------

// Installs a thread pool for the duration of each test, then restores whatever
// pool (including null) was in place before. This prevents cross-test
// interference when test order changes or when the process was started with a
// pre-configured pool (e.g. SKETCH2_THREAD_POOL_SIZE).
class ScannerConcurrentTest : public ScannerTest {
protected:
    void SetUp() override {
        ScannerTest::SetUp();
        prior_pool_ = get_singleton().thread_pool();
        Singleton::force_thread_pool_for_testing(4);
    }

    void TearDown() override {
        Singleton::force_thread_pool_for_testing(prior_pool_);
        ScannerTest::TearDown();
    }

    // Create a two-directory dataset with range_size=10. Storing 30 sequential
    // vectors (ids 0..29) produces three files across two dirs: 0.data in d0,
    // 1.data in d1, 2.data in d0. That guarantees readers.size() >= 2 so the
    // parallel code path is taken.
    void make_multi_reader_dataset(const std::string& d0, const std::string& d1,
            DatasetNode& ds, DataType type = DataType::f32, DistFunc func = DistFunc::L1) {
        fs::create_directories(d0);
        fs::create_directories(d1);
        ASSERT_EQ(0, ds.init({d0, d1}, 10, type, 4, kAccumulatorBufferSize, func).code());
        generate_input_file(input_path_,
            GeneratorConfig{PatternType::Sequential, 30, 0, type, 4, 1000});
        ASSERT_EQ(0, ds.store(input_path_).code());
    }

    std::string dir(const std::string& tag) {
        return tmp_dir() + "/sketch2_utest_sc_par_" + tag + "_" + std::to_string(getpid());
    }

private:
    std::shared_ptr<ThreadPool> prior_pool_;
};

// Query near the boundary between reader files (id 9 in file 0, id 10 in file 1)
// so the top-k merge step must combine results from two different workers.
TEST_F(ScannerConcurrentTest, L1TopKSpansMultipleReaders) {
    auto d0 = dir("l1_0"), d1 = dir("l1_1");
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d0); fs::remove_all(d1); });

    DatasetNode ds;
    make_multi_reader_dataset(d0, d1, ds);
    Scanner s;
    auto q = f32_vec(9.5f, 4); // equidistant from id=9 and id=10
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(ds, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    // id=9 (dist=0.1*4*|9-9.5|=2) and id=10 (same dist) — tie broken by lower id first.
    EXPECT_EQ(9u,  result[0]);
    EXPECT_EQ(10u, result[1]);
    // id=8 or id=11 both at distance 6; tie broken by lower id.
    EXPECT_EQ(8u,  result[2]);
}

TEST_F(ScannerConcurrentTest, L2TopKSpansMultipleReaders) {
    auto d0 = dir("l2_0"), d1 = dir("l2_1");
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d0); fs::remove_all(d1); });

    DatasetNode ds;
    make_multi_reader_dataset(d0, d1, ds, DataType::f32, DistFunc::L2);
    Scanner s;
    auto q = f32_vec(9.5f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(ds, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(9u,  result[0]);
    EXPECT_EQ(10u, result[1]);
    EXPECT_EQ(8u,  result[2]);
}

// find_items parallel path returns correct ids and distances.
TEST_F(ScannerConcurrentTest, FindItemsSpansMultipleReaders) {
    auto d0 = dir("items_0"), d1 = dir("items_1");
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d0); fs::remove_all(d1); });

    DatasetNode ds;
    make_multi_reader_dataset(d0, d1, ds);
    Scanner s;
    auto q = f32_vec(15.2f, 4);
    std::vector<DistItem> result;
    ASSERT_EQ(0, s.find_items(ds, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(15u, result[0].id);
    EXPECT_EQ(16u, result[1].id);
    EXPECT_EQ(14u, result[2].id);
    EXPECT_NEAR(0.4, result[0].dist, 1e-5);
    EXPECT_NEAR(3.6, result[1].dist, 1e-5);
    EXPECT_NEAR(4.4, result[2].dist, 1e-5);
}

// With a pool installed but only one reader, scan_dataset_heap_custom must
// fall back to sequential and still return correct results.
TEST_F(ScannerConcurrentTest, SingleReaderWithPoolFallsBackToSequential) {
    auto d = dir("single");
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    // range_size=1000 keeps all 5 vectors in a single file.
    DatasetNode ds;
    ASSERT_EQ(0, ds.init({d}, 1000, DataType::f32, 4).code());
    generate_input_file(input_path_,
        GeneratorConfig{PatternType::Sequential, 5, 0, DataType::f32, 4, 1000});
    ASSERT_EQ(0, ds.store(input_path_).code());

    Scanner s;
    auto q = f32_vec(2.2f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(ds, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(2u, result[0]);
    EXPECT_EQ(3u, result[1]);
    EXPECT_EQ(1u, result[2]);
}

// Cosine scan across multiple readers with the pool active. Each reader file
// may or may not have stored inverse norms; this tests the inv_score /
// query_score branch selection under the parallel path.
TEST_F(ScannerConcurrentTest, CosineTopKSpansMultipleReaders) {
    auto d0 = dir("cos_0"), d1 = dir("cos_1");
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d0); fs::remove_all(d1); });

    fs::create_directories(d0);
    fs::create_directories(d1);
    DatasetNode ds;
    ASSERT_EQ(0, ds.init({d0, d1}, 10, DataType::f32, 4,
        kAccumulatorBufferSize, DistFunc::COS).code());

    // Three vectors in different id ranges (different reader files).
    write_input_raw(input_path_,
        "f32,4\n"
        "5  : [ 100.0, 1.0, 0.0, 0.0 ]\n"
        "15 : [ 1.0,   1.0, 0.0, 0.0 ]\n"
        "25 : [ -1.0,  0.0, 0.0, 0.0 ]\n");
    ASSERT_EQ(0, ds.store(input_path_).code());

    Scanner s;
    auto q = f32_values({1.0f, 0.0f, 0.0f, 0.0f});
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(ds, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(5u,  result[0]);
    EXPECT_EQ(15u, result[1]);
    EXPECT_EQ(25u, result[2]);
}
