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
#include "core/storage/input_generator.h"
#include "core/storage/data_writer.h"
#include "core/storage/data_reader.h"
#include "core/storage/dataset.h"

using namespace sketch2;
namespace fs = std::filesystem;

class ScannerTest : public ::testing::Test {
protected:
    std::string input_path_;
    std::string data_path_;
    std::string delta_input_path_;
    std::string delta_path_;

    void SetUp() override {
        std::string base = "/tmp/sketch2_utest_sc_" + std::to_string(getpid());
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

    std::vector<uint8_t> f32_vec(float val, size_t dim) {
        std::vector<uint8_t> buf(dim * sizeof(float));
        auto* p = reinterpret_cast<float*>(buf.data());
        for (size_t i = 0; i < dim; ++i) p[i] = val;
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
    DataReader reader;
    ASSERT_EQ(0, reader.init(data_path_).code());
    Scanner s;
    auto q = f32_vec(0.0f, 4);
    std::vector<uint64_t> result;
    EXPECT_NE(0, s.find(reader, DistFunc::L1, 0, q.data(), result).code());
}

TEST_F(ScannerTest, FindFailsOnNullQueryPointer) {
    generate(3, 0, DataType::f32, 4);
    DataReader reader;
    ASSERT_EQ(0, reader.init(data_path_).code());
    Scanner s;
    std::vector<uint64_t> result;
    EXPECT_NE(0, s.find(reader, DistFunc::L1, 1, nullptr, result).code());
}

TEST_F(ScannerTest, FindFailsOnUnknownFunction) {
    generate(3, 0, DataType::f32, 4);
    DataReader reader;
    ASSERT_EQ(0, reader.init(data_path_).code());
    Scanner s;
    auto q = f32_vec(0.0f, 4);
    std::vector<uint64_t> result;
    EXPECT_NE(0, s.find(reader, static_cast<DistFunc>(999), 1, q.data(), result).code());
}

TEST_F(ScannerTest, FindCountExceedsTotalReturnsCapped) {
    const size_t total = 3;
    generate(total, 0, DataType::f32, 4);
    DataReader reader;
    ASSERT_EQ(0, reader.init(data_path_).code());
    Scanner s;
    auto q = f32_vec(0.0f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(reader, DistFunc::L1, 100, q.data(), result).code());
    EXPECT_EQ(total, result.size());
}

TEST_F(ScannerTest, FindResultSizeMatchesRequest) {
    generate(5, 0, DataType::f32, 4);
    DataReader reader;
    ASSERT_EQ(0, reader.init(data_path_).code());
    Scanner s;
    auto q = f32_vec(3.2f, 4);
    std::vector<uint64_t> result;

    ASSERT_EQ(0, s.find(reader, DistFunc::L1, 1, q.data(), result).code());
    EXPECT_EQ(1u, result.size());

    ASSERT_EQ(0, s.find(reader, DistFunc::L1, 3, q.data(), result).code());
    EXPECT_EQ(3u, result.size());

    ASSERT_EQ(0, s.find(reader, DistFunc::L1, 5, q.data(), result).code());
    EXPECT_EQ(5u, result.size());
}

TEST_F(ScannerTest, FindF32K3ReturnsInOrder) {
    generate(5, 0, DataType::f32, 4);
    DataReader reader;
    ASSERT_EQ(0, reader.init(data_path_).code());
    Scanner s;
    auto q = f32_vec(3.2f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(reader, DistFunc::L1, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(3u, result[0]);
    EXPECT_EQ(4u, result[1]);
    EXPECT_EQ(2u, result[2]);
}

TEST_F(ScannerTest, FindF32L2K3ReturnsInOrder) {
    generate(5, 0, DataType::f32, 4);
    DataReader reader;
    ASSERT_EQ(0, reader.init(data_path_).code());
    Scanner s;
    auto q = f32_vec(3.2f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(reader, DistFunc::L2, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(3u, result[0]);
    EXPECT_EQ(4u, result[1]);
    EXPECT_EQ(2u, result[2]);
}

TEST_F(ScannerTest, FindI16AllSortedByDistance) {
    generate(3, 0, DataType::i16, 4);
    DataReader reader;
    ASSERT_EQ(0, reader.init(data_path_).code());
    Scanner s;
    auto q = i16_vec(0, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(reader, DistFunc::L1, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(0u, result[0]);
    EXPECT_EQ(1u, result[1]);
    EXPECT_EQ(2u, result[2]);
}

TEST_F(ScannerTest, FindF16Works) {
    if (!supports_f16()) {
        return;
    }
    generate(3, 0, DataType::f16, 4);
    DataReader reader;
    ASSERT_EQ(0, reader.init(data_path_).code());
    Scanner s;
    auto q = f16_vec(1.1f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(reader, DistFunc::L1, 1, q.data(), result).code());
    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(1u, result[0]);
}

TEST_F(ScannerTest, DeltaSkipsDeletedIds) {
    generate(6, 0, DataType::f32, 4);
    generate_delta(6, 0, DataType::f32, 4, 2); // deleted ids: 2,4

    auto delta = std::make_unique<DataReader>();
    ASSERT_EQ(0, delta->init(delta_path_).code());
    DataReader reader;
    ASSERT_EQ(0, reader.init(data_path_, std::move(delta)).code());

    Scanner s;
    auto q = f32_vec(3.2f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(reader, DistFunc::L1, 6, q.data(), result).code());

    for (uint64_t id : result) {
        EXPECT_NE(2u, id);
        EXPECT_NE(4u, id);
    }
}

TEST_F(ScannerTest, DeltaUsesUpdatedVectors) {
    generate(4, 10, DataType::f32, 4); // values 10.1, 11.1, 12.1, 13.1
    write_delta_raw(
        "f32,4\n"
        "11 : [ 20.0, 20.0, 20.0, 20.0 ]\n");

    auto delta = std::make_unique<DataReader>();
    ASSERT_EQ(0, delta->init(delta_path_).code());
    DataReader reader;
    ASSERT_EQ(0, reader.init(data_path_, std::move(delta)).code());

    Scanner s;
    auto q = f32_vec(20.0f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(reader, DistFunc::L1, 1, q.data(), result).code());
    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(11u, result[0]);
}

TEST_F(ScannerTest, FindDatasetWorks) {
    std::string d0 = "/tmp/sketch2_utest_sc_ds0_" + std::to_string(getpid());
    std::string d1 = "/tmp/sketch2_utest_sc_ds1_" + std::to_string(getpid());
    fs::create_directories(d0);
    fs::create_directories(d1);
    std::experimental::scope_exit cleanup([&]() {
        fs::remove_all(d0);
        fs::remove_all(d1);
    });

    Dataset ds;
    ASSERT_EQ(0, ds.init({d0, d1}, 10, DataType::f32, 4).code());

    // Generate 30 items: 0..29.
    // They go into 0.data (0..9), 1.data (10..19), 2.data (20..29).
    generate_input_file(input_path_, GeneratorConfig{PatternType::Sequential, 30, 0, DataType::f32, 4, 1000});
    ASSERT_EQ(0, ds.store(input_path_).code());

    Scanner s;
    auto q = f32_vec(15.2f, 4); // nearest to id 15
    std::vector<uint64_t> result;
    const auto ret = s.find(ds, DistFunc::L1, 3, q.data(), result);
    ASSERT_EQ(0, ret.code()) << "\n\nfind failed: " << ret.message() << "\n\n";
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(15u, result[0]);
    EXPECT_EQ(16u, result[1]);
    EXPECT_EQ(14u, result[2]);
}

TEST_F(ScannerTest, FindDatasetL2Works) {
    std::string d0 = "/tmp/sketch2_utest_sc_l2ds0_" + std::to_string(getpid());
    std::string d1 = "/tmp/sketch2_utest_sc_l2ds1_" + std::to_string(getpid());
    fs::create_directories(d0);
    fs::create_directories(d1);
    std::experimental::scope_exit cleanup([&]() {
        fs::remove_all(d0);
        fs::remove_all(d1);
    });

    Dataset ds;
    ASSERT_EQ(0, ds.init({d0, d1}, 10, DataType::f32, 4).code());
    generate_input_file(input_path_, GeneratorConfig{PatternType::Sequential, 30, 0, DataType::f32, 4, 1000});
    ASSERT_EQ(0, ds.store(input_path_).code());

    Scanner s;
    auto q = f32_vec(15.2f, 4);
    std::vector<uint64_t> result;
    const auto ret = s.find(ds, DistFunc::L2, 3, q.data(), result);
    ASSERT_EQ(0, ret.code()) << "\n\nfind failed: " << ret.message() << "\n\n";
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(15u, result[0]);
    EXPECT_EQ(16u, result[1]);
    EXPECT_EQ(14u, result[2]);
}

TEST_F(ScannerTest, FindDatasetFailsOnNullQueryPointer) {
    std::string d = "/tmp/sketch2_utest_sc_dsnull_" + std::to_string(getpid());
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    Dataset ds;
    ASSERT_EQ(0, ds.init({d}, 100, DataType::f32, 4).code());
    generate_input_file(input_path_, GeneratorConfig{PatternType::Sequential, 3, 0, DataType::f32, 4, 1000});
    ASSERT_EQ(0, ds.store(input_path_).code());

    Scanner s;
    std::vector<uint64_t> result;
    EXPECT_NE(0, s.find(ds, DistFunc::L1, 1, nullptr, result).code());
}

TEST_F(ScannerTest, FindDatasetFailsOnZeroCount) {
    std::string d = "/tmp/sketch2_utest_sc_dszero_" + std::to_string(getpid());
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    Dataset ds;
    ASSERT_EQ(0, ds.init({d}, 100, DataType::f32, 4).code());
    generate_input_file(input_path_, GeneratorConfig{PatternType::Sequential, 3, 0, DataType::f32, 4, 1000});
    ASSERT_EQ(0, ds.store(input_path_).code());

    Scanner s;
    auto q = f32_vec(1.0f, 4);
    std::vector<uint64_t> result;
    EXPECT_NE(0, s.find(ds, DistFunc::L1, 0, q.data(), result).code());
}

TEST_F(ScannerTest, FindDatasetFailsOnUnknownFunction) {
    std::string d = "/tmp/sketch2_utest_sc_dsfunc_" + std::to_string(getpid());
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    Dataset ds;
    ASSERT_EQ(0, ds.init({d}, 100, DataType::f32, 4).code());
    generate_input_file(input_path_, GeneratorConfig{PatternType::Sequential, 3, 0, DataType::f32, 4, 1000});
    ASSERT_EQ(0, ds.store(input_path_).code());

    Scanner s;
    auto q = f32_vec(1.0f, 4);
    std::vector<uint64_t> result;
    EXPECT_NE(0, s.find(ds, static_cast<DistFunc>(999), 1, q.data(), result).code());
}

TEST_F(ScannerTest, FindDatasetSkipsDeletedVectorsFromDelta) {
    std::string d = "/tmp/sketch2_utest_sc_dsdel_" + std::to_string(getpid());
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    Dataset ds;
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
    ASSERT_EQ(0, s.find(ds, DistFunc::L1, 5, q.data(), result).code());

    // Only 4 vectors survive (0,1,3,4); id=2 must not appear.
    EXPECT_EQ(4u, result.size());
    for (uint64_t id : result) {
        EXPECT_NE(2u, id) << "deleted id=2 must not appear in results";
    }
}

TEST_F(ScannerTest, FindDatasetUsesUpdatedVectorFromDelta) {
    std::string d = "/tmp/sketch2_utest_sc_dsupd_" + std::to_string(getpid());
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    Dataset ds;
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
    ASSERT_EQ(0, s.find(ds, DistFunc::L1, 1, q.data(), result).code());
    ASSERT_EQ(1u, result.size());
    // id=0 (value 0.1, dist=0.4) beats all others; id=1 at 500 is not the nearest.
    EXPECT_EQ(0u, result[0]);
}

TEST_F(ScannerTest, FindDatasetSkipsIdsDeletedInAccumulator) {
    std::string d = "/tmp/sketch2_utest_sc_dsaccdel_" + std::to_string(getpid());
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    Dataset ds;
    ASSERT_EQ(0, ds.init({d}, 100, DataType::f32, 4).code());

    generate_input_file(input_path_, GeneratorConfig{PatternType::Sequential, 5, 0, DataType::f32, 4, 1000});
    ASSERT_EQ(0, ds.store(input_path_).code());

    ASSERT_EQ(0, ds.delete_vector(2).code());
    EXPECT_TRUE(ds.is_deleted(2));

    Scanner s;
    auto q = f32_vec(2.1f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(ds, DistFunc::L1, 5, q.data(), result).code());

    EXPECT_EQ(4u, result.size());
    for (uint64_t id : result) {
        EXPECT_NE(2u, id);
    }
}

TEST_F(ScannerTest, FindDatasetIncludesVectorsFromAccumulator) {
    std::string d = "/tmp/sketch2_utest_sc_dsaccadd_" + std::to_string(getpid());
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    Dataset ds;
    ASSERT_EQ(0, ds.init({d}, 100, DataType::f32, 4).code());

    generate_input_file(input_path_, GeneratorConfig{PatternType::Sequential, 5, 0, DataType::f32, 4, 1000});
    ASSERT_EQ(0, ds.store(input_path_).code());

    const auto pending = f32_vec(1000.0f, 4);
    ASSERT_EQ(0, ds.add_vector(50, pending.data()).code());

    Scanner s;
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(ds, DistFunc::L1, 1, pending.data(), result).code());
    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(50u, result[0]);
}

TEST_F(ScannerTest, FindDatasetUsesUpdatedVectorFromAccumulator) {
    std::string d = "/tmp/sketch2_utest_sc_dsaccupd_" + std::to_string(getpid());
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    Dataset ds;
    ASSERT_EQ(0, ds.init({d}, 100, DataType::f32, 4).code());

    generate_input_file(input_path_, GeneratorConfig{PatternType::Sequential, 5, 0, DataType::f32, 4, 1000});
    ASSERT_EQ(0, ds.store(input_path_).code());

    const auto updated = f32_vec(500.0f, 4);
    ASSERT_EQ(0, ds.add_vector(1, updated.data()).code());

    Scanner s;
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(ds, DistFunc::L1, 1, updated.data(), result).code());
    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(1u, result[0]);
}
