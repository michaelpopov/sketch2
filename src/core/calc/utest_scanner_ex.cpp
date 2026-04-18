// Unit tests for ScannerEx nearest-neighbor scanning.

#include <gtest/gtest.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <fstream>
#include <memory>
#include <filesystem>
#include <experimental/scope>
#include "core/calc/scanner_ex.h"
#include "core/utils/singleton.h"
#include "core/utils/thread_pool.h"
#include "core/storage/input_generator.h"
#include "core/storage/data_writer.h"
#include "core/storage/data_reader.h"
#include "core/storage/dataset_node.h"
#include "utest_tmp_dir.h"

using namespace sketch2;
namespace fs = std::filesystem;

namespace {

constexpr CalcEngine kCompiledEngine = compiled_calc_engine();

ScannerEx make_compiled_scanner() {
    return ScannerEx(kCompiledEngine);
}

}

class ScannerExTest : public ::testing::Test {
protected:
    std::string input_path_;
    std::string data_path_;
    std::string delta_input_path_;
    std::string delta_path_;
    std::vector<std::string> cleanup_dirs_;
    std::vector<std::string> cleanup_files_;

    void SetUp() override {
        std::string base = tmp_dir() + "/sketch2_utest_scanner_ex_" + std::to_string(getpid());
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
        ASSERT_EQ(0, w.exec_for_testing(in_path, out_path).code());
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
        ASSERT_EQ(0, w.exec_for_testing(delta_input_path_, delta_path_).code());
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
        EXPECT_EQ(0, ds.init_for_test({dataset_dir}, range_size, type, dim, func).code());
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

    void overwrite_stored_norm(const std::string& path, size_t index, float value) {
        const int fd = open(path.c_str(), O_RDWR);
        ASSERT_GE(fd, 0);

        DataFileHeader hdr{};
        ASSERT_EQ(static_cast<ssize_t>(sizeof(hdr)), pread(fd, &hdr, sizeof(hdr), 0));
        ASSERT_TRUE(data_file_has_norms(hdr));
        ASSERT_LT(index, static_cast<size_t>(hdr.count));

        const off_t norm_offset =
            static_cast<off_t>(hdr.norms_offset + index * sizeof(float));
        ASSERT_EQ(static_cast<ssize_t>(sizeof(value)),
                  pwrite(fd, &value, sizeof(value), norm_offset));
        ASSERT_EQ(0, close(fd));
    }
};

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

TEST_F(ScannerExTest, FindFailsOnCountZero) {
    generate(3, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_});
    ScannerEx s;
    auto q = f32_vec(0.0f, 4);
    std::vector<uint64_t> result;
    EXPECT_NE(0, s.find(*reader, 0, q.data(), result).code());
}

TEST_F(ScannerExTest, FindFailsOnNullQueryPointer) {
    generate(3, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_});
    ScannerEx s;
    std::vector<uint64_t> result;
    EXPECT_NE(0, s.find(*reader, 1, nullptr, result).code());
}

TEST_F(ScannerExTest, FindFailsOnUnknownFunction) {
    DatasetReader reader;
    ScannerEx s = make_compiled_scanner();
    auto q = f32_vec(0.0f, 4);
    std::vector<uint64_t> result;
    EXPECT_NE(0, s.find(reader, 1, q.data(), result).code());
}

// ---------------------------------------------------------------------------
// DOT metric
// ---------------------------------------------------------------------------

TEST_F(ScannerExTest, FindF32DOTK3ReturnsInOrder) {
    generate(5, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_});
    ScannerEx s;
    auto q = f32_vec(3.2f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(4u, result[0]);
    EXPECT_EQ(3u, result[1]);
    EXPECT_EQ(2u, result[2]);
}

#if SKETCH_CALC_ENGINE_HIGHWAY
TEST_F(ScannerExTest, FindF32DOTK3ReturnsInOrderWithHighway) {
    generate(5, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_});
    ScannerEx s = make_compiled_scanner();
    auto q = f32_vec(3.2f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(4u, result[0]);
    EXPECT_EQ(3u, result[1]);
    EXPECT_EQ(2u, result[2]);
}
#endif

#if SKETCH_CALC_ENGINE_NUMKONG
TEST_F(ScannerExTest, FindF32DOTK3ReturnsInOrderWithNumKong) {
    generate(5, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_});
    ScannerEx s(CalcEngine::numkong);
    auto q = f32_vec(3.2f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(4u, result[0]);
    EXPECT_EQ(3u, result[1]);
    EXPECT_EQ(2u, result[2]);
}
#endif

TEST_F(ScannerExTest, FindCountExceedsTotalReturnsCapped) {
    const size_t total = 3;
    generate(total, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_});
    ScannerEx s;
    auto q = f32_vec(0.0f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 100, q.data(), result).code());
    EXPECT_EQ(total, result.size());
}

TEST_F(ScannerExTest, FindResultSizeMatchesRequest) {
    generate(5, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_});
    ScannerEx s;
    auto q = f32_vec(3.2f, 4);
    std::vector<uint64_t> result;

    ASSERT_EQ(0, s.find(*reader, 1, q.data(), result).code());
    EXPECT_EQ(1u, result.size());

    ASSERT_EQ(0, s.find(*reader, 3, q.data(), result).code());
    EXPECT_EQ(3u, result.size());

    ASSERT_EQ(0, s.find(*reader, 5, q.data(), result).code());
    EXPECT_EQ(5u, result.size());
}

TEST_F(ScannerExTest, FindItemsF32DOTReturnsIdsAndDistances) {
    generate(5, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_});
    ScannerEx s;
    auto q = f32_vec(3.2f, 4);
    std::vector<DistItem> result;
    ASSERT_EQ(0, s.find_items(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(4u, result[0].id);
    EXPECT_EQ(3u, result[1].id);
    EXPECT_EQ(2u, result[2].id);
    EXPECT_NEAR(52.48, result[0].score, 1e-4);
    EXPECT_NEAR(39.68, result[1].score, 1e-4);
    EXPECT_NEAR(26.88, result[2].score, 1e-4);
}

// ---------------------------------------------------------------------------
// L2 metric
// ---------------------------------------------------------------------------

TEST_F(ScannerExTest, FindF32L2K3ReturnsInOrder) {
    generate(5, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::L2, {input_path_});
    ScannerEx s;
    auto q = f32_vec(3.2f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(3u, result[0]);
    EXPECT_EQ(4u, result[1]);
    EXPECT_EQ(2u, result[2]);
}

#if SKETCH_CALC_ENGINE_NUMKONG
TEST_F(ScannerExTest, FindF32L2K3ReturnsInOrderWithNumKong) {
    generate(5, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::L2, {input_path_});
    ScannerEx s(CalcEngine::numkong);
    auto q = f32_vec(3.2f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(3u, result[0]);
    EXPECT_EQ(4u, result[1]);
    EXPECT_EQ(2u, result[2]);
}
#endif

#if SKETCH_CALC_ENGINE_HIGHWAY
TEST_F(ScannerExTest, FindF32L2K3ReturnsInOrderWithHighway) {
    generate(5, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::L2, {input_path_});
    ScannerEx s = make_compiled_scanner();
    auto q = f32_vec(3.2f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(3u, result[0]);
    EXPECT_EQ(4u, result[1]);
    EXPECT_EQ(2u, result[2]);
}
#endif

// ---------------------------------------------------------------------------
// Cosine metric
// ---------------------------------------------------------------------------

TEST_F(ScannerExTest, FindF32CosK3ReturnsInOrder) {
    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 100.0, 1.0, 0.0, 0.0 ]\n"
        "20 : [ 1.0, 1.0, 0.0, 0.0 ]\n"
        "30 : [ -1.0, 0.0, 0.0, 0.0 ]\n");
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::COS, {input_path_});
    ScannerEx s;
    auto q = f32_values({1.0f, 0.0f, 0.0f, 0.0f});
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(10u, result[0]);
    EXPECT_EQ(20u, result[1]);
    EXPECT_EQ(30u, result[2]);
}

#if SKETCH_CALC_ENGINE_NUMKONG
TEST_F(ScannerExTest, FindF32CosK3ReturnsInOrderWithNumKong) {
    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 100.0, 1.0, 0.0, 0.0 ]\n"
        "20 : [ 1.0, 1.0, 0.0, 0.0 ]\n"
        "30 : [ -1.0, 0.0, 0.0, 0.0 ]\n");
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::COS, {input_path_});
    ScannerEx s(CalcEngine::numkong);
    auto q = f32_values({1.0f, 0.0f, 0.0f, 0.0f});
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(10u, result[0]);
    EXPECT_EQ(20u, result[1]);
    EXPECT_EQ(30u, result[2]);
}
#endif

#if SKETCH_CALC_ENGINE_HIGHWAY
TEST_F(ScannerExTest, FindF32CosK3ReturnsInOrderWithHighway) {
    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 100.0, 1.0, 0.0, 0.0 ]\n"
        "20 : [ 1.0, 1.0, 0.0, 0.0 ]\n"
        "30 : [ -1.0, 0.0, 0.0, 0.0 ]\n");
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::COS, {input_path_});
    ScannerEx s = make_compiled_scanner();
    auto q = f32_values({1.0f, 0.0f, 0.0f, 0.0f});
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(10u, result[0]);
    EXPECT_EQ(20u, result[1]);
    EXPECT_EQ(30u, result[2]);
}
#endif

TEST_F(ScannerExTest, FindF32CosStoredCosineValuesHandleZeroVectors) {
    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 0.0, 0.0, 0.0, 0.0 ]\n"
        "20 : [ 1.0, 0.0, 0.0, 0.0 ]\n");
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::COS, {input_path_});

    ScannerEx s = make_compiled_scanner();
    auto q = f32_values({0.0f, 0.0f, 0.0f, 0.0f});
    std::vector<DistItem> result;
    ASSERT_EQ(0, s.find_items(*reader, 2, q.data(), result).code());
    ASSERT_EQ(2u, result.size());
    EXPECT_EQ(10u, result[0].id);
    EXPECT_DOUBLE_EQ(0.0, result[0].score);
    EXPECT_EQ(20u, result[1].id);
    EXPECT_DOUBLE_EQ(1.0, result[1].score);
}

TEST_F(ScannerExTest, FindF32CosStoredPathsMatchRanking) {
    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 10.0, 0.0, 0.0, 0.0 ]\n"
        "20 : [ 2.0, 1.0, 0.0, 0.0 ]\n"
        "30 : [ 1.0, 2.0, 0.0, 0.0 ]\n"
        "40 : [ 0.0, 1.0, 0.0, 0.0 ]\n"
        "50 : [ -1.0, 0.0, 0.0, 0.0 ]\n");

    auto reader_a = make_dataset_reader(DataType::f32, 4, DistFunc::COS, {input_path_});
    auto reader_b = make_dataset_reader(DataType::f32, 4, DistFunc::COS, {input_path_});

    ScannerEx s = make_compiled_scanner();
    auto q = f32_values({1.0f, 0.0f, 0.0f, 0.0f});
    std::vector<uint64_t> result_a;
    std::vector<uint64_t> result_b;
    ASSERT_EQ(0, s.find(*reader_a, 5, q.data(), result_a).code());
    ASSERT_EQ(0, s.find(*reader_b, 5, q.data(), result_b).code());

    ASSERT_EQ((std::vector<uint64_t> {10u, 20u, 30u, 40u, 50u}), result_a);
    EXPECT_EQ(result_a, result_b);
}

// ---------------------------------------------------------------------------
// Other data types
// ---------------------------------------------------------------------------

TEST_F(ScannerExTest, FindI16AllSortedByDistance) {
#if SKETCH_CALC_ENGINE_HIGHWAY
    generate(3, 0, DataType::i16, 4);
    auto reader = make_dataset_reader(DataType::i16, 4, DistFunc::DOT, {input_path_});
    ScannerEx s;
    auto q = i16_vec(0, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(0u, result[0]);
    EXPECT_EQ(1u, result[1]);
    EXPECT_EQ(2u, result[2]);
#else
    GTEST_SKIP() << "i16 is not supported in NumKong builds";
#endif
}

TEST_F(ScannerExTest, FindI16RejectsNumKong) {
#if SKETCH_CALC_ENGINE_NUMKONG
    generate(3, 0, DataType::i16, 4);
    auto reader = make_dataset_reader(DataType::i16, 4, DistFunc::DOT, {input_path_});
    ScannerEx s(CalcEngine::numkong);
    auto q = i16_vec(0, 4);
    std::vector<uint64_t> result;
    const Ret ret = s.find(*reader, 3, q.data(), result);
    EXPECT_NE(0, ret.code());
    EXPECT_TRUE(result.empty());
    EXPECT_NE(std::string(ret.message()).find("NumKong does not support i16"),
              std::string::npos);
#else
    GTEST_SKIP() << "NumKong-only rejection is only relevant in NumKong builds";
#endif
}

TEST_F(ScannerExTest, FindF16Works) {
    generate(3, 0, DataType::f16, 4);
    auto reader = make_dataset_reader(DataType::f16, 4, DistFunc::DOT, {input_path_});
    ScannerEx s;
    auto q = f16_vec(1.1f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 1, q.data(), result).code());
    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(2u, result[0]);
}

#if SKETCH_CALC_ENGINE_HIGHWAY
TEST_F(ScannerExTest, FindF16WorksWithHighway) {
    generate(3, 0, DataType::f16, 4);
    auto reader = make_dataset_reader(DataType::f16, 4, DistFunc::DOT, {input_path_});
    ScannerEx s = make_compiled_scanner();
    auto q = f16_vec(1.1f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 1, q.data(), result).code());
    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(2u, result[0]);
}
#endif

#if SKETCH_CALC_ENGINE_NUMKONG
TEST_F(ScannerExTest, FindF16CosWorksWithNumKong) {
    write_input_raw(
        input_path_,
        "f16,4\n"
        "10 : [ 100.0, 1.0, 0.0, 0.0 ]\n"
        "20 : [ 1.0, 1.0, 0.0, 0.0 ]\n"
        "30 : [ -1.0, 0.0, 0.0, 0.0 ]\n");
    auto reader = make_dataset_reader(DataType::f16, 4, DistFunc::COS, {input_path_});
    ScannerEx s(CalcEngine::numkong);
    auto q = f16_vec(0.0f, 4);
    reinterpret_cast<uint16_t*>(q.data())[0] = float_to_f16(1.0f);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(10u, result[0]);
    EXPECT_EQ(20u, result[1]);
    EXPECT_EQ(30u, result[2]);
}
#endif

#if SKETCH_CALC_ENGINE_HIGHWAY
TEST_F(ScannerExTest, FindF16CosWorksWithHighway) {
    write_input_raw(
        input_path_,
        "f16,4\n"
        "10 : [ 100.0, 1.0, 0.0, 0.0 ]\n"
        "20 : [ 1.0, 1.0, 0.0, 0.0 ]\n"
        "30 : [ -1.0, 0.0, 0.0, 0.0 ]\n");
    auto reader = make_dataset_reader(DataType::f16, 4, DistFunc::COS, {input_path_});
    ScannerEx s = make_compiled_scanner();
    auto q = f16_vec(0.0f, 4);
    reinterpret_cast<uint16_t*>(q.data())[0] = float_to_f16(1.0f);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(10u, result[0]);
    EXPECT_EQ(20u, result[1]);
    EXPECT_EQ(30u, result[2]);
}
#endif

// ---------------------------------------------------------------------------
// Delta tests
// ---------------------------------------------------------------------------

TEST_F(ScannerExTest, DeltaSkipsDeletedIds) {
    generate(6, 0, DataType::f32, 4);
    generate_delta(6, 0, DataType::f32, 4, 2);

    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_, delta_input_path_});

    ScannerEx s;
    auto q = f32_vec(3.2f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 6, q.data(), result).code());

    for (uint64_t id : result) {
        EXPECT_NE(2u, id);
        EXPECT_NE(4u, id);
    }
}

TEST_F(ScannerExTest, DeltaUsesUpdatedVectors) {
    generate(4, 10, DataType::f32, 4);
    write_delta_raw(
        "f32,4\n"
        "11 : [ 20.0, 20.0, 20.0, 20.0 ]\n");

    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_, delta_input_path_});

    ScannerEx s;
    auto q = f32_vec(20.0f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 1, q.data(), result).code());
    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(11u, result[0]);
}

TEST_F(ScannerExTest, DeltaDeletingAllVectorsReturnsEmptyResult) {
    generate(3, 0, DataType::f32, 4);
    write_delta_raw(
        "f32,4\n"
        "0 : []\n"
        "1 : []\n"
        "2 : []\n");

    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_, delta_input_path_});

    ScannerEx s = make_compiled_scanner();
    auto q = f32_vec(1.1f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 3, q.data(), result).code());
    EXPECT_TRUE(result.empty());
}

// ---------------------------------------------------------------------------
// Multi-file dataset
// ---------------------------------------------------------------------------

TEST_F(ScannerExTest, FindDatasetWorks) {
    generate_input_file(input_path_, GeneratorConfig{PatternType::Sequential, 30, 0, DataType::f32, 4, 1000});
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_}, 10);

    ScannerEx s;
    auto q = f32_vec(15.2f, 4);
    std::vector<uint64_t> result;
    const auto ret = s.find(*reader, 3, q.data(), result);
    ASSERT_EQ(0, ret.code()) << "\n\nfind failed: " << ret.message() << "\n\n";
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(29u, result[0]);
    EXPECT_EQ(28u, result[1]);
    EXPECT_EQ(27u, result[2]);
}

TEST_F(ScannerExTest, FindDatasetItemsReturnsIdsAndDistancesInOrder) {
    std::string d = tmp_dir() + "/sketch2_utest_scanner_ex_dsitems_" + std::to_string(getpid());
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    auto input = d + "/input.txt";
    generate_input_file(input, GeneratorConfig{PatternType::Sequential, 30, 0, DataType::f32, 4, 1000});
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input}, 100);

    ScannerEx s = make_compiled_scanner();
    auto q = f32_vec(15.2f, 4);
    std::vector<DistItem> result;
    ASSERT_EQ(0, s.find_items(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(29u, result[0].id);
    EXPECT_EQ(28u, result[1].id);
    EXPECT_EQ(27u, result[2].id);
    EXPECT_NEAR(1769.28, result[0].score, 1e-2);
    EXPECT_NEAR(1708.48, result[1].score, 1e-2);
    EXPECT_NEAR(1647.68, result[2].score, 1e-2);
}

TEST_F(ScannerExTest, FindDatasetL2Works) {
    generate_input_file(input_path_, GeneratorConfig{PatternType::Sequential, 30, 0, DataType::f32, 4, 1000});
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::L2, {input_path_}, 10);

    ScannerEx s = make_compiled_scanner();
    auto q = f32_vec(15.2f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(15u, result[0]);
    EXPECT_EQ(16u, result[1]);
    EXPECT_EQ(14u, result[2]);
}

TEST_F(ScannerExTest, FindDatasetL2UsesStoredSquaredNormsForHighwayAndNumKong) {
    const std::string dataset_dir =
        tmp_dir() + "/sketch2_utest_scanner_ex_l2_stored_norms_" + std::to_string(getpid());
    const std::string config_path = dataset_dir + ".ini";
    fs::create_directories(dataset_dir);
    std::experimental::scope_exit cleanup([&]() {
        fs::remove_all(dataset_dir);
        std::remove(config_path.c_str());
    });

    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 1.0, 0.0, 0.0, 0.0 ]\n"
        "20 : [ 0.0, 1.0, 0.0, 0.0 ]\n");

    DatasetNode ds;
    ASSERT_EQ(0, ds.init_for_test({dataset_dir}, 100, DataType::f32, 4, DistFunc::L2).code());
    ASSERT_EQ(0, ds.store(input_path_).code());

    overwrite_stored_norm(dataset_dir + "/0.data", 0, 100.0f);

    write_input_raw(
        config_path,
        std::string("[dataset]\n") +
        "dirs = " + dataset_dir + "\n"
        "range_size = 100\n"
        "type = f32\n"
        "dist_func = l2\n"
        "dim = 4\n");

    DatasetReader reader;
    ASSERT_EQ(0, reader.init(config_path).code());

    const auto q = f32_values({1.0f, 0.0f, 0.0f, 0.0f});
    ScannerEx s = make_compiled_scanner();
    std::vector<DistItem> result;
    ASSERT_EQ(0, s.find_items(reader, 2, q.data(), result).code())
        << "engine=" << calc_engine_name(kCompiledEngine);
    ASSERT_EQ(2u, result.size());
    EXPECT_EQ(20u, result[0].id) << "engine=" << calc_engine_name(kCompiledEngine);
    EXPECT_NEAR(2.0, result[0].score, 1e-6) << "engine=" << calc_engine_name(kCompiledEngine);
    EXPECT_EQ(10u, result[1].id) << "engine=" << calc_engine_name(kCompiledEngine);
    EXPECT_NEAR(99.0, result[1].score, 1e-6) << "engine=" << calc_engine_name(kCompiledEngine);
}

TEST_F(ScannerExTest, FindDatasetCosWorks) {
    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 100.0, 1.0, 0.0, 0.0 ]\n"
        "20 : [ 1.0, 1.0, 0.0, 0.0 ]\n"
        "30 : [ -1.0, 0.0, 0.0, 0.0 ]\n");
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::COS, {input_path_}, 100);

    ScannerEx s = make_compiled_scanner();
    auto q = f32_values({1.0f, 0.0f, 0.0f, 0.0f});
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(10u, result[0]);
    EXPECT_EQ(20u, result[1]);
    EXPECT_EQ(30u, result[2]);
}

TEST_F(ScannerExTest, FindDatasetCosRejectsFilesMissingStoredInverseNorms) {
    std::string d = tmp_dir() + "/sketch2_utest_scanner_ex_cos_missing_inv_" + std::to_string(getpid());
    std::string cfg = d + ".ini";
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() {
        fs::remove_all(d);
        std::remove(cfg.c_str());
    });

    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 100.0, 1.0, 0.0, 0.0 ]\n"
        "20 : [ 1.0, 1.0, 0.0, 0.0 ]\n"
        "30 : [ -1.0, 0.0, 0.0, 0.0 ]\n");
    DataWriter writer;
    ASSERT_EQ(0, writer.exec_for_testing(input_path_, d + "/0.data").code());

    DatasetNode ds;
    ASSERT_EQ(0, ds.init_for_test({d}, 100, DataType::f32, 4, DistFunc::COS).code());

    write_input_raw(
        cfg,
        std::string("[dataset]\n") +
        "dirs = " + d + "\n"
        "range_size = 100\n"
        "type = f32\n"
        "dist_func = cos\n"
        "dim = 4\n");

    DatasetReader reader;
    ASSERT_EQ(0, reader.init(cfg).code());

    ScannerEx s = make_compiled_scanner();
    auto q = f32_values({1.0f, 0.0f, 0.0f, 0.0f});
    std::vector<uint64_t> result;
    const Ret ret = s.find(reader, 3, q.data(), result);
    EXPECT_NE(0, ret.code());
    EXPECT_TRUE(result.empty());
    EXPECT_NE(std::string(ret.message()).find("missing stored norms"), std::string::npos);
}

TEST_F(ScannerExTest, FindDatasetL2RejectsFilesMissingStoredNorms) {
    std::string d = tmp_dir() + "/sketch2_utest_scanner_ex_l2_missing_norms_" + std::to_string(getpid());
    std::string cfg = d + ".ini";
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() {
        fs::remove_all(d);
        std::remove(cfg.c_str());
    });

    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 100.0, 1.0, 0.0, 0.0 ]\n"
        "20 : [ 1.0, 1.0, 0.0, 0.0 ]\n"
        "30 : [ -1.0, 0.0, 0.0, 0.0 ]\n");
    DataWriter writer;
    ASSERT_EQ(0, writer.exec_for_testing(input_path_, d + "/0.data").code());

    DatasetNode ds;
    ASSERT_EQ(0, ds.init_for_test({d}, 100, DataType::f32, 4, DistFunc::L2).code());

    write_input_raw(
        cfg,
        std::string("[dataset]\n") +
        "dirs = " + d + "\n"
        "range_size = 100\n"
        "type = f32\n"
        "dist_func = l2\n"
        "dim = 4\n");

    DatasetReader reader;
    ASSERT_EQ(0, reader.init(cfg).code());

    ScannerEx s = make_compiled_scanner();
    auto q = f32_values({1.0f, 0.0f, 0.0f, 0.0f});
    std::vector<uint64_t> result;
    const Ret ret = s.find(reader, 3, q.data(), result);
    EXPECT_NE(0, ret.code());
    EXPECT_TRUE(result.empty());
    EXPECT_NE(std::string(ret.message()).find("missing stored norms"), std::string::npos);
}

TEST_F(ScannerExTest, FindDatasetFailsOnNullQueryPointer) {
    generate(3, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_}, 100);
    ScannerEx s = make_compiled_scanner();
    std::vector<uint64_t> result;
    EXPECT_NE(0, s.find(*reader, 1, nullptr, result).code());
}

TEST_F(ScannerExTest, FindDatasetFailsOnZeroCount) {
    generate(3, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_}, 100);
    ScannerEx s = make_compiled_scanner();
    auto q = f32_vec(1.0f, 4);
    std::vector<uint64_t> result;
    EXPECT_NE(0, s.find(*reader, 0, q.data(), result).code());
}

TEST_F(ScannerExTest, FindDatasetSkipsDeletedVectorsFromDelta) {
    std::string d = tmp_dir() + "/sketch2_utest_scanner_ex_dsdel_" + std::to_string(getpid());
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    DatasetNode ds;
    ASSERT_EQ(0, ds.init_for_test({d}, 100, DataType::f32, 4).code());

    generate_input_file(input_path_,
        GeneratorConfig{PatternType::Sequential, 5, 0, DataType::f32, 4, 1000});
    ASSERT_EQ(0, ds.store(input_path_).code());

    {
        std::ofstream f(input_path_);
        f << "f32,4\n2 : []\n";
    }
    ASSERT_EQ(0, ds.store(input_path_).code());
    ASSERT_TRUE(fs::exists(d + "/0.delta")) << "expected a delta file to exist";

    ScannerEx s = make_compiled_scanner();
    auto q = f32_vec(2.1f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(ds, 5, q.data(), result).code());

    EXPECT_EQ(4u, result.size());
    for (uint64_t id : result) {
        EXPECT_NE(2u, id) << "deleted id=2 must not appear in results";
    }
}

TEST_F(ScannerExTest, FindDatasetUsesUpdatedVectorFromDelta) {
    std::string d = tmp_dir() + "/sketch2_utest_scanner_ex_dsupd_" + std::to_string(getpid());
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    DatasetNode ds;
    ASSERT_EQ(0, ds.init_for_test({d}, 100, DataType::f32, 4).code());

    generate_input_file(input_path_,
        GeneratorConfig{PatternType::Sequential, 5, 0, DataType::f32, 4, 1000});
    ASSERT_EQ(0, ds.store(input_path_).code());

    {
        std::ofstream f(input_path_);
        f << "f32,4\n1 : [ 500.0, 500.0, 500.0, 500.0 ]\n";
    }
    ASSERT_EQ(0, ds.store(input_path_).code());
    ASSERT_TRUE(fs::exists(d + "/0.delta")) << "expected a delta file to exist";

    ScannerEx s = make_compiled_scanner();
    auto q = f32_vec(0.0f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(ds, 1, q.data(), result).code());
    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(0u, result[0]);
}

// ---------------------------------------------------------------------------
// Concurrent scan tests
// ---------------------------------------------------------------------------

class ScannerExConcurrentTest : public ScannerExTest {
protected:
    void SetUp() override {
        ScannerExTest::SetUp();
        prior_pool_ = get_singleton().thread_pool();
        Singleton::force_thread_pool_for_testing(4);
    }

    void TearDown() override {
        Singleton::force_thread_pool_for_testing(prior_pool_);
        ScannerExTest::TearDown();
    }

private:
    std::shared_ptr<ThreadPool> prior_pool_;
};

TEST_F(ScannerExConcurrentTest, DOTTopKSpansMultipleReaders) {
    generate(30, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_}, 10);

    ScannerEx s = make_compiled_scanner();
    auto q = f32_vec(9.5f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(29u, result[0]);
    EXPECT_EQ(28u, result[1]);
    EXPECT_EQ(27u, result[2]);
}

TEST_F(ScannerExConcurrentTest, L2TopKSpansMultipleReaders) {
    generate(30, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::L2, {input_path_}, 10);

    ScannerEx s = make_compiled_scanner();
    auto q = f32_vec(9.5f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(9u, result[0]);
    EXPECT_EQ(10u, result[1]);
    EXPECT_EQ(8u, result[2]);
}

TEST_F(ScannerExConcurrentTest, FindItemsSpansMultipleReaders) {
    generate(30, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_}, 10);

    ScannerEx s = make_compiled_scanner();
    auto q = f32_vec(15.2f, 4);
    std::vector<DistItem> result;
    ASSERT_EQ(0, s.find_items(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(29u, result[0].id);
    EXPECT_EQ(28u, result[1].id);
    EXPECT_EQ(27u, result[2].id);
    EXPECT_NEAR(1769.28, result[0].score, 1e-2);
    EXPECT_NEAR(1708.48, result[1].score, 1e-2);
    EXPECT_NEAR(1647.68, result[2].score, 1e-2);
}

TEST_F(ScannerExConcurrentTest, SingleReaderWithPoolFallsBackToSequential) {
    generate(5, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_}, 1000);

    ScannerEx s = make_compiled_scanner();
    auto q = f32_vec(2.2f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(4u, result[0]);
    EXPECT_EQ(3u, result[1]);
    EXPECT_EQ(2u, result[2]);
}

TEST_F(ScannerExConcurrentTest, CosineTopKSpansMultipleReaders) {
    write_input_raw(
        input_path_,
        "f32,4\n"
        "5  : [ 100.0, 1.0, 0.0, 0.0 ]\n"
        "15 : [ 1.0, 1.0, 0.0, 0.0 ]\n"
        "25 : [ -1.0, 0.0, 0.0, 0.0 ]\n");
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::COS, {input_path_}, 10);

    ScannerEx s = make_compiled_scanner();
    auto q = f32_values({1.0f, 0.0f, 0.0f, 0.0f});
    std::vector<uint64_t> result;
    ASSERT_EQ(0, s.find(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(5u, result[0]);
    EXPECT_EQ(15u, result[1]);
    EXPECT_EQ(25u, result[2]);
}
