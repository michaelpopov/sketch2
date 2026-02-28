#include <gtest/gtest.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <unistd.h>
#include <vector>
#include <fstream>
#include <memory>
#include "core/compute/scanner.h"
#include "core/storage/input_generator.h"
#include "core/storage/data_writer.h"
#include "core/storage/data_reader.h"

using namespace sketch2;

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

TEST_F(ScannerTest, FindFailsOnUnsupportedFunction) {
    generate(3, 0, DataType::f32, 4);
    DataReader reader;
    ASSERT_EQ(0, reader.init(data_path_).code());
    Scanner s;
    auto q = f32_vec(0.0f, 4);
    std::vector<uint64_t> result;
    EXPECT_NE(0, s.find(reader, DistFunc::L2, 1, q.data(), result).code());
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
