#include <gtest/gtest.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <unistd.h>
#include <vector>
#include <fstream>
#include <stdexcept>
#include "core/compute/scanner.h"
#include "core/storage/input_generator.h"
#include "core/storage/data_writer.h"

using namespace sketch2;

class ScannerTest : public ::testing::Test {
protected:
    std::string input_path_;
    std::string data_path_;

    void SetUp() override {
        std::string base = "/tmp/sketch2_utest_sc_" + std::to_string(getpid());
        input_path_ = base + ".txt";
        data_path_  = base + ".bin";
    }

    void TearDown() override {
        std::remove(input_path_.c_str());
        std::remove(data_path_.c_str());
    }

    void generate(size_t count, size_t min_id, DataType type, size_t dim) {
        GeneratorConfig cfg{PatternType::Sequential, count, min_id, type, dim, 1000};
        generate_input_file(input_path_, cfg);
        DataWriter w;
        w.init(input_path_, data_path_);
        w.exec();
    }

    // Query vector helpers: all elements set to the same value.
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

    // f16-encoded query vector, uses the same encoding as InputReader.
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

// --- init ---

TEST_F(ScannerTest, FailsOnBadPath) {
    Scanner s;
    EXPECT_NE(0, s.init("/nonexistent/path/file.bin").code());
}

TEST_F(ScannerTest, FailsOnInvalidFile) {
    { std::ofstream f(data_path_); f << "bad"; }
    Scanner s;
    EXPECT_NE(0, s.init(data_path_).code());
}

TEST_F(ScannerTest, FindBeforeInitReturnsEmpty) {
    Scanner s;
    auto q = f32_vec(0.0f, 4);
    EXPECT_TRUE(s.find(DistFunc::L1, 1, q.data()).empty());
}

TEST_F(ScannerTest, SuccessReturnCode) {
    generate(3, 0, DataType::f32, 4);
    Scanner s;
    EXPECT_EQ(0, s.init(data_path_).code());
}

// --- find() edge cases ---

TEST_F(ScannerTest, FindCountZeroReturnsEmpty) {
    generate(3, 0, DataType::f32, 4);
    Scanner s;
    s.init(data_path_);
    auto q = f32_vec(0.0f, 4);
    EXPECT_TRUE(s.find(DistFunc::L1, 0, q.data()).empty());
}

TEST_F(ScannerTest, FindNullQueryPointerReturnsEmpty) {
    generate(3, 0, DataType::f32, 4);
    Scanner s;
    s.init(data_path_);
    EXPECT_TRUE(s.find(DistFunc::L1, 1, nullptr).empty());
}

TEST_F(ScannerTest, FindUnsupportedFuncReturnsEmpty) {
    generate(3, 0, DataType::f32, 4);
    Scanner s;
    s.init(data_path_);
    auto q = f32_vec(0.0f, 4);
    EXPECT_TRUE(s.find(DistFunc::L2, 1, q.data()).empty());
}

TEST_F(ScannerTest, FindUnknownDistFuncReturnsEmpty) {
    generate(3, 0, DataType::f32, 4);
    Scanner s;
    s.init(data_path_);
    auto q = f32_vec(0.0f, 4);
    auto unknown = static_cast<DistFunc>(999);
    EXPECT_TRUE(s.find(unknown, 1, q.data()).empty());
}

TEST_F(ScannerTest, FindCountExceedsTotalReturnsCapped) {
    const size_t total = 3;
    generate(total, 0, DataType::f32, 4);
    Scanner s;
    s.init(data_path_);
    auto q = f32_vec(0.0f, 4);
    EXPECT_EQ(total, s.find(DistFunc::L1, 100, q.data()).size());
}

TEST_F(ScannerTest, FindResultSizeMatchesRequest) {
    generate(5, 0, DataType::f32, 4);
    Scanner s;
    s.init(data_path_);
    auto q = f32_vec(3.2f, 4);
    EXPECT_EQ(1u, s.find(DistFunc::L1, 1, q.data()).size());
    EXPECT_EQ(3u, s.find(DistFunc::L1, 3, q.data()).size());
    EXPECT_EQ(5u, s.find(DistFunc::L1, 5, q.data()).size());
}

// --- f32 correctness ---
//
// Sequential f32: vector id=N has all elements = N + 0.1.
// Query [3.2, 3.2, 3.2, 3.2] over ids [0..4], dim=4:
//   dist(3) = 4*0.1 = 0.4,  dist(4) = 4*0.9 = 3.6,  dist(2) = 4*1.1 = 4.4
//   dist(1) = 4*2.1 = 8.4,  dist(0) = 4*3.1 = 12.4
// Expected ascending order: [3, 4, 2, 1, 0].

TEST_F(ScannerTest, FindF32SingleVector) {
    generate(1, 42, DataType::f32, 4);
    Scanner s;
    s.init(data_path_);
    auto q = f32_vec(42.1f, 4); // exact value stored for id=42
    auto result = s.find(DistFunc::L1, 1, q.data());
    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(42u, result[0]);
}

TEST_F(ScannerTest, FindF32K1ReturnsNearest) {
    generate(5, 0, DataType::f32, 4);
    Scanner s;
    s.init(data_path_);
    auto q = f32_vec(3.2f, 4);
    auto result = s.find(DistFunc::L1, 1, q.data());
    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(3u, result[0]);
}

TEST_F(ScannerTest, FindF32K3ReturnsInOrder) {
    generate(5, 0, DataType::f32, 4);
    Scanner s;
    s.init(data_path_);
    auto q = f32_vec(3.2f, 4);
    auto result = s.find(DistFunc::L1, 3, q.data());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(3u, result[0]);
    EXPECT_EQ(4u, result[1]);
    EXPECT_EQ(2u, result[2]);
}

TEST_F(ScannerTest, FindF32TieCaseDeterministicOrdering) {
    // Stored values are id + 0.1. Query 2.6 is equidistant to ids 2 (2.1) and 3 (3.1).
    generate(5, 0, DataType::f32, 4);
    Scanner s;
    s.init(data_path_);
    auto q = f32_vec(2.6f, 4);
    auto result = s.find(DistFunc::L1, 2, q.data());
    ASSERT_EQ(2u, result.size());
    EXPECT_EQ(2u, result[0]);
    EXPECT_EQ(3u, result[1]);
}

TEST_F(ScannerTest, FindF32KAllReturnsSortedByDistance) {
    generate(5, 0, DataType::f32, 4);
    Scanner s;
    s.init(data_path_);
    auto q = f32_vec(3.2f, 4);
    auto result = s.find(DistFunc::L1, 5, q.data());
    ASSERT_EQ(5u, result.size());
    EXPECT_EQ(3u, result[0]);
    EXPECT_EQ(4u, result[1]);
    EXPECT_EQ(2u, result[2]);
    EXPECT_EQ(1u, result[3]);
    EXPECT_EQ(0u, result[4]);
}

TEST_F(ScannerTest, FindF32NonZeroMinId) {
    // ids [10..14]; query near id=12 → nearest is 12
    generate(5, 10, DataType::f32, 4);
    Scanner s;
    s.init(data_path_);
    auto q = f32_vec(12.2f, 4);
    auto result = s.find(DistFunc::L1, 1, q.data());
    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(12u, result[0]);
}

// --- i16 correctness ---
//
// Sequential i16: vector id=N has all elements = N.
// Query [0, 0, 0, 0] over ids [0, 1, 2], dim=4:
//   dist(0) = 0, dist(1) = 4, dist(2) = 8
// Expected ascending order: [0, 1, 2].

TEST_F(ScannerTest, FindI16K1ReturnsNearest) {
    generate(3, 0, DataType::i16, 4);
    Scanner s;
    s.init(data_path_);
    auto q = i16_vec(0, 4);
    auto result = s.find(DistFunc::L1, 1, q.data());
    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(0u, result[0]);
}

TEST_F(ScannerTest, FindI16K2ReturnsCorrectOrder) {
    generate(3, 0, DataType::i16, 4);
    Scanner s;
    s.init(data_path_);
    auto q = i16_vec(0, 4);
    auto result = s.find(DistFunc::L1, 2, q.data());
    ASSERT_EQ(2u, result.size());
    EXPECT_EQ(0u, result[0]);
    EXPECT_EQ(1u, result[1]);
}

TEST_F(ScannerTest, FindI16AllSortedByDistance) {
    generate(3, 0, DataType::i16, 4);
    Scanner s;
    s.init(data_path_);
    auto q = i16_vec(0, 4);
    auto result = s.find(DistFunc::L1, 3, q.data());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(0u, result[0]);
    EXPECT_EQ(1u, result[1]);
    EXPECT_EQ(2u, result[2]);
}

// --- f16 ---

TEST_F(ScannerTest, FindF16Works) {
    if (!supports_f16()) {
        return;
    }
    generate(3, 0, DataType::f16, 4);
    Scanner s;
    s.init(data_path_);
    auto q = f16_vec(1.1f, 4);
    // Nearest should be id=1 (which has elements 1.1f)
    auto result = s.find(DistFunc::L1, 1, q.data());
    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(1u, result[0]);
}
