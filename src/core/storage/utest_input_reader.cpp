#include <gtest/gtest.h>
#include <fstream>
#include <cstdio>
#include <unistd.h>
#include "core/storage/input_generator.h"
#include "core/storage/input_reader.h"

using namespace sketch2;

class InputReaderTest : public ::testing::Test {
protected:
    std::string path_;

    void SetUp() override {
        path_ = "/tmp/sketch2_utest_ir_" + std::to_string(getpid()) + ".txt";
    }

    void TearDown() override {
        std::remove(path_.c_str());
    }

    void write_raw(const std::string& content) {
        std::ofstream f(path_);
        f << content;
    }

    GeneratorConfig cfg(size_t count, size_t min_id, DataType type, size_t dim) {
        return {PatternType::Sequential, count, min_id, type, dim, 1000};
    }
};

// --- init error cases ---

TEST_F(InputReaderTest, FailsOnBadPath) {
    InputReader r;
    Ret ret = r.init("/nonexistent/dir/file.txt");
    EXPECT_NE(0, ret.code());
}

TEST_F(InputReaderTest, FailsOnEmptyFile) {
    write_raw("");
    InputReader r;
    Ret ret = r.init(path_);
    EXPECT_NE(0, ret.code());
}

TEST_F(InputReaderTest, FailsOnMissingCommaInHeader) {
    write_raw("f32\n0 : [ 0.1 ]\n");
    InputReader r;
    Ret ret = r.init(path_);
    EXPECT_NE(0, ret.code());
}

TEST_F(InputReaderTest, FailsOnUnknownType) {
    write_raw("f64,4\n0 : [ 0.1, 0.1, 0.1, 0.1 ]\n");
    InputReader r;
    Ret ret = r.init(path_);
    EXPECT_NE(0, ret.code());
}

TEST_F(InputReaderTest, FailsOnUnsortedIds) {
    write_raw("f32,4\n10 : [ 1.0, 1.0, 1.0, 1.0 ]\n9 : [ 2.0, 2.0, 2.0, 2.0 ]\n");
    InputReader r;
    EXPECT_NE(0, r.init(path_).code());
}

// --- init success + metadata ---

TEST_F(InputReaderTest, SuccessReturnCode) {
    generate_input_file(path_, cfg(3, 0, DataType::f32, 4));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
}

TEST_F(InputReaderTest, TypeF32) {
    generate_input_file(path_, cfg(1, 0, DataType::f32, 4));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    EXPECT_EQ(DataType::f32, r.type());
}

TEST_F(InputReaderTest, TypeF16) {
    if (!supports_f16()) {
        return;
    }
    generate_input_file(path_, cfg(1, 0, DataType::f16, 4));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    EXPECT_EQ(DataType::f16, r.type());
}

TEST_F(InputReaderTest, TypeI16) {
    generate_input_file(path_, cfg(1, 0, DataType::i16, 4));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    EXPECT_EQ(DataType::i16, r.type());
}

TEST_F(InputReaderTest, DimIsCorrect) {
    generate_input_file(path_, cfg(1, 0, DataType::f32, 128));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    EXPECT_EQ(128u, r.dim());
}

TEST_F(InputReaderTest, CountMatchesConfig) {
    const size_t count = 7;
    generate_input_file(path_, cfg(count, 0, DataType::f32, 4));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    EXPECT_EQ(count, r.count());
}

TEST_F(InputReaderTest, CountWithMinIdOffset) {
    generate_input_file(path_, cfg(5, 100, DataType::f32, 4));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    EXPECT_EQ(5u, r.count());
}

TEST_F(InputReaderTest, SingleVector) {
    generate_input_file(path_, cfg(1, 42, DataType::f32, 8));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    EXPECT_EQ(1u, r.count());
    EXPECT_EQ(DataType::f32, r.type());
    EXPECT_EQ(8u, r.dim());
}

// --- size() ---

TEST_F(InputReaderTest, SizeF32) {
    generate_input_file(path_, cfg(1, 0, DataType::f32, 4));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    EXPECT_EQ(4u * sizeof(float), r.size());
}

TEST_F(InputReaderTest, SizeF16) {
    if (!supports_f16()) {
        write_raw("f16,4\n0 : [ 0.1, 0.1, 0.1, 0.1 ]\n");
        InputReader r;
        EXPECT_NE(0, r.init(path_).code());
        return;
    }

    generate_input_file(path_, cfg(1, 0, DataType::f16, 4));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    EXPECT_EQ(4u * 2u, r.size());
}

TEST_F(InputReaderTest, SizeI16) {
    generate_input_file(path_, cfg(1, 0, DataType::i16, 4));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    EXPECT_EQ(4u * sizeof(uint16_t), r.size());
}

// --- data() ---

TEST_F(InputReaderTest, DataReturnsSuccess) {
    generate_input_file(path_, cfg(3, 0, DataType::f32, 4));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    std::vector<uint8_t> buf(r.size());
    EXPECT_EQ(0, r.data(0, buf.data(), buf.size()).code());
}

TEST_F(InputReaderTest, F32DataValuesAreIdPlusPointOne) {
    // generator writes value = id + 0.1 for each dimension
    generate_input_file(path_, cfg(3, 10, DataType::f32, 4));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    std::vector<uint8_t> buf(r.size());
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(0, r.data(i, buf.data(), buf.size()).code());
        const float* v = reinterpret_cast<const float*>(buf.data());
        float expected = static_cast<float>(10 + i) + 0.1f;
        for (size_t d = 0; d < 4; ++d) {
            EXPECT_NEAR(expected, v[d], 1e-4f) << "vector " << i << " dim " << d;
        }
    }
}

TEST_F(InputReaderTest, I16DataValuesAreId) {
    // generator writes id for each dimension
    generate_input_file(path_, cfg(3, 5, DataType::i16, 4));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    std::vector<uint8_t> buf(r.size());
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(0, r.data(i, buf.data(), buf.size()).code());
        const int16_t* v = reinterpret_cast<const int16_t*>(buf.data());
        int16_t expected = static_cast<int16_t>(5 + i);
        for (size_t d = 0; d < 4; ++d) {
            EXPECT_EQ(expected, v[d]) << "vector " << i << " dim " << d;
        }
    }
}

TEST_F(InputReaderTest, DataDoesNotCrossIntoNextLineWhenPayloadIsShort) {
    write_raw(
        "f32,4\n"
        "10 : [ 10.1, 10.1 ]\n"
        "11 : [ 11.1, 11.1, 11.1, 11.1 ]\n");
    InputReader r;
    ASSERT_EQ(0, r.init(path_).code());
    std::vector<uint8_t> buf(r.size());
    EXPECT_NE(0, r.data(0, buf.data(), buf.size()).code());
}

TEST_F(InputReaderTest, DataFailsOnExtraTokensInPayload) {
    write_raw(
        "f32,4\n"
        "10 : [ 10.1, 10.1, 10.1, 10.1, 10.1 ]\n");
    InputReader r;
    ASSERT_EQ(0, r.init(path_).code());
    std::vector<uint8_t> buf(r.size());
    EXPECT_NE(0, r.data(0, buf.data(), buf.size()).code());
}

// --- id() ---

TEST_F(InputReaderTest, IdReturnsCorrectIds) {
    generate_input_file(path_, cfg(4, 10, DataType::f32, 4));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(10u + i, r.id(i));
    }
}

TEST_F(InputReaderTest, IdOutOfBoundsThrows) {
    generate_input_file(path_, cfg(3, 0, DataType::f32, 4));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    EXPECT_THROW(r.id(3),   std::out_of_range);
    EXPECT_THROW(r.id(100), std::out_of_range);
}

// --- is_no_data() ---

TEST_F(InputReaderTest, IsNoDataOutOfBoundsThrows) {
    generate_input_file(path_, cfg(3, 0, DataType::f32, 4));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    EXPECT_THROW(r.is_no_data(3),   std::out_of_range);
    EXPECT_THROW(r.is_no_data(100), std::out_of_range);
}

TEST_F(InputReaderTest, DataOutOfBoundsReturnsError) {
    generate_input_file(path_, cfg(3, 0, DataType::f32, 4));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    std::vector<uint8_t> buf(r.size());
    EXPECT_NE(0, r.data(3,   buf.data(), buf.size()).code());
    EXPECT_NE(0, r.data(100, buf.data(), buf.size()).code());
}

TEST_F(InputReaderTest, IsNoDataReturnsFalseForNormalVector) {
    generate_input_file(path_, cfg(3, 0, DataType::f32, 4));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    EXPECT_FALSE(r.is_no_data(0));
    EXPECT_FALSE(r.is_no_data(1));
    EXPECT_FALSE(r.is_no_data(2));
}

TEST_F(InputReaderTest, IsNoDataReturnsTrueForEmptyBrackets) {
    // Write a line with empty brackets: "5 : [  ]\n"
    write_raw("f32,4\n5 : []\n");
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    ASSERT_EQ(1u, r.count());
    EXPECT_TRUE(r.is_no_data(0));
}

TEST_F(InputReaderTest, F16DataWorks) {
    if (!supports_f16()) {
        return;
    }
    generate_input_file(path_, cfg(1, 42, DataType::f16, 4));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    std::vector<uint8_t> buf(r.size());
    EXPECT_EQ(0, r.data(0, buf.data(), buf.size()).code());
    const uint16_t* v = reinterpret_cast<const uint16_t*>(buf.data());
    EXPECT_NE(0, v[0]); // should contain parsed f16 value
}

// --- is_range_present() ---

TEST_F(InputReaderTest, IsRangePresentReturnsTrueForOverlap) {
    generate_input_file(path_, cfg(5, 10, DataType::f32, 4)); // ids: 10..14
    InputReader r;
    ASSERT_EQ(0, r.init(path_).code());
    EXPECT_TRUE(r.is_range_present(11, 13));
}

TEST_F(InputReaderTest, IsRangePresentReturnsFalseForRangeBeforeAllIds) {
    generate_input_file(path_, cfg(5, 10, DataType::f32, 4));
    InputReader r;
    ASSERT_EQ(0, r.init(path_).code());
    EXPECT_FALSE(r.is_range_present(0, 10));
}

TEST_F(InputReaderTest, IsRangePresentReturnsFalseForRangeAfterAllIds) {
    generate_input_file(path_, cfg(5, 10, DataType::f32, 4));
    InputReader r;
    ASSERT_EQ(0, r.init(path_).code());
    EXPECT_FALSE(r.is_range_present(15, 20));
}

TEST_F(InputReaderTest, IsRangePresentTreatsEndAsExclusive) {
    generate_input_file(path_, cfg(5, 10, DataType::f32, 4));
    InputReader r;
    ASSERT_EQ(0, r.init(path_).code());
    EXPECT_FALSE(r.is_range_present(0, 10));
    EXPECT_TRUE(r.is_range_present(14, 15));
}

TEST_F(InputReaderTest, IsRangePresentReturnsFalseForEmptyOrInvalidRange) {
    generate_input_file(path_, cfg(5, 10, DataType::f32, 4));
    InputReader r;
    ASSERT_EQ(0, r.init(path_).code());
    EXPECT_FALSE(r.is_range_present(12, 12));
    EXPECT_FALSE(r.is_range_present(13, 12));
}

// =============================================================================
// InputReaderView tests
// =============================================================================

class InputReaderViewTest : public ::testing::Test {
protected:
    std::string path_;

    void SetUp() override {
        path_ = "/tmp/sketch2_utest_irv_" + std::to_string(getpid()) + ".txt";
    }

    void TearDown() override {
        std::remove(path_.c_str());
    }

    void write_raw(const std::string& content) {
        std::ofstream f(path_);
        f << content;
    }

    GeneratorConfig cfg(size_t count, size_t min_id, DataType type, size_t dim) {
        return {PatternType::Sequential, count, min_id, type, dim, 1000};
    }
};

// --- whole-reader view (0, 0) ---

TEST_F(InputReaderViewTest, WholeViewCount) {
    generate_input_file(path_, cfg(5, 10, DataType::f32, 4));
    InputReader r;
    ASSERT_EQ(0, r.init(path_).code());
    InputReaderView v(r, 0, 0);
    EXPECT_EQ(5u, v.count());
}

TEST_F(InputReaderViewTest, WholeViewMetadata) {
    generate_input_file(path_, cfg(3, 0, DataType::f32, 8));
    InputReader r;
    ASSERT_EQ(0, r.init(path_).code());
    InputReaderView v(r, 0, 0);
    EXPECT_EQ(r.type(), v.type());
    EXPECT_EQ(r.dim(),  v.dim());
    EXPECT_EQ(r.size(), v.size());
}

TEST_F(InputReaderViewTest, WholeViewIds) {
    generate_input_file(path_, cfg(4, 10, DataType::f32, 4));
    InputReader r;
    ASSERT_EQ(0, r.init(path_).code());
    InputReaderView v(r, 0, 0);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(r.id(i), v.id(i));
    }
}

TEST_F(InputReaderViewTest, WholeViewData) {
    generate_input_file(path_, cfg(3, 5, DataType::f32, 4));
    InputReader r;
    ASSERT_EQ(0, r.init(path_).code());
    InputReaderView v(r, 0, 0);
    std::vector<uint8_t> buf(v.size());
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(0, v.data(i, buf.data(), buf.size()).code());
        const float* vd = reinterpret_cast<const float*>(buf.data());
        float expected = static_cast<float>(5 + i) + 0.1f;
        for (size_t d = 0; d < 4; ++d) {
            EXPECT_NEAR(expected, vd[d], 1e-4f) << "vector " << i << " dim " << d;
        }
    }
}

// --- partial range ---

TEST_F(InputReaderViewTest, PartialViewCount) {
    // reader has ids 10..14, view [11, 14) -> ids 11, 12, 13
    generate_input_file(path_, cfg(5, 10, DataType::f32, 4));
    InputReader r;
    ASSERT_EQ(0, r.init(path_).code());
    InputReaderView v(r, 11, 14);
    EXPECT_EQ(3u, v.count());
}

TEST_F(InputReaderViewTest, PartialViewIds) {
    generate_input_file(path_, cfg(5, 10, DataType::f32, 4));
    InputReader r;
    ASSERT_EQ(0, r.init(path_).code());
    InputReaderView v(r, 11, 14);
    EXPECT_EQ(11u, v.id(0));
    EXPECT_EQ(12u, v.id(1));
    EXPECT_EQ(13u, v.id(2));
}

TEST_F(InputReaderViewTest, PartialViewData) {
    generate_input_file(path_, cfg(5, 10, DataType::f32, 4));
    InputReader r;
    ASSERT_EQ(0, r.init(path_).code());
    InputReaderView v(r, 11, 14);
    std::vector<uint8_t> buf(v.size());
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(0, v.data(i, buf.data(), buf.size()).code());
        const float* vd = reinterpret_cast<const float*>(buf.data());
        float expected = static_cast<float>(11 + i) + 0.1f;
        for (size_t d = 0; d < 4; ++d) {
            EXPECT_NEAR(expected, vd[d], 1e-4f) << "vector " << i << " dim " << d;
        }
    }
}

TEST_F(InputReaderViewTest, SingleEntryAtStart) {
    generate_input_file(path_, cfg(5, 10, DataType::f32, 4));
    InputReader r;
    ASSERT_EQ(0, r.init(path_).code());
    InputReaderView v(r, 10, 11);
    ASSERT_EQ(1u, v.count());
    EXPECT_EQ(10u, v.id(0));
}

TEST_F(InputReaderViewTest, SingleEntryAtEnd) {
    generate_input_file(path_, cfg(5, 10, DataType::f32, 4));
    InputReader r;
    ASSERT_EQ(0, r.init(path_).code());
    InputReaderView v(r, 14, 15);
    ASSERT_EQ(1u, v.count());
    EXPECT_EQ(14u, v.id(0));
}

TEST_F(InputReaderViewTest, EmptyRangeNoMatch) {
    generate_input_file(path_, cfg(5, 10, DataType::f32, 4));
    InputReader r;
    ASSERT_EQ(0, r.init(path_).code());
    InputReaderView v(r, 20, 30);
    EXPECT_EQ(0u, v.count());
}

// --- invalid argument ---

TEST_F(InputReaderViewTest, StartGreaterThanEndThrows) {
    generate_input_file(path_, cfg(3, 0, DataType::f32, 4));
    InputReader r;
    ASSERT_EQ(0, r.init(path_).code());
    EXPECT_THROW((InputReaderView{r, 10, 5}), std::invalid_argument);
}

// --- out of bounds ---

TEST_F(InputReaderViewTest, IdOutOfBoundsThrows) {
    generate_input_file(path_, cfg(5, 10, DataType::f32, 4));
    InputReader r;
    ASSERT_EQ(0, r.init(path_).code());
    InputReaderView v(r, 11, 13); // 2 entries
    EXPECT_THROW(v.id(2),   std::out_of_range);
    EXPECT_THROW(v.id(100), std::out_of_range);
}

TEST_F(InputReaderViewTest, DataOutOfBoundsReturnsError) {
    generate_input_file(path_, cfg(5, 10, DataType::f32, 4));
    InputReader r;
    ASSERT_EQ(0, r.init(path_).code());
    InputReaderView v(r, 11, 13);
    std::vector<uint8_t> buf(v.size());
    EXPECT_NE(0, v.data(2,   buf.data(), buf.size()).code());
    EXPECT_NE(0, v.data(100, buf.data(), buf.size()).code());
}

TEST_F(InputReaderViewTest, IsNoDataOutOfBoundsThrows) {
    generate_input_file(path_, cfg(5, 10, DataType::f32, 4));
    InputReader r;
    ASSERT_EQ(0, r.init(path_).code());
    InputReaderView v(r, 11, 13);
    EXPECT_THROW(v.is_no_data(2),   std::out_of_range);
    EXPECT_THROW(v.is_no_data(100), std::out_of_range);
}

// --- is_no_data ---

TEST_F(InputReaderViewTest, IsNoDataFalseForNormalVector) {
    generate_input_file(path_, cfg(3, 10, DataType::f32, 4));
    InputReader r;
    ASSERT_EQ(0, r.init(path_).code());
    InputReaderView v(r, 10, 13);
    EXPECT_FALSE(v.is_no_data(0));
    EXPECT_FALSE(v.is_no_data(1));
    EXPECT_FALSE(v.is_no_data(2));
}

TEST_F(InputReaderViewTest, IsNoDataTrueForEmptyBrackets) {
    write_raw("f32,4\n10 : [ 1.0, 1.0, 1.0, 1.0 ]\n11 : []\n12 : [ 3.0, 3.0, 3.0, 3.0 ]\n");
    InputReader r;
    ASSERT_EQ(0, r.init(path_).code());
    InputReaderView v(r, 11, 12);
    ASSERT_EQ(1u, v.count());
    EXPECT_TRUE(v.is_no_data(0));
}
