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
        return {PatternType::Sequential, count, min_id, type, dim};
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
    generate_input_file(path_, cfg(1, 0, DataType::f16, 4));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    EXPECT_EQ(DataType::f16, r.type());
}

TEST_F(InputReaderTest, TypeI32) {
    generate_input_file(path_, cfg(1, 0, DataType::i32, 4));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    EXPECT_EQ(DataType::i32, r.type());
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
    generate_input_file(path_, cfg(1, 0, DataType::f16, 4));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    EXPECT_EQ(4u * 2u, r.size());
}

TEST_F(InputReaderTest, SizeI32) {
    generate_input_file(path_, cfg(1, 0, DataType::i32, 4));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    EXPECT_EQ(4u * sizeof(uint32_t), r.size());
}

// --- data() ---

TEST_F(InputReaderTest, DataReturnsNonNull) {
    generate_input_file(path_, cfg(3, 0, DataType::f32, 4));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    EXPECT_NE(nullptr, r.data(0));
}

TEST_F(InputReaderTest, F32DataValuesAreIdPlusPointOne) {
    // generator writes value = id + 0.1 for each dimension
    generate_input_file(path_, cfg(3, 10, DataType::f32, 4));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    for (size_t i = 0; i < 3; ++i) {
        const float* v = reinterpret_cast<const float*>(r.data(i));
        float expected = static_cast<float>(10 + i) + 0.1f;
        for (size_t d = 0; d < 4; ++d) {
            EXPECT_NEAR(expected, v[d], 1e-4f) << "vector " << i << " dim " << d;
        }
    }
}

TEST_F(InputReaderTest, I32DataValuesAreId) {
    // generator writes id for each dimension
    generate_input_file(path_, cfg(3, 5, DataType::i32, 4));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    for (size_t i = 0; i < 3; ++i) {
        const uint32_t* v = reinterpret_cast<const uint32_t*>(r.data(i));
        uint32_t expected = static_cast<uint32_t>(5 + i);
        for (size_t d = 0; d < 4; ++d) {
            EXPECT_EQ(expected, v[d]) << "vector " << i << " dim " << d;
        }
    }
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

TEST_F(InputReaderTest, DataOutOfBoundsThrows) {
    generate_input_file(path_, cfg(3, 0, DataType::f32, 4));
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    EXPECT_THROW(r.data(3),   std::out_of_range);
    EXPECT_THROW(r.data(100), std::out_of_range);
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
    write_raw("f32,4\n5 : [  ]\n");
    InputReader r;
    EXPECT_EQ(0, r.init(path_).code());
    ASSERT_EQ(1u, r.count());
    EXPECT_TRUE(r.is_no_data(0));
}
