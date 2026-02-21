#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <unistd.h>
#include "core/storage/input_generator.h"

using namespace sketch2;

class InputGeneratorTest : public ::testing::Test {
protected:
    std::string path_;

    void SetUp() override {
        path_ = "/tmp/sketch2_utest_ig_" + std::to_string(getpid()) + ".txt";
    }

    void TearDown() override {
        std::remove(path_.c_str());
    }

    std::vector<std::string> read_lines() {
        std::ifstream f(path_);
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(f, line))
            lines.push_back(line);
        return lines;
    }
};

TEST_F(InputGeneratorTest, FailsOnNonSequentialPattern) {
    GeneratorConfig cfg{PatternType::Random, 10, 0, DataType::f32, 4};
    Ret ret = generate_input_file(path_, cfg);
    EXPECT_NE(0, ret.code());
}

TEST_F(InputGeneratorTest, FailsOnBadPath) {
    GeneratorConfig cfg{PatternType::Sequential, 10, 0, DataType::f32, 4};
    Ret ret = generate_input_file("/nonexistent/dir/file.txt", cfg);
    EXPECT_NE(0, ret.code());
}

TEST_F(InputGeneratorTest, SuccessReturnCode) {
    GeneratorConfig cfg{PatternType::Sequential, 3, 0, DataType::f32, 4};
    Ret ret = generate_input_file(path_, cfg);
    EXPECT_EQ(0, ret.code());
}

TEST_F(InputGeneratorTest, HeaderLineF32) {
    GeneratorConfig cfg{PatternType::Sequential, 1, 0, DataType::f32, 128};
    generate_input_file(path_, cfg);
    auto lines = read_lines();
    ASSERT_FALSE(lines.empty());
    EXPECT_EQ("f32,128", lines[0]);
}

TEST_F(InputGeneratorTest, HeaderLineF16) {
    GeneratorConfig cfg{PatternType::Sequential, 1, 0, DataType::f16, 64};
    generate_input_file(path_, cfg);
    auto lines = read_lines();
    ASSERT_FALSE(lines.empty());
    EXPECT_EQ("f16,64", lines[0]);
}

TEST_F(InputGeneratorTest, LineCount) {
    const size_t count = 5;
    GeneratorConfig cfg{PatternType::Sequential, count, 0, DataType::f32, 4};
    generate_input_file(path_, cfg);
    auto lines = read_lines();
    EXPECT_EQ(count + 1, lines.size()); // header + one line per vector
}

TEST_F(InputGeneratorTest, IdsStartAtMinId) {
    const size_t min_id = 100;
    GeneratorConfig cfg{PatternType::Sequential, 3, min_id, DataType::f32, 4};
    generate_input_file(path_, cfg);
    auto lines = read_lines();
    ASSERT_EQ(4u, lines.size());
    EXPECT_EQ(0u, lines[1].find("100 : [ "));
    EXPECT_EQ(0u, lines[2].find("101 : [ "));
    EXPECT_EQ(0u, lines[3].find("102 : [ "));
}

TEST_F(InputGeneratorTest, IdsAreSequential) {
    GeneratorConfig cfg{PatternType::Sequential, 4, 0, DataType::f32, 4};
    generate_input_file(path_, cfg);
    auto lines = read_lines();
    ASSERT_EQ(5u, lines.size());
    for (size_t i = 0; i < 4; ++i) {
        std::string expected_prefix = std::to_string(i) + " : [ ";
        EXPECT_EQ(0u, lines[i + 1].find(expected_prefix)) << "line " << i + 1;
    }
}

TEST_F(InputGeneratorTest, ValueIsIdPlusPointOne) {
    GeneratorConfig cfg{PatternType::Sequential, 1, 7, DataType::f32, 4};
    generate_input_file(path_, cfg);
    auto lines = read_lines();
    ASSERT_EQ(2u, lines.size());
    // id=7, value=7.1 -> printed as %f -> "7.1"
    EXPECT_NE(std::string::npos, lines[1].find("7.1"));
}

TEST_F(InputGeneratorTest, WritesExactlyDimValuesPerVector) {
    const size_t dim = 4;
    GeneratorConfig cfg{PatternType::Sequential, 1, 5, DataType::f32, dim};
    generate_input_file(path_, cfg);
    auto lines = read_lines();
    ASSERT_EQ(2u, lines.size());
    // Count occurrences of the expected value string in the vector line
    const std::string value_str = "5.1";
    const std::string& line = lines[1];
    size_t count = 0;
    size_t pos = 0;
    while ((pos = line.find(value_str, pos)) != std::string::npos) {
        ++count;
        pos += value_str.size();
    }
    EXPECT_EQ(dim, count);
}

TEST_F(InputGeneratorTest, SingleDimVector) {
    GeneratorConfig cfg{PatternType::Sequential, 1, 3, DataType::f32, 1};
    generate_input_file(path_, cfg);
    auto lines = read_lines();
    ASSERT_EQ(2u, lines.size());
    EXPECT_EQ("3 : [ 3.1 ]", lines[1]);
}

// i32 tests

TEST_F(InputGeneratorTest, I32SuccessReturnCode) {
    GeneratorConfig cfg{PatternType::Sequential, 3, 0, DataType::i32, 4};
    Ret ret = generate_input_file(path_, cfg);
    EXPECT_EQ(0, ret.code());
}

TEST_F(InputGeneratorTest, HeaderLineI32) {
    GeneratorConfig cfg{PatternType::Sequential, 1, 0, DataType::i32, 4};
    generate_input_file(path_, cfg);
    auto lines = read_lines();
    ASSERT_FALSE(lines.empty());
    EXPECT_EQ("i32,4", lines[0]);
}

TEST_F(InputGeneratorTest, I32ValueIsId) {
    GeneratorConfig cfg{PatternType::Sequential, 1, 9, DataType::i32, 1};
    generate_input_file(path_, cfg);
    auto lines = read_lines();
    ASSERT_EQ(2u, lines.size());
    EXPECT_EQ("9 : [ 9 ]", lines[1]);
}

TEST_F(InputGeneratorTest, I32WritesExactlyDimValuesPerVector) {
    const size_t dim = 4;
    GeneratorConfig cfg{PatternType::Sequential, 1, 5, DataType::i32, dim};
    generate_input_file(path_, cfg);
    auto lines = read_lines();
    ASSERT_EQ(2u, lines.size());
    // Each value is printed as the id "5"; count occurrences after "[ "
    // Line: "5 : [ 5, 5, 5, 5 ]" — the id prefix "5 " won't match standalone "5,"
    // Count ", " separators: should be dim-1
    const std::string sep = ", ";
    const std::string& line = lines[1];
    size_t count = 0;
    size_t pos = 0;
    while ((pos = line.find(sep, pos)) != std::string::npos) {
        ++count;
        pos += sep.size();
    }
    EXPECT_EQ(dim - 1, count);
}
