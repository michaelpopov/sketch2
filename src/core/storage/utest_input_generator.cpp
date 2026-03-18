// Unit tests for textual input generation helpers.

#include <gtest/gtest.h>
#include <array>
#include <fstream>
#include <cstring>
#include <sstream>
#include <cstdio>
#include <unistd.h>
#include "core/storage/input_generator.h"
#include "utest_tmp_dir.h"

using namespace sketch2;

class InputGeneratorTest : public ::testing::Test {
protected:
    std::string path_;

    void SetUp() override {
        path_ = tmp_dir() + "/sketch2_utest_ig_" + std::to_string(getpid()) + ".txt";
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

    std::string read_binary_header() {
        std::ifstream f(path_, std::ios::binary);
        std::string header;
        std::getline(f, header);
        return header;
    }

    std::string read_file_bytes() {
        std::ifstream f(path_, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    }
};

TEST_F(InputGeneratorTest, FailsOnZeroCount) {
    GeneratorConfig cfg{PatternType::Sequential, 0, 0, DataType::f32, 4, 1000};
    EXPECT_NE(0, generate_input_file(path_, cfg).code());
}

TEST_F(InputGeneratorTest, FailsOnDimTooSmall) {
    GeneratorConfig cfg{PatternType::Sequential, 1, 0, DataType::f32, 3, 1000};
    EXPECT_NE(0, generate_input_file(path_, cfg).code());
}

TEST_F(InputGeneratorTest, FailsOnDimTooLarge) {
    GeneratorConfig cfg{PatternType::Sequential, 1, 0, DataType::f32, 4097, 1000};
    EXPECT_NE(0, generate_input_file(path_, cfg).code());
}

TEST_F(InputGeneratorTest, SucceedsOnMinDim) {
    GeneratorConfig cfg{PatternType::Sequential, 1, 0, DataType::f32, 4, 1000};
    EXPECT_EQ(0, generate_input_file(path_, cfg).code());
}

TEST_F(InputGeneratorTest, SucceedsOnMaxDim) {
    GeneratorConfig cfg{PatternType::Sequential, 1, 0, DataType::f32, 4096, 1000};
    EXPECT_EQ(0, generate_input_file(path_, cfg).code());
}

TEST_F(InputGeneratorTest, FailsOnBadPath) {
    GeneratorConfig cfg{PatternType::Sequential, 10, 0, DataType::f32, 4, 1000};
    Ret ret = generate_input_file("/nonexistent/dir/file.txt", cfg);
    EXPECT_NE(0, ret.code());
}

TEST_F(InputGeneratorTest, BinaryModeRejectsDeletedItems) {
    GeneratorConfig cfg{PatternType::Sequential, 10, 0, DataType::f32, 4, 1000, 2, true};
    const Ret ret = generate_input_file(path_, cfg);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("binary input format does not support deleted items", ret.message());
}

TEST_F(InputGeneratorTest, SuccessReturnCode) {
    GeneratorConfig cfg{PatternType::Sequential, 3, 0, DataType::f32, 4, 1000};
    Ret ret = generate_input_file(path_, cfg);
    EXPECT_EQ(0, ret.code());
}

TEST_F(InputGeneratorTest, SuccessReplacesExistingFile) {
    {
        std::ofstream out(path_, std::ios::binary);
        ASSERT_TRUE(out.is_open());
        out << "old contents";
    }

    GeneratorConfig cfg{PatternType::Sequential, 1, 7, DataType::f32, 4, 1000};
    ASSERT_EQ(0, generate_input_file(path_, cfg).code());

    EXPECT_EQ("f32,4\n7 : [ 7.10, 7.10, 7.10, 7.10 ]\n", read_file_bytes());
}

TEST_F(InputGeneratorTest, FailureLeavesExistingFileUntouched) {
    {
        std::ofstream out(path_, std::ios::binary);
        ASSERT_TRUE(out.is_open());
        out << "keep me";
    }

    GeneratorConfig cfg{PatternType::Sequential, 10, 0, DataType::f32, 4, 1000, 2, true};
    const Ret ret = generate_input_file(path_, cfg);
    ASSERT_NE(0, ret.code());
    EXPECT_EQ("keep me", read_file_bytes());
}

TEST_F(InputGeneratorTest, HeaderLineF32) {
    GeneratorConfig cfg{PatternType::Sequential, 1, 0, DataType::f32, 128, 1000};
    generate_input_file(path_, cfg);
    auto lines = read_lines();
    ASSERT_FALSE(lines.empty());
    EXPECT_EQ("f32,128", lines[0]);
}

TEST_F(InputGeneratorTest, HeaderLineF16) {
    GeneratorConfig cfg{PatternType::Sequential, 1, 0, DataType::f16, 64, 1000};
    const Ret ret = generate_input_file(path_, cfg);
    EXPECT_EQ(0, ret.code());
    auto lines = read_lines();
    ASSERT_FALSE(lines.empty());
    EXPECT_EQ("f16,64", lines[0]);
}

TEST_F(InputGeneratorTest, F16SequentialTextValueIsIdPlusPointOneWhenSupported) {


    GeneratorConfig cfg{PatternType::Sequential, 1, 7, DataType::f16, 4, 1000};
    ASSERT_EQ(0, generate_input_file(path_, cfg).code());

    auto lines = read_lines();
    ASSERT_EQ(2u, lines.size());
    EXPECT_EQ("7 : [ 7.10, 7.10, 7.10, 7.10 ]", lines[1]);
}

TEST_F(InputGeneratorTest, LineCount) {
    const size_t count = 5;
    GeneratorConfig cfg{PatternType::Sequential, count, 0, DataType::f32, 4, 1000};
    generate_input_file(path_, cfg);
    auto lines = read_lines();
    EXPECT_EQ(count + 1, lines.size()); // header + one line per vector
}

TEST_F(InputGeneratorTest, IdsStartAtMinId) {
    const size_t min_id = 100;
    GeneratorConfig cfg{PatternType::Sequential, 3, min_id, DataType::f32, 4, 1000};
    generate_input_file(path_, cfg);
    auto lines = read_lines();
    ASSERT_EQ(4u, lines.size());
    EXPECT_EQ(0u, lines[1].find("100 : [ "));
    EXPECT_EQ(0u, lines[2].find("101 : [ "));
    EXPECT_EQ(0u, lines[3].find("102 : [ "));
}

TEST_F(InputGeneratorTest, IdsAreSequential) {
    GeneratorConfig cfg{PatternType::Sequential, 4, 0, DataType::f32, 4, 1000};
    generate_input_file(path_, cfg);
    auto lines = read_lines();
    ASSERT_EQ(5u, lines.size());
    for (size_t i = 0; i < 4; ++i) {
        std::string expected_prefix = std::to_string(i) + " : [ ";
        EXPECT_EQ(0u, lines[i + 1].find(expected_prefix)) << "line " << i + 1;
    }
}

TEST_F(InputGeneratorTest, ValueIsIdPlusPointOne) {
    GeneratorConfig cfg{PatternType::Sequential, 1, 7, DataType::f32, 4, 1000};
    generate_input_file(path_, cfg);
    auto lines = read_lines();
    ASSERT_EQ(2u, lines.size());
    // id=7, value=7.1 -> printed as %f -> "7.1"
    EXPECT_NE(std::string::npos, lines[1].find("7.1"));
}

TEST_F(InputGeneratorTest, WritesExactlyDimValuesPerVector) {
    const size_t dim = 4;
    GeneratorConfig cfg{PatternType::Sequential, 1, 5, DataType::f32, dim, 1000};
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

TEST_F(InputGeneratorTest, VectorLineFormat) {
    GeneratorConfig cfg{PatternType::Sequential, 1, 3, DataType::f32, 4, 1000};
    generate_input_file(path_, cfg);
    auto lines = read_lines();
    ASSERT_EQ(2u, lines.size());
    EXPECT_EQ("3 : [ 3.10, 3.10, 3.10, 3.10 ]", lines[1]);
}

// i16 tests

TEST_F(InputGeneratorTest, I16SuccessReturnCode) {
    GeneratorConfig cfg{PatternType::Sequential, 3, 0, DataType::i16, 4, 1000};
    Ret ret = generate_input_file(path_, cfg);
    EXPECT_EQ(0, ret.code());
}

TEST_F(InputGeneratorTest, HeaderLineI16) {
    GeneratorConfig cfg{PatternType::Sequential, 1, 0, DataType::i16, 4, 1000};
    generate_input_file(path_, cfg);
    auto lines = read_lines();
    ASSERT_FALSE(lines.empty());
    EXPECT_EQ("i16,4", lines[0]);
}

TEST_F(InputGeneratorTest, I16ValueIsId) {
    GeneratorConfig cfg{PatternType::Sequential, 1, 9, DataType::i16, 4, 1000};
    generate_input_file(path_, cfg);
    auto lines = read_lines();
    ASSERT_EQ(2u, lines.size());
    EXPECT_EQ("9 : [ 9, 9, 9, 9 ]", lines[1]);
}

TEST_F(InputGeneratorTest, I16WritesExactlyDimValuesPerVector) {
    const size_t dim = 4;
    GeneratorConfig cfg{PatternType::Sequential, 1, 5, DataType::i16, dim, 1000};
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

// i16 detailed tests

TEST_F(InputGeneratorTest, DetailedI16SuccessReturnCode) {
    GeneratorConfig cfg{PatternType::Detailed, 3, 0, DataType::i16, 4, 1000};
    Ret ret = generate_input_file(path_, cfg);
    EXPECT_EQ(0, ret.code());
}

TEST_F(InputGeneratorTest, DetailedHeaderLineI16) {
    GeneratorConfig cfg{PatternType::Detailed, 1, 0, DataType::i16, 4, 1000};
    generate_input_file(path_, cfg);
    auto lines = read_lines();
    ASSERT_FALSE(lines.empty());
    EXPECT_EQ("i16,4", lines[0]);
}

TEST_F(InputGeneratorTest, BinarySequentialHeaderAddsBinMarker) {
    GeneratorConfig cfg{PatternType::Sequential, 1, 7, DataType::f32, 4, 1000, 0, true};
    ASSERT_EQ(0, generate_input_file(path_, cfg).code());
    EXPECT_EQ("f32,4,bin", read_binary_header());
}

TEST_F(InputGeneratorTest, BinarySequentialWritesIdAndVectorPayload) {
    GeneratorConfig cfg{PatternType::Sequential, 1, 7, DataType::f32, 4, 1000, 0, true};
    ASSERT_EQ(0, generate_input_file(path_, cfg).code());

    std::ifstream in(path_, std::ios::binary);
    std::string header;
    std::getline(in, header);
    ASSERT_EQ("f32,4,bin", header);

    uint64_t id = 0;
    std::array<float, 4> vec {};
    in.read(reinterpret_cast<char*>(&id), sizeof(id));
    in.read(reinterpret_cast<char*>(vec.data()), sizeof(vec));

    ASSERT_TRUE(in.good());
    EXPECT_EQ(7u, id);
    EXPECT_FLOAT_EQ(7.1f, vec[0]);
    EXPECT_FLOAT_EQ(7.1f, vec[3]);
}

TEST_F(InputGeneratorTest, BinarySequentialLargeFilePreservesChunkBoundaryRecords) {
    constexpr size_t kCount = 20050;
    constexpr size_t kMinId = 100;
    GeneratorConfig cfg{PatternType::Sequential, kCount, kMinId, DataType::i16, 4, 1000, 0, true};
    ASSERT_EQ(0, generate_input_file(path_, cfg).code());

    const std::string header = "i16,4,bin\n";
    const std::streamoff record_size =
        static_cast<std::streamoff>(sizeof(uint64_t) + cfg.dim * sizeof(int16_t));

    std::ifstream in(path_, std::ios::binary | std::ios::ate);
    ASSERT_TRUE(in.is_open());
    const std::streamoff file_size = in.tellg();
    ASSERT_NE(-1, file_size);
    EXPECT_EQ(static_cast<std::streamoff>(header.size()) + static_cast<std::streamoff>(kCount) * record_size,
        file_size);

    auto expect_record = [&](size_t index, uint64_t expected_id) {
        in.seekg(static_cast<std::streamoff>(header.size()) +
                 static_cast<std::streamoff>(index) * record_size,
            std::ios::beg);
        ASSERT_TRUE(in.good());

        uint64_t id = 0;
        std::array<int16_t, 4> vec {};
        in.read(reinterpret_cast<char*>(&id), sizeof(id));
        in.read(reinterpret_cast<char*>(vec.data()), sizeof(vec));

        ASSERT_TRUE(in.good());
        EXPECT_EQ(expected_id, id);
        const std::array<int16_t, 4> expected_vec {
            static_cast<int16_t>(expected_id),
            static_cast<int16_t>(expected_id),
            static_cast<int16_t>(expected_id),
            static_cast<int16_t>(expected_id),
        };
        EXPECT_EQ(expected_vec, vec);
    };

    expect_record(0, kMinId);
    expect_record(9999, kMinId + 9999);
    expect_record(10000, kMinId + 10000);
    expect_record(kCount - 1, kMinId + kCount - 1);
}

TEST_F(InputGeneratorTest, BinarySequentialLargeF32FilePreservesChunkBoundaryRecords) {
    constexpr size_t kCount = 20050;
    constexpr size_t kMinId = 100;
    GeneratorConfig cfg{PatternType::Sequential, kCount, kMinId, DataType::f32, 4, 1000, 0, true};
    ASSERT_EQ(0, generate_input_file(path_, cfg).code());

    const std::string header = "f32,4,bin\n";
    const std::streamoff record_size =
        static_cast<std::streamoff>(sizeof(uint64_t) + cfg.dim * sizeof(float));

    std::ifstream in(path_, std::ios::binary | std::ios::ate);
    ASSERT_TRUE(in.is_open());
    const std::streamoff file_size = in.tellg();
    ASSERT_NE(-1, file_size);
    EXPECT_EQ(static_cast<std::streamoff>(header.size()) + static_cast<std::streamoff>(kCount) * record_size,
        file_size);

    auto expect_record = [&](size_t index, uint64_t expected_id) {
        in.seekg(static_cast<std::streamoff>(header.size()) +
                 static_cast<std::streamoff>(index) * record_size,
            std::ios::beg);
        ASSERT_TRUE(in.good());

        uint64_t id = 0;
        std::array<float, 4> vec {};
        in.read(reinterpret_cast<char*>(&id), sizeof(id));
        in.read(reinterpret_cast<char*>(vec.data()), sizeof(vec));

        ASSERT_TRUE(in.good());
        EXPECT_EQ(expected_id, id);
        const float expected_value = static_cast<float>(expected_id) + 0.1f;
        EXPECT_FLOAT_EQ(expected_value, vec[0]);
        EXPECT_FLOAT_EQ(expected_value, vec[3]);
    };

    expect_record(0, kMinId);
    expect_record(9999, kMinId + 9999);
    expect_record(10000, kMinId + 10000);
    expect_record(kCount - 1, kMinId + kCount - 1);
}

TEST_F(InputGeneratorTest, BinarySequentialF16WritesIdAndVectorPayloadWhenSupported) {


    GeneratorConfig cfg{PatternType::Sequential, 1, 7, DataType::f16, 4, 1000, 0, true};
    ASSERT_EQ(0, generate_input_file(path_, cfg).code());

    std::ifstream in(path_, std::ios::binary);
    std::string header;
    std::getline(in, header);
    ASSERT_EQ("f16,4,bin", header);

    uint64_t id = 0;
    std::array<float16, 4> vec {};
    in.read(reinterpret_cast<char*>(&id), sizeof(id));
    in.read(reinterpret_cast<char*>(vec.data()), sizeof(vec));

    ASSERT_TRUE(in.good());
    EXPECT_EQ(7u, id);
    const float expected = static_cast<float>(static_cast<float16>(7.1f));
    EXPECT_FLOAT_EQ(expected, static_cast<float>(vec[0]));
    EXPECT_FLOAT_EQ(expected, static_cast<float>(vec[3]));
}

TEST_F(InputGeneratorTest, BinaryDetailedWritesPerDimensionVariation) {
    GeneratorConfig cfg{PatternType::Detailed, 2, 20, DataType::i16, 4, 1000, 0, true};
    ASSERT_EQ(0, generate_input_file(path_, cfg).code());

    std::ifstream in(path_, std::ios::binary);
    std::string header;
    std::getline(in, header);
    ASSERT_EQ("i16,4,bin", header);

    uint64_t first_id = 0;
    std::array<int16_t, 4> first_vec {};
    uint64_t second_id = 0;
    std::array<int16_t, 4> second_vec {};
    in.read(reinterpret_cast<char*>(&first_id), sizeof(first_id));
    in.read(reinterpret_cast<char*>(first_vec.data()), sizeof(first_vec));
    in.read(reinterpret_cast<char*>(&second_id), sizeof(second_id));
    in.read(reinterpret_cast<char*>(second_vec.data()), sizeof(second_vec));

    ASSERT_TRUE(in.good());
    EXPECT_EQ(20u, first_id);
    EXPECT_EQ(21u, second_id);
    const std::array<int16_t, 4> expected_first {0, 0, 0, 0};
    const std::array<int16_t, 4> expected_second {1, 0, 0, 0};
    EXPECT_EQ(expected_first, first_vec);
    EXPECT_EQ(expected_second, second_vec);
}

TEST_F(InputGeneratorTest, DetailedI16LineCount) {
    const size_t count = 3;
    GeneratorConfig cfg{PatternType::Detailed, count, 7, DataType::i16, 4, 1000};
    generate_input_file(path_, cfg);
    auto lines = read_lines();
    EXPECT_EQ(count + 1, lines.size()); // header + one line per vector
}

TEST_F(InputGeneratorTest, DetailedI16ValueProgression) {
    GeneratorConfig cfg{PatternType::Detailed, 3, 9, DataType::i16, 4, 1000};
    generate_input_file(path_, cfg);
    auto lines = read_lines();
    ASSERT_EQ(4u, lines.size());
    EXPECT_EQ("9 : [ 0, 0, 0, 0 ]", lines[1]);
    EXPECT_EQ("10 : [ 1, 0, 0, 0 ]", lines[2]);
    EXPECT_EQ("11 : [ 2, 0, 0, 0 ]", lines[3]);
}

TEST_F(InputGeneratorTest, DetailedI16WritesExactlyDimValuesPerVector) {
    const size_t dim = 4;
    GeneratorConfig cfg{PatternType::Detailed, 2, 0, DataType::i16, dim, 1000};
    generate_input_file(path_, cfg);
    auto lines = read_lines();
    ASSERT_EQ(3u, lines.size());
    const std::string sep = ", ";
    const std::string& line = lines[2];
    size_t count = 0;
    size_t pos = 0;
    while ((pos = line.find(sep, pos)) != std::string::npos) {
        ++count;
        pos += sep.size();
    }
    EXPECT_EQ(dim - 1, count);
}

// --- deleted-item generation ---

TEST_F(InputGeneratorTest, SequentialEveryNDeletedWritesEmptyBrackets) {
    GeneratorConfig cfg{PatternType::Sequential, 6, 0, DataType::f32, 4, 1000, 2};
    ASSERT_EQ(0, generate_input_file(path_, cfg).code());
    auto lines = read_lines();
    ASSERT_EQ(7u, lines.size()); // header + 6 rows
    EXPECT_EQ("2 : []", lines[3]);
    EXPECT_EQ("4 : []", lines[5]);
}

TEST_F(InputGeneratorTest, SequentialEveryOneDoesNotDeleteFirstRow) {
    GeneratorConfig cfg{PatternType::Sequential, 4, 10, DataType::f32, 4, 1000, 1};
    ASSERT_EQ(0, generate_input_file(path_, cfg).code());
    auto lines = read_lines();
    ASSERT_EQ(5u, lines.size());
    EXPECT_NE(std::string::npos, lines[1].find("10 : [ "));
    EXPECT_EQ("11 : []", lines[2]);
    EXPECT_EQ("12 : []", lines[3]);
    EXPECT_EQ("13 : []", lines[4]);
}

TEST_F(InputGeneratorTest, DetailedEveryNDeletedKeepsNonDeletedProgression) {
    GeneratorConfig cfg{PatternType::Detailed, 5, 10, DataType::i16, 4, 1000, 2};
    ASSERT_EQ(0, generate_input_file(path_, cfg).code());
    auto lines = read_lines();
    ASSERT_EQ(6u, lines.size());
    EXPECT_EQ("10 : [ 0, 0, 0, 0 ]", lines[1]);
    EXPECT_EQ("11 : [ 1, 0, 0, 0 ]", lines[2]);
    EXPECT_EQ("12 : []", lines[3]);
    EXPECT_EQ("13 : [ 2, 0, 0, 0 ]", lines[4]);
    EXPECT_EQ("14 : []", lines[5]);
}

// --- manual-input generation ---

TEST_F(InputGeneratorTest, ManualFailsOnDimTooSmall) {
    ManualInputGenerator gen;
    gen.dim = 3;
    gen.add(1, 1);
    EXPECT_NE(0, generate_input_file(path_, gen).code());
}

TEST_F(InputGeneratorTest, ManualFailsOnDimTooLarge) {
    ManualInputGenerator gen;
    gen.dim = 4097;
    gen.add(1, 1);
    EXPECT_NE(0, generate_input_file(path_, gen).code());
}

TEST_F(InputGeneratorTest, ManualFailsOnBadPath) {
    ManualInputGenerator gen;
    gen.type = DataType::i16;
    gen.dim = 4;
    gen.add(10, 3);
    EXPECT_NE(0, generate_input_file("/nonexistent/dir/file.txt", gen).code());
}

TEST_F(InputGeneratorTest, ManualI16HeaderAndLineFormat) {
    ManualInputGenerator gen;
    gen.type = DataType::i16;
    gen.dim = 4;
    gen.add(9, 123);

    ASSERT_EQ(0, generate_input_file(path_, gen).code());
    auto lines = read_lines();
    ASSERT_EQ(2u, lines.size());
    EXPECT_EQ("i16,4", lines[0]);
    EXPECT_EQ("9 : [ 9, 9, 9, 9 ]", lines[1]);
}

TEST_F(InputGeneratorTest, ManualF32WritesIdPlusPointOneRepeatedByDim) {
    ManualInputGenerator gen;
    gen.type = DataType::f32;
    gen.dim = 4;
    gen.add(7, 1);

    ASSERT_EQ(0, generate_input_file(path_, gen).code());
    auto lines = read_lines();
    ASSERT_EQ(2u, lines.size());
    EXPECT_EQ("f32,4", lines[0]);
    EXPECT_EQ("7 : [ 7.10, 7.10, 7.10, 7.10 ]", lines[1]);
}

TEST_F(InputGeneratorTest, ManualF16HeaderAndFormatWhenSupported) {


    ManualInputGenerator gen;
    gen.type = DataType::f16;
    gen.dim = 4;
    gen.add(5, 1);

    ASSERT_EQ(0, generate_input_file(path_, gen).code());
    auto lines = read_lines();
    ASSERT_EQ(2u, lines.size());
    EXPECT_EQ("f16,4", lines[0]);
    EXPECT_EQ("5 : [ 5.10, 5.10, 5.10, 5.10 ]", lines[1]);
}

TEST_F(InputGeneratorTest, ManualDeletedWritesEmptyBrackets) {
    ManualInputGenerator gen;
    gen.type = DataType::i16;
    gen.dim = 4;
    gen.add(10, 1);
    gen.deleted(11);
    gen.add(12, 2);

    ASSERT_EQ(0, generate_input_file(path_, gen).code());
    auto lines = read_lines();
    ASSERT_EQ(4u, lines.size());
    EXPECT_EQ("10 : [ 10, 10, 10, 10 ]", lines[1]);
    EXPECT_EQ("11 : []", lines[2]);
    EXPECT_EQ("12 : [ 12, 12, 12, 12 ]", lines[3]);
}

TEST_F(InputGeneratorTest, ManualItemsAreWrittenInSortedIdOrder) {
    ManualInputGenerator gen;
    gen.type = DataType::i16;
    gen.dim = 4;
    gen.add(20, 1);
    gen.add(3, 1);
    gen.add(11, 1);

    ASSERT_EQ(0, generate_input_file(path_, gen).code());
    auto lines = read_lines();
    ASSERT_EQ(4u, lines.size());
    EXPECT_EQ("3 : [ 3, 3, 3, 3 ]", lines[1]);
    EXPECT_EQ("11 : [ 11, 11, 11, 11 ]", lines[2]);
    EXPECT_EQ("20 : [ 20, 20, 20, 20 ]", lines[3]);
}

// InputVector tests

TEST(InputVectorTest, FloatInitializesWithZeros) {
    InputVector<float> v(4, 10000.0f);
    const float* data = v.data();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(0.0f, data[i]) << "index " << i;
    }
}

TEST(InputVectorTest, FloatNext) {
    const size_t dim = 8;
    InputVector<float> v(dim, 10000.0f);

    v.next();
    const float* a = v.data();
    EXPECT_NEAR(0.01f, a[0], 1e-6f);
    for (size_t i = 1; i < dim; ++i) {
        EXPECT_FLOAT_EQ(0.0f, a[i]) << "index " << i;
    }

    v.next();
    const float* b = v.data();
    EXPECT_NEAR(0.02f, b[0], 1e-6f);
    for (size_t i = 1; i < dim; ++i) {
        EXPECT_FLOAT_EQ(0.0f, b[i]) << "index " << i;
    }

    // Continue with many steps to cover accumulation behavior.
    for (size_t i = 0; i < 198; ++i) {
        v.next();
    }

    const float* c = v.data();
    EXPECT_NEAR(2.0f, c[0], 1e-5f);
    for (size_t i = 1; i < dim; ++i) {
        EXPECT_FLOAT_EQ(0.0f, c[i]) << "index " << i;
    }
}

TEST(InputVectorTest, FloatNextRollover) {
    const size_t dim = 4;
    float rollover_val = 0.0f;
    for (size_t i = 0; i < 400; ++i) {
        rollover_val += 0.01f;
    }
    ASSERT_NEAR(4.0f, rollover_val, 1e-4f);

    InputVector<float> v(dim, rollover_val);

    for (size_t i = 0; i < 400; ++i) {
        v.next();
    }

    const float* after_first_rollover = v.data();
    EXPECT_NEAR(rollover_val, after_first_rollover[0], 1e-6f);
    EXPECT_FLOAT_EQ(0.0f, after_first_rollover[1]);
    EXPECT_FLOAT_EQ(0.0f, after_first_rollover[2]);
    EXPECT_FLOAT_EQ(0.0f, after_first_rollover[3]);

    v.next();
    const float* after_second_dim_step = v.data();
    EXPECT_NEAR(rollover_val, after_second_dim_step[0], 1e-6f);
    EXPECT_NEAR(0.01f, after_second_dim_step[1], 1e-6f);
    EXPECT_FLOAT_EQ(0.0f, after_second_dim_step[2]);
    EXPECT_FLOAT_EQ(0.0f, after_second_dim_step[3]);
}

TEST(InputVectorTest, I16InitializesWithZeros) {
    InputVector<int16_t> v(4, static_cast<int16_t>(1000));
    const int16_t* data = v.data();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(0, data[i]) << "index " << i;
    }
}

TEST(InputVectorTest, I16NextIncrementsFirstColumn) {
    InputVector<int16_t> v(3, static_cast<int16_t>(1000));

    v.next();
    const int16_t* a = v.data();
    EXPECT_EQ(1, a[0]);
    EXPECT_EQ(0, a[1]);
    EXPECT_EQ(0, a[2]);

    v.next();
    const int16_t* b = v.data();
    EXPECT_EQ(2, b[0]);
    EXPECT_EQ(0, b[1]);
    EXPECT_EQ(0, b[2]);
}

TEST(InputVectorTest, I16RolloverIncrementsOtherDimensions) {
    const size_t dim = 4;
    InputVector<int16_t> v(dim, static_cast<int16_t>(4));

    for (size_t i = 0; i < 4; ++i) {
        v.next();
    }

    const int16_t* after_first_rollover = v.data();
    EXPECT_EQ(4, after_first_rollover[0]);
    EXPECT_EQ(0, after_first_rollover[1]);
    EXPECT_EQ(0, after_first_rollover[2]);
    EXPECT_EQ(0, after_first_rollover[3]);

    v.next();
    const int16_t* after_second_dim_step = v.data();
    EXPECT_EQ(4, after_second_dim_step[0]);
    EXPECT_EQ(1, after_second_dim_step[1]);
    EXPECT_EQ(0, after_second_dim_step[2]);
    EXPECT_EQ(0, after_second_dim_step[3]);
}
