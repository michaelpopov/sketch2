// Unit tests for textual input generation helpers.

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <cstdio>
#include <unistd.h>
#include "core/storage/input_generator.h"
#include "core/utils/string_utils.h"
#include "utest_tmp_dir.h"

using namespace sketch2;

namespace {

struct F8BinaryRecord {
    uint64_t id = 0;
    std::vector<uint8_t> payload;
};

Ret parse_f8_generated_line(const std::string& line, size_t dim, std::vector<uint8_t>* output) {
    if (output == nullptr) {
        return Ret("f8 test output is null");
    }

    const size_t begin = line.find("[ ");
    const size_t end = line.rfind(" ]");
    if (begin == std::string::npos || end == std::string::npos || begin + 2 > end) {
        return Ret("f8 generated line has invalid vector syntax");
    }

    output->assign(dim, 0);
    return parse_vector(output->data(), output->size(), DataType::f8,
        static_cast<uint16_t>(dim), line.data() + begin + 2, line.data() + end);
}

Ret read_f8_binary_records(const std::string& path, size_t dim, std::string* header,
        std::vector<F8BinaryRecord>* records) {
    if (header == nullptr || records == nullptr) {
        return Ret("f8 binary test output is null");
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return Ret("failed to open f8 binary test input");
    }
    if (!std::getline(input, *header)) {
        return Ret("failed to read f8 binary test header");
    }

    records->clear();
    while (true) {
        F8BinaryRecord record;
        input.read(reinterpret_cast<char*>(&record.id), sizeof(record.id));
        if (input.eof()) {
            break;
        }
        if (!input) {
            return Ret("failed to read f8 binary test id");
        }

        record.payload.resize(dim);
        input.read(reinterpret_cast<char*>(record.payload.data()),
            static_cast<std::streamsize>(record.payload.size()));
        if (!input) {
            return Ret("failed to read f8 binary test payload");
        }
        records->push_back(record);
    }

    return Ret(0);
}

std::vector<uint8_t> f8_ordinal_bits(uint64_t ordinal, size_t dim) {
    std::vector<float8> values(dim);
    if (!float8_codebook::fill_ordinal_vector(ordinal, values.data(), dim)) {
        return {};
    }

    std::vector<uint8_t> bits;
    bits.reserve(dim);
    for (const float8 value : values) {
        bits.push_back(value.to_bits());
    }
    return bits;
}

bool is_canonical_f8_generator_byte(uint8_t bits) {
    const auto& codebook = float8_codebook::bits();
    return (bits & 0x7fU) < 0x7cU &&
        std::find(codebook.begin(), codebook.end(), bits) != codebook.end();
}

} // namespace

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

TEST(Float8CodebookTest, ContainsExactlyTheBoundedSortedE5M2Normals) {
    const auto& bits = float8_codebook::bits();
    ASSERT_EQ(float8_codebook::kSize, bits.size());
    EXPECT_EQ(0xcfU, bits.front());   // -28
    EXPECT_EQ(0xccU, bits[3]);        // -16
    EXPECT_EQ(0xacU, bits[35]);       // -0.0625
    EXPECT_EQ(0x2cU, bits[36]);       // +0.0625
    EXPECT_EQ(0x3cU, bits[52]);       // +1
    EXPECT_EQ(0x4fU, bits.back());    // +28

    std::set<uint8_t> unique_bits;
    float previous = -std::numeric_limits<float>::infinity();
    for (uint8_t bit : bits) {
        const float value = static_cast<float>(float8::from_bits(bit));
        EXPECT_TRUE(std::isfinite(value));
        EXPECT_GE(value, -28.0f);
        EXPECT_LE(value, 28.0f);
        EXPECT_GE(std::fabs(value), 0.0625f);
        EXPECT_LT(previous, value);
        EXPECT_LT((bit & 0x7fU), 0x7cU);
        EXPECT_GE((bit >> 2) & 0x1fU, 11U);
        EXPECT_LE((bit >> 2) & 0x1fU, 19U);
        previous = value;
        unique_bits.insert(bit);
    }
    EXPECT_EQ(float8_codebook::kSize, unique_bits.size());

    for (uint8_t sign : {uint8_t{0x00}, uint8_t{0x80}}) {
        for (uint8_t exponent = 11; exponent <= 19; ++exponent) {
            for (uint8_t mantissa = 0; mantissa <= 3; ++mantissa) {
                const uint8_t expected = static_cast<uint8_t>(
                    sign | (exponent << 2) | mantissa);
                EXPECT_NE(unique_bits.end(), unique_bits.find(expected));
            }
        }
    }
}

TEST(Float8CodebookTest, MapsBase72OrdinalsAndChecksCapacityWithoutOverflow) {
    uint64_t capacity = 0;
    ASSERT_TRUE(float8_codebook::capacity(0, &capacity));
    EXPECT_EQ(1U, capacity);
    ASSERT_TRUE(float8_codebook::capacity(1, &capacity));
    EXPECT_EQ(72U, capacity);
    ASSERT_TRUE(float8_codebook::capacity(2, &capacity));
    EXPECT_EQ(5184U, capacity);
    ASSERT_TRUE(float8_codebook::capacity(10, &capacity));
    EXPECT_FALSE(float8_codebook::capacity(11, &capacity));
    EXPECT_EQ(0U, capacity);

    EXPECT_TRUE(float8_codebook::range_fits(1, 72));
    EXPECT_FALSE(float8_codebook::range_fits(1, 73));
    EXPECT_TRUE(float8_codebook::range_fits(2, 5184));
    EXPECT_FALSE(float8_codebook::range_fits(2, 5185));
    EXPECT_TRUE(float8_codebook::range_fits(11, std::numeric_limits<uint64_t>::max()));

    auto expect_ordinal = [](uint64_t ordinal, const std::array<uint8_t, 3>& expected) {
        std::array<float8, 3> values {};
        ASSERT_TRUE(float8_codebook::fill_ordinal_vector(ordinal, values.data(), values.size()));
        std::array<uint8_t, 3> actual {};
        for (size_t d = 0; d < values.size(); ++d) {
            actual[d] = values[d].to_bits();
        }
        EXPECT_EQ(expected, actual);
    };

    expect_ordinal(0, {0xcfU, 0xcfU, 0xcfU});
    expect_ordinal(71, {0x4fU, 0xcfU, 0xcfU});
    expect_ordinal(72, {0xcfU, 0xceU, 0xcfU});
    expect_ordinal(72U * 72U - 1U, {0x4fU, 0x4fU, 0xcfU});

    std::array<float8, 1> unchanged {float8::from_bits(0x55)};
    EXPECT_FALSE(float8_codebook::fill_ordinal_vector(72, unchanged.data(), unchanged.size()));
    EXPECT_EQ(0x55U, unchanged[0].to_bits());
}

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

TEST_F(InputGeneratorTest, DummyMetadataWritesHeaderAndPeriodicColumns) {
    ASSERT_EQ(0, generate_dummy_metadata(path_, 20).code());

    auto lines = read_lines();
    ASSERT_EQ(21u, lines.size());
    EXPECT_EQ("id,aaa,bbb,ccc,text", lines[0]);
    EXPECT_EQ("0,0,0,0,\"aaa=0, bbb=0, ccc=0\"", lines[1]);
    EXPECT_EQ("1,1,1,1,\"aaa=1, bbb=1, ccc=1\"", lines[2]);
    EXPECT_EQ("2,0,2,2,\"aaa=0, bbb=2, ccc=2\"", lines[3]);
    EXPECT_EQ("4,0,4,4,\"aaa=0, bbb=4, ccc=4\"", lines[5]);
    EXPECT_EQ("5,1,0,5,\"aaa=1, bbb=0, ccc=5\"", lines[6]);
    EXPECT_EQ("9,1,4,9,\"aaa=1, bbb=4, ccc=9\"", lines[10]);
    EXPECT_EQ("10,0,0,0,\"aaa=0, bbb=0, ccc=0\"", lines[11]);
    EXPECT_EQ("11,1,1,1,\"aaa=1, bbb=1, ccc=1\"", lines[12]);
    EXPECT_EQ("14,0,4,4,\"aaa=0, bbb=4, ccc=4\"", lines[15]);
    EXPECT_EQ("15,1,0,5,\"aaa=1, bbb=0, ccc=5\"", lines[16]);
    EXPECT_EQ("19,1,4,9,\"aaa=1, bbb=4, ccc=9\"", lines[20]);
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

TEST_F(InputGeneratorTest, F16SequentialTextUsesBoundedMultidimensionalPayload) {
    GeneratorConfig cfg{PatternType::Sequential, 1, 7, DataType::f16, 4, 1000};
    ASSERT_EQ(0, generate_input_file(path_, cfg).code());

    auto lines = read_lines();
    ASSERT_EQ(2u, lines.size());
    EXPECT_EQ("7 : [ 8.00, 5.00, -3.00, -2.00 ]", lines[1]);
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

TEST_F(InputGeneratorTest, DotCompatibleTextUsesMonotonicPositivePayload) {
    GeneratorConfig cfg{PatternType::DotCompatible, 2, 3, DataType::f32, 4, 1000};
    ASSERT_EQ(0, generate_input_file(path_, cfg).code());
    auto lines = read_lines();
    ASSERT_EQ(3u, lines.size());
    EXPECT_EQ("3 : [ 3.10, 3.10, 3.10, 3.10 ]", lines[1]);
    EXPECT_EQ("4 : [ 4.10, 4.10, 4.10, 4.10 ]", lines[2]);
}

TEST_F(InputGeneratorTest, PerfTestTextWritesDenseDeterministicPayload) {
    GeneratorConfig cfg{PatternType::PerfTest, 2, 3, DataType::f32, 4, 1000};
    ASSERT_EQ(0, generate_input_file(path_, cfg).code());
    auto lines = read_lines();
    ASSERT_EQ(3u, lines.size());
    EXPECT_EQ("f32,4", lines[0]);
    EXPECT_EQ(0u, lines[1].find("3 : [ "));
    EXPECT_EQ(0u, lines[2].find("4 : [ "));
    EXPECT_EQ(std::string::npos, lines[1].find("3.10, 3.10, 3.10, 3.10"));
    EXPECT_NE(lines[1], lines[2]);
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

TEST_F(InputGeneratorTest, I16SequentialTextUsesBoundedMultidimensionalPayload) {
    GeneratorConfig cfg{PatternType::Sequential, 1, 9, DataType::i16, 4, 1000};
    generate_input_file(path_, cfg);
    auto lines = read_lines();
    ASSERT_EQ(2u, lines.size());
    EXPECT_EQ("9 : [ 10, 0, 0, 0 ]", lines[1]);
}

TEST_F(InputGeneratorTest, I16WritesExactlyDimValuesPerVector) {
    const size_t dim = 4;
    GeneratorConfig cfg{PatternType::Sequential, 1, 5, DataType::i16, dim, 1000};
    generate_input_file(path_, cfg);
    auto lines = read_lines();
    ASSERT_EQ(2u, lines.size());
    // The bounded payload still emits exactly one value per dimension.
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

TEST_F(InputGeneratorTest, BinaryDotCompatibleHeaderAddsBinMarker) {
    GeneratorConfig cfg{PatternType::DotCompatible, 1, 7, DataType::f32, 4, 1000, 0, true};
    ASSERT_EQ(0, generate_input_file(path_, cfg).code());
    EXPECT_EQ("f32,4,bin", read_binary_header());
}

TEST_F(InputGeneratorTest, BinaryPerfTestWritesBoundedPerDimensionPayload) {
    GeneratorConfig cfg{PatternType::PerfTest, 2, 7, DataType::f32, 4, 1000, 0, true};
    ASSERT_EQ(0, generate_input_file(path_, cfg).code());

    std::ifstream in(path_, std::ios::binary);
    std::string header;
    std::getline(in, header);
    ASSERT_EQ("f32,4,bin", header);

    uint64_t first_id = 0;
    std::array<float, 4> first_vec {};
    uint64_t second_id = 0;
    std::array<float, 4> second_vec {};
    in.read(reinterpret_cast<char*>(&first_id), sizeof(first_id));
    in.read(reinterpret_cast<char*>(first_vec.data()), sizeof(first_vec));
    in.read(reinterpret_cast<char*>(&second_id), sizeof(second_id));
    in.read(reinterpret_cast<char*>(second_vec.data()), sizeof(second_vec));

    ASSERT_TRUE(in.good());
    EXPECT_EQ(7u, first_id);
    EXPECT_EQ(8u, second_id);
    EXPECT_NE(first_vec[0], first_vec[1]);
    EXPECT_NE(first_vec, second_vec);
    for (float value : first_vec) {
        EXPECT_LE(std::abs(value), 4.0f);
    }
}

TEST_F(InputGeneratorTest, BinaryPerfTestLargeFilePreservesChunkBoundaryRecords) {
    constexpr size_t kCount = 20050;
    constexpr size_t kMinId = 100;
    GeneratorConfig cfg{PatternType::PerfTest, kCount, kMinId, DataType::f32, 4, 1000, 0, true};
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
        EXPECT_NE(vec[0], vec[1]);
        for (float value : vec) {
            EXPECT_LE(std::abs(value), 4.0f);
        }
    };

    expect_record(0, kMinId);
    expect_record(9999, kMinId + 9999);
    expect_record(10000, kMinId + 10000);
    expect_record(kCount - 1, kMinId + kCount - 1);
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
        EXPECT_GE(vec[0], 1);
        EXPECT_LE(vec[0], 17);
        EXPECT_GE(vec[1], -5);
        EXPECT_LE(vec[1], 5);
        EXPECT_GE(vec[2], -3);
        EXPECT_LE(vec[2], 3);
        EXPECT_GE(vec[3], -2);
        EXPECT_LE(vec[3], 2);
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

TEST_F(InputGeneratorTest, BinarySequentialF16WritesBoundedVectorPayload) {
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
    const std::array<float, 4> expected {8.0f, 5.0f, -3.0f, -2.0f};
    for (size_t index = 0; index < expected.size(); ++index) {
        EXPECT_FLOAT_EQ(expected[index], static_cast<float>(vec[index]));
    }
}

TEST_F(InputGeneratorTest, BinarySequentialF16StaysFinitePastScalarLimit) {
    GeneratorConfig cfg{PatternType::Sequential, 2, 65520, DataType::f16, 4, 1000, 0, true};
    ASSERT_EQ(0, generate_input_file(path_, cfg).code());

    std::ifstream in(path_, std::ios::binary);
    std::string header;
    std::getline(in, header);
    ASSERT_EQ("f16,4,bin", header);

    uint64_t first_id = 0;
    std::array<float16, 4> first_vec {};
    uint64_t second_id = 0;
    std::array<float16, 4> second_vec {};
    in.read(reinterpret_cast<char*>(&first_id), sizeof(first_id));
    in.read(reinterpret_cast<char*>(first_vec.data()), sizeof(first_vec));
    in.read(reinterpret_cast<char*>(&second_id), sizeof(second_id));
    in.read(reinterpret_cast<char*>(second_vec.data()), sizeof(second_vec));

    ASSERT_TRUE(in.good());
    EXPECT_EQ(65520u, first_id);
    EXPECT_EQ(65521u, second_id);
    const std::array<float, 4> expected_first {3.0f, -4.0f, -3.0f, 1.0f};
    const std::array<float, 4> expected_second {4.0f, -1.0f, 2.0f, 2.0f};
    for (size_t index = 0; index < expected_first.size(); ++index) {
        EXPECT_TRUE(std::isfinite(static_cast<float>(first_vec[index])));
        EXPECT_TRUE(std::isfinite(static_cast<float>(second_vec[index])));
        EXPECT_FLOAT_EQ(expected_first[index], static_cast<float>(first_vec[index]));
        EXPECT_FLOAT_EQ(expected_second[index], static_cast<float>(second_vec[index]));
    }
}

TEST_F(InputGeneratorTest, BinarySequentialI16AvoidsNarrowingPastInt16Limit) {
    GeneratorConfig cfg{PatternType::Sequential, 2, 32768, DataType::i16, 4, 1000, 0, true};
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
    EXPECT_EQ(32768u, first_id);
    EXPECT_EQ(32769u, second_id);
    EXPECT_EQ((std::array<int16_t, 4> {10, 3, 2, -1}), first_vec);
    EXPECT_EQ((std::array<int16_t, 4> {11, -5, 0, 0}), second_vec);
}

TEST_F(InputGeneratorTest, CosCompatibleI16WritesBoundedPayloadInTextAndBinary) {
    GeneratorConfig cfg {PatternType::CosCompatible, 2, 7, DataType::i16, 4, 1000};
    ASSERT_EQ(0, generate_input_file(path_, cfg).code());

    const auto lines = read_lines();
    ASSERT_EQ(3U, lines.size());
    EXPECT_EQ("i16,4", lines[0]);
    EXPECT_EQ("7 : [ 8, 5, -3, -2 ]", lines[1]);
    EXPECT_EQ("8 : [ 9, -3, 2, -1 ]", lines[2]);

    cfg.binary = true;
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
    EXPECT_EQ(7U, first_id);
    EXPECT_EQ(8U, second_id);
    EXPECT_EQ((std::array<int16_t, 4> {8, 5, -3, -2}), first_vec);
    EXPECT_EQ((std::array<int16_t, 4> {9, -3, 2, -1}), second_vec);
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

TEST_F(InputGeneratorTest, F8SequentialTextAndBinaryAreLosslessBase72Parity) {
    constexpr size_t kDim = 4;
    constexpr size_t kCount = 73;
    const size_t min_id = std::numeric_limits<size_t>::max() - 80;
    GeneratorConfig config {PatternType::Sequential, kCount, min_id, DataType::f8, kDim, 1000};

    ASSERT_EQ(0, generate_input_file(path_, config).code());
    const auto lines = read_lines();
    ASSERT_EQ(kCount + 1, lines.size());
    EXPECT_EQ("f8,4", lines[0]);

    std::vector<std::vector<uint8_t>> text_vectors;
    text_vectors.reserve(kCount);
    for (size_t ordinal = 0; ordinal < kCount; ++ordinal) {
        EXPECT_EQ(0U, lines[ordinal + 1].find(std::to_string(min_id + ordinal) + " : [ "));
        std::vector<uint8_t> payload;
        ASSERT_EQ(0, parse_f8_generated_line(lines[ordinal + 1], kDim, &payload).code());
        EXPECT_EQ(f8_ordinal_bits(ordinal, kDim), payload);
        for (uint8_t bit : payload) {
            EXPECT_TRUE(is_canonical_f8_generator_byte(bit));
        }
        text_vectors.push_back(payload);
    }
    const std::set<std::vector<uint8_t>> unique_vectors(text_vectors.begin(), text_vectors.end());
    EXPECT_EQ(kCount, unique_vectors.size());
    // This entry would be rendered as 0.06 by the presentation helper; its
    // presence proves generator text uses the lossless %.9g path.
    EXPECT_NE(std::string::npos, lines[37].find("0.0625"));

    config.binary = true;
    ASSERT_EQ(0, generate_input_file(path_, config).code());
    std::string header;
    std::vector<F8BinaryRecord> binary_records;
    ASSERT_EQ(0, read_f8_binary_records(path_, kDim, &header, &binary_records).code());
    EXPECT_EQ("f8,4,bin", header);
    ASSERT_EQ(kCount, binary_records.size());
    for (size_t ordinal = 0; ordinal < kCount; ++ordinal) {
        EXPECT_EQ(static_cast<uint64_t>(min_id + ordinal), binary_records[ordinal].id);
        EXPECT_EQ(text_vectors[ordinal], binary_records[ordinal].payload);
        for (uint8_t bit : binary_records[ordinal].payload) {
            EXPECT_TRUE(is_canonical_f8_generator_byte(bit));
        }
    }
}

TEST_F(InputGeneratorTest, F8SequentialRejectsRangesBeyondBase72Capacity) {
    constexpr size_t kDim = 4;
    constexpr size_t kCapacity = 72U * 72U * 72U * 72U;
    GeneratorConfig config {
        PatternType::Sequential, kCapacity + 1, 0, DataType::f8, kDim, 1000};

    const Ret ret = generate_input_file(path_, config);
    EXPECT_NE(0, ret.code());
    EXPECT_NE(std::string::npos, ret.message().find("72^dim"));
}

TEST_F(InputGeneratorTest, F8RejectsIdRangesThatWouldWrapBeforeReplacingOutput) {
    constexpr size_t kDim = 4;
    const std::array<PatternType, 5> patterns {
        PatternType::Sequential,
        PatternType::Detailed,
        PatternType::CosCompatible,
        PatternType::DotCompatible,
        PatternType::PerfTest,
    };

    for (const bool binary : {false, true}) {
        for (const PatternType pattern : patterns) {
            SCOPED_TRACE(static_cast<int>(pattern));
            SCOPED_TRACE(binary);
            GeneratorConfig config {pattern, 2,
                std::numeric_limits<size_t>::max() - 1, DataType::f8, kDim, 1000};
            config.binary = binary;

            ASSERT_EQ(0, generate_input_file(path_, config).code());
            if (binary) {
                std::string header;
                std::vector<F8BinaryRecord> records;
                ASSERT_EQ(0, read_f8_binary_records(path_, kDim, &header, &records).code());
                ASSERT_EQ(2U, records.size());
                EXPECT_EQ(std::numeric_limits<size_t>::max() - 1, records[0].id);
                EXPECT_EQ(std::numeric_limits<size_t>::max(), records[1].id);
            } else {
                const auto lines = read_lines();
                ASSERT_EQ(3U, lines.size());
                EXPECT_EQ(0U, lines[1].find(
                    std::to_string(std::numeric_limits<size_t>::max() - 1) + " : [ "));
                EXPECT_EQ(0U, lines[2].find(
                    std::to_string(std::numeric_limits<size_t>::max()) + " : [ "));
            }

            {
                std::ofstream output(path_, std::ios::binary | std::ios::trunc);
                ASSERT_TRUE(output.is_open());
                output << "preserve existing output";
            }

            config.count = 3;
            const Ret ret = generate_input_file(path_, config);
            EXPECT_NE(0, ret.code());
            EXPECT_NE(std::string::npos, ret.message().find("ID range overflow"));
            EXPECT_EQ("preserve existing output", read_file_bytes());
        }
    }
}

TEST_F(InputGeneratorTest, F8SequentialAndManualTombstonesUseLiveSortedOrdinals) {
    constexpr size_t kDim = 4;
    auto expect_ordinal = [&](const std::string& line, uint64_t ordinal) {
        std::vector<uint8_t> payload;
        ASSERT_EQ(0, parse_f8_generated_line(line, kDim, &payload).code());
        EXPECT_EQ(f8_ordinal_bits(ordinal, kDim), payload);
        for (uint8_t bit : payload) {
            EXPECT_TRUE(is_canonical_f8_generator_byte(bit));
        }
    };

    GeneratorConfig sequential {
        PatternType::Sequential, 5, 10, DataType::f8, kDim, 1000, 2};
    ASSERT_EQ(0, generate_input_file(path_, sequential).code());
    auto lines = read_lines();
    ASSERT_EQ(6U, lines.size());
    expect_ordinal(lines[1], 0);
    expect_ordinal(lines[2], 1);
    EXPECT_EQ("12 : []", lines[3]);
    expect_ordinal(lines[4], 2);
    EXPECT_EQ("14 : []", lines[5]);

    ManualInputGenerator manual;
    manual.type = DataType::f8;
    manual.dim = kDim;
    manual.add(20, 1);
    manual.deleted(11);
    manual.add(10, 1);
    manual.add(12, 1);

    ASSERT_EQ(0, generate_input_file(path_, manual).code());
    lines = read_lines();
    ASSERT_EQ(5U, lines.size());
    EXPECT_EQ("f8,4", lines[0]);
    EXPECT_EQ(0U, lines[1].find("10 : [ "));
    expect_ordinal(lines[1], 0);
    EXPECT_EQ("11 : []", lines[2]);
    EXPECT_EQ(0U, lines[3].find("12 : [ "));
    expect_ordinal(lines[3], 1);
    EXPECT_EQ(0U, lines[4].find("20 : [ "));
    expect_ordinal(lines[4], 2);
}

TEST_F(InputGeneratorTest, F8AllPatternDispatchesProduceCanonicalTextBinaryParity) {
    constexpr size_t kDim = 4;
    const std::array<PatternType, 5> patterns {
        PatternType::Sequential,
        PatternType::Detailed,
        PatternType::CosCompatible,
        PatternType::DotCompatible,
        PatternType::PerfTest,
    };

    for (const PatternType pattern : patterns) {
        SCOPED_TRACE(static_cast<int>(pattern));
        GeneratorConfig config {pattern, 4, 17, DataType::f8, kDim, 1000};
        ASSERT_EQ(0, generate_input_file(path_, config).code());
        const auto lines = read_lines();
        ASSERT_EQ(config.count + 1, lines.size());
        EXPECT_EQ("f8,4", lines[0]);

        std::vector<std::vector<uint8_t>> text_vectors;
        text_vectors.reserve(config.count);
        for (size_t index = 0; index < config.count; ++index) {
            std::vector<uint8_t> payload;
            ASSERT_EQ(0, parse_f8_generated_line(lines[index + 1], kDim, &payload).code());
            ASSERT_EQ(kDim, payload.size());
            for (uint8_t bit : payload) {
                EXPECT_TRUE(is_canonical_f8_generator_byte(bit));
            }
            text_vectors.push_back(payload);
        }

        if (pattern == PatternType::Sequential || pattern == PatternType::DotCompatible ||
                pattern == PatternType::CosCompatible) {
            EXPECT_EQ(f8_ordinal_bits(0, kDim), text_vectors[0]);
            EXPECT_EQ(f8_ordinal_bits(1, kDim), text_vectors[1]);
        }

        config.binary = true;
        ASSERT_EQ(0, generate_input_file(path_, config).code());
        std::string header;
        std::vector<F8BinaryRecord> binary_records;
        ASSERT_EQ(0, read_f8_binary_records(path_, kDim, &header, &binary_records).code());
        EXPECT_EQ("f8,4,bin", header);
        ASSERT_EQ(config.count, binary_records.size());
        for (size_t index = 0; index < config.count; ++index) {
            EXPECT_EQ(static_cast<uint64_t>(config.min_id + index), binary_records[index].id);
            EXPECT_EQ(text_vectors[index], binary_records[index].payload);
            for (uint8_t bit : binary_records[index].payload) {
                EXPECT_TRUE(is_canonical_f8_generator_byte(bit));
            }
        }
    }
}

// InputVector tests

TEST(InputVectorTest, Float8UsesBoundedCodebookProgressionAndReset) {
    EXPECT_EQ(0U, float8_codebook::upper_bound_index(-100.0f));
    EXPECT_EQ(35U, float8_codebook::upper_bound_index(0.0f));
    EXPECT_EQ(float8_codebook::kSize - 1, float8_codebook::upper_bound_index(1000.0f));

    InputVector<float8> v(2, -24.0f);
    ASSERT_EQ(0xcfU, v.data()[0].to_bits());
    ASSERT_EQ(0xcfU, v.data()[1].to_bits());

    v.next();
    EXPECT_EQ(0xceU, v.data()[0].to_bits());
    EXPECT_EQ(0xcfU, v.data()[1].to_bits());

    v.next();
    EXPECT_EQ(0xceU, v.data()[0].to_bits());
    EXPECT_EQ(0xceU, v.data()[1].to_bits());

    // The call after the final column completes mirrors the existing
    // InputVector reset timing and returns to the first codebook entry.
    v.next();
    EXPECT_EQ(0xcfU, v.data()[0].to_bits());
    EXPECT_EQ(0xcfU, v.data()[1].to_bits());

    InputVector<float8> zero_bound(1, 0.0f);
    for (size_t step = 0; step < 35; ++step) {
        zero_bound.next();
        EXPECT_TRUE(is_canonical_f8_generator_byte(zero_bound.data()[0].to_bits()));
    }
    // max_val == 0 selects the final negative codebook member, never an f8
    // signed zero, subnormal, Inf, or NaN.
    EXPECT_EQ(0xacU, zero_bound.data()[0].to_bits());
    zero_bound.next();
    EXPECT_EQ(0xcfU, zero_bound.data()[0].to_bits());
}

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
