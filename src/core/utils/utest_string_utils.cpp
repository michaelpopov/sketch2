// Unit tests for textual vector parsing and formatting helpers.

#include "string_utils.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <sys/mman.h>
#include <unistd.h>

#include <gtest/gtest.h>

namespace sketch2 {

namespace {

class GuardedPages {
public:
    explicit GuardedPages(size_t page_size)
        : page_size_(page_size),
          size_(page_size * 2),
          mapping_(mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)) {
    }

    ~GuardedPages() {
        if (mapping_ != MAP_FAILED) {
            munmap(mapping_, size_);
        }
    }

    bool valid() const {
        return mapping_ != MAP_FAILED;
    }

    char* write_at_first_page_end(const char* payload, size_t payload_size) {
        char* const end = first_page_end();
        char* const start = end - payload_size;
        std::memcpy(start, payload, payload_size);
        return start;
    }

    char* first_page_end() const {
        return static_cast<char*>(mapping_) + page_size_;
    }

    bool protect_second_page() const {
        return mprotect(static_cast<char*>(mapping_) + page_size_, page_size_, PROT_NONE) == 0;
    }

private:
    size_t page_size_;
    size_t size_;
    void* mapping_;
};

size_t system_page_size() {
    const long page_size = sysconf(_SC_PAGESIZE);
    return page_size > 0 ? static_cast<size_t>(page_size) : 0;
}

} // namespace

TEST(string_utils, parse_vector_f32_success) {
    std::array<float, 3> out {};
    const Ret ret = parse_vector(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::f32, out.size(),
        "1.25, -2.5, 3");
    ASSERT_EQ(ret.code(), 0) << ret.message();
    EXPECT_FLOAT_EQ(out[0], 1.25f);
    EXPECT_FLOAT_EQ(out[1], -2.5f);
    EXPECT_FLOAT_EQ(out[2], 3.0f);
}

TEST(string_utils, parse_vector_i16_success) {
    std::array<int16_t, 4> out {};
    const Ret ret = parse_vector(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::i16, out.size(),
        "10, -20, 30, 40");
    ASSERT_EQ(ret.code(), 0) << ret.message();
    EXPECT_EQ(out[0], 10);
    EXPECT_EQ(out[1], -20);
    EXPECT_EQ(out[2], 30);
    EXPECT_EQ(out[3], 40);
}

TEST(string_utils, parse_vector_invalid_arguments) {
    std::array<float, 2> out {};
    EXPECT_NE(parse_vector(nullptr, sizeof(out), DataType::f32, out.size(), "1,2").code(), 0);
    EXPECT_NE(parse_vector(reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::f32,
                  out.size(), nullptr).code(), 0);
}

TEST(string_utils, parse_vector_unsupported_data_type_fails) {
    std::array<uint8_t, 4> out {};
    const Ret ret = parse_vector(
        out.data(), out.size(), static_cast<DataType>(99), 1, "");
    EXPECT_NE(ret.code(), 0);
    EXPECT_EQ("Unsupported dataset type", ret.message());
}

TEST(string_utils, parse_vector_buffer_too_small) {
    std::array<float, 2> out {};
    const Ret ret = parse_vector(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(float), DataType::f32, out.size(),
        "1,2");
    EXPECT_NE(ret.code(), 0);
}

TEST(string_utils, parse_vector_truncated_payload) {
    std::array<float, 3> out {};
    const Ret ret = parse_vector(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::f32, out.size(),
        "1,2");
    EXPECT_NE(ret.code(), 0);
}

TEST(string_utils, parse_vector_invalid_token) {
    std::array<float, 2> out {};
    const Ret ret = parse_vector(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::f32, out.size(),
        "1,abc");
    EXPECT_NE(ret.code(), 0);
}

TEST(string_utils, parse_vector_i16_out_of_range_fails) {
    std::array<int16_t, 2> out {};
    const Ret ret = parse_vector(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::i16, out.size(),
        "32768, 0");
    EXPECT_NE(ret.code(), 0);
    EXPECT_EQ("InputReader::data: i16 token out of range", ret.message());
}

TEST(string_utils, parse_vector_f32_nan_and_inf_fail) {
    std::array<float, 3> out {};
    const Ret ret = parse_vector(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::f32, out.size(),
        "nan, inf, -inf");
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("InputReader::data: non-finite f32 token", ret.message());
}


TEST(string_utils, parse_vector_extra_tokens) {
    std::array<float, 2> out {};
    const Ret ret = parse_vector(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::f32, out.size(),
        "1,2,3");
    EXPECT_NE(ret.code(), 0);
}

TEST(string_utils, parse_vector_respects_end_pointer) {
    std::array<float, 2> out {};
    const char* line = "1,2,3";
    const char* end = line + 3; // "1,2"
    const Ret ret = parse_vector(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::f32, out.size(),
        line, end);
    ASSERT_EQ(ret.code(), 0) << ret.message();
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
}

TEST(string_utils, parse_vector_end_pointer_cut_mid_token_fails) {
    std::array<float, 2> out {};
    const char* line = "1,2";
    const char* end = line + 2; // "1,"
    const Ret ret = parse_vector(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::f32, out.size(),
        line, end);
    EXPECT_NE(ret.code(), 0);
}

TEST(string_utils, print_vector_f32_success) {
    std::array<float, 4> vec {1.0f, 2.5f, -3.0f, 4.25f};
    char out[128] {};
    const Ret ret = print_vector(reinterpret_cast<uint8_t*>(vec.data()), DataType::f32, vec.size(), out, sizeof(out), 0);
    ASSERT_EQ(0, ret.code()) << ret.message();
    EXPECT_STREQ("[ 1, 2.5, -3, 4.25 ]", out);
}

TEST(string_utils, print_vector_i16_success) {
    std::array<int16_t, 4> vec {10, -20, 30, 40};
    char out[128] {};
    const Ret ret = print_vector(reinterpret_cast<uint8_t*>(vec.data()), DataType::i16, vec.size(), out, sizeof(out));
    ASSERT_EQ(0, ret.code()) << ret.message();
    EXPECT_STREQ("[ 10, -20, 30, 40 ]", out);
}

TEST(string_utils, print_vector_buffer_too_small_fails) {
    std::array<float, 2> vec {1.0f, 2.0f};
    char out[4] {};
    const Ret ret = print_vector(reinterpret_cast<uint8_t*>(vec.data()), DataType::f32, vec.size(), out, sizeof(out));
    EXPECT_NE(0, ret.code());
}

TEST(string_utils, data_type_from_int_f8_and_invalid_values) {
    EXPECT_THROW(data_type_from_int(-1), std::runtime_error);
    EXPECT_EQ(DataType::f8, data_type_from_int(3));
    EXPECT_THROW(data_type_from_int(4), std::runtime_error);
    EXPECT_THROW(data_type_from_int(99), std::runtime_error);
}

TEST(string_utils, parse_vector_f32_scientific_notation) {
    // strtof accepts scientific notation; verify the parser does too.
    std::array<float, 3> out {};
    const Ret ret = parse_vector(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::f32, out.size(),
        "1.5e2, -2.5e-1, 3.0e0");
    ASSERT_EQ(ret.code(), 0) << ret.message();
    EXPECT_FLOAT_EQ(out[0], 150.0f);
    EXPECT_FLOAT_EQ(out[1], -0.25f);
    EXPECT_FLOAT_EQ(out[2], 3.0f);
}

TEST(string_utils, parse_vector_i16_min_value_succeeds) {
    // INT16_MIN (-32768) is the lower boundary and must be accepted.
    std::array<int16_t, 2> out {};
    const Ret ret = parse_vector(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::i16, out.size(),
        "-32768, 32767");
    ASSERT_EQ(ret.code(), 0) << ret.message();
    EXPECT_EQ(out[0], -32768);
    EXPECT_EQ(out[1], 32767);
}

TEST(string_utils, parse_vector_i16_respects_end_pointer_at_page_boundary) {
    const size_t page_size = system_page_size();
    ASSERT_GT(page_size, 0u);
    GuardedPages pages(page_size);
    ASSERT_TRUE(pages.valid());

    const char payload[] = "1,2";
    char* const line = pages.write_at_first_page_end(payload, sizeof(payload) - 1);
    ASSERT_TRUE(pages.protect_second_page());

    std::array<int16_t, 2> out {};
    const Ret ret = parse_vector(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::i16, out.size(),
        line, pages.first_page_end());
    ASSERT_EQ(ret.code(), 0) << ret.message();
    EXPECT_EQ(out[0], 1);
    EXPECT_EQ(out[1], 2);
}

TEST(string_utils, parse_vector_f32_leading_whitespace_succeeds) {
    // The parser skips separators (space/comma) between tokens; leading
    // whitespace is implicitly handled by strtof which skips leading spaces.
    std::array<float, 2> out {};
    const Ret ret = parse_vector(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::f32, out.size(),
        " 1.0,  2.0");
    ASSERT_EQ(ret.code(), 0) << ret.message();
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
}

TEST(string_utils, parse_vector_dist_func_invalid_string_throws) {
    EXPECT_THROW(dist_func_from_string("bad"), std::runtime_error);
    EXPECT_THROW(dist_func_from_string(""), std::runtime_error);
}

TEST(string_utils, parse_vector_data_type_roundtrip) {
    EXPECT_EQ(DataType::f32, data_type_from_string("f32"));
    EXPECT_EQ(DataType::f16, data_type_from_string("f16"));
    EXPECT_EQ(DataType::i16, data_type_from_string("i16"));
    EXPECT_EQ(DataType::f8, data_type_from_string("f8"));
    EXPECT_THROW(data_type_from_string("bad"), std::runtime_error);
    EXPECT_STREQ("f32", data_type_to_string(DataType::f32));
    EXPECT_STREQ("f16", data_type_to_string(DataType::f16));
    EXPECT_STREQ("i16", data_type_to_string(DataType::i16));
    EXPECT_STREQ("f8", data_type_to_string(DataType::f8));
}

TEST(string_utils, parse_vector_spaces_f32_success) {
    std::array<float, 3> out {};
    const Ret ret = parse_vector_spaces(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::f32, out.size(),
        "1.25 -2.5 3");
    ASSERT_EQ(ret.code(), 0) << ret.message();
    EXPECT_FLOAT_EQ(out[0], 1.25f);
    EXPECT_FLOAT_EQ(out[1], -2.5f);
    EXPECT_FLOAT_EQ(out[2], 3.0f);
}

TEST(string_utils, parse_vector_spaces_i16_success) {
    std::array<int16_t, 4> out {};
    const Ret ret = parse_vector_spaces(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::i16, out.size(),
        "10 -20 30 40");
    ASSERT_EQ(ret.code(), 0) << ret.message();
    EXPECT_EQ(out[0], 10);
    EXPECT_EQ(out[1], -20);
    EXPECT_EQ(out[2], 30);
    EXPECT_EQ(out[3], 40);
}

TEST(string_utils, parse_vector_spaces_invalid_arguments) {
    std::array<float, 2> out {};
    EXPECT_NE(parse_vector_spaces(nullptr, sizeof(out), DataType::f32, out.size(), "1 2").code(), 0);
    EXPECT_NE(parse_vector_spaces(reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::f32,
                  out.size(), nullptr).code(), 0);
}

TEST(string_utils, parse_vector_spaces_unsupported_data_type_fails) {
    std::array<uint8_t, 4> out {};
    const Ret ret = parse_vector_spaces(
        out.data(), out.size(), static_cast<DataType>(99), 1, "");
    EXPECT_NE(ret.code(), 0);
    EXPECT_EQ("Unsupported dataset type", ret.message());
}

TEST(string_utils, parse_vector_spaces_buffer_too_small) {
    std::array<float, 2> out {};
    const Ret ret = parse_vector_spaces(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(float), DataType::f32, out.size(),
        "1 2");
    EXPECT_NE(ret.code(), 0);
}

TEST(string_utils, parse_vector_spaces_truncated_payload) {
    std::array<float, 3> out {};
    const Ret ret = parse_vector_spaces(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::f32, out.size(),
        "1 2");
    EXPECT_NE(ret.code(), 0);
}

TEST(string_utils, parse_vector_spaces_invalid_token) {
    std::array<float, 2> out {};
    const Ret ret = parse_vector_spaces(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::f32, out.size(),
        "1 abc");
    EXPECT_NE(ret.code(), 0);
}

TEST(string_utils, parse_vector_spaces_i16_out_of_range_fails) {
    std::array<int16_t, 2> out {};
    const Ret ret = parse_vector_spaces(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::i16, out.size(),
        "32768 0");
    EXPECT_NE(ret.code(), 0);
    EXPECT_EQ("InputReader::data: i16 token out of range", ret.message());
}

TEST(string_utils, parse_vector_spaces_f32_nan_and_inf_fail) {
    std::array<float, 3> out {};
    const Ret ret = parse_vector_spaces(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::f32, out.size(),
        "nan inf -inf");
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("InputReader::data: non-finite f32 token", ret.message());
}

TEST(string_utils, parse_vector_spaces_extra_tokens) {
    std::array<float, 2> out {};
    const Ret ret = parse_vector_spaces(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::f32, out.size(),
        "1 2 3");
    EXPECT_NE(ret.code(), 0);
}

TEST(string_utils, parse_vector_spaces_respects_end_pointer) {
    std::array<float, 2> out {};
    const char* line = "1 2 3";
    const char* end = line + 3; // "1 2"
    const Ret ret = parse_vector_spaces(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::f32, out.size(),
        line, end);
    ASSERT_EQ(ret.code(), 0) << ret.message();
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
}

TEST(string_utils, parse_vector_spaces_comma_separator_fails) {
    // Commas must not be accepted as separators.
    std::array<float, 2> out {};
    const Ret ret = parse_vector_spaces(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::f32, out.size(),
        "1,2");
    EXPECT_NE(ret.code(), 0);
}

TEST(string_utils, parse_vector_spaces_i16_min_max_succeeds) {
    std::array<int16_t, 2> out {};
    const Ret ret = parse_vector_spaces(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::i16, out.size(),
        "-32768 32767");
    ASSERT_EQ(ret.code(), 0) << ret.message();
    EXPECT_EQ(out[0], -32768);
    EXPECT_EQ(out[1], 32767);
}

TEST(string_utils, parse_vector_spaces_i16_respects_end_pointer_at_page_boundary) {
    const size_t page_size = system_page_size();
    ASSERT_GT(page_size, 0u);
    GuardedPages pages(page_size);
    ASSERT_TRUE(pages.valid());

    const char payload[] = "1 2";
    char* const line = pages.write_at_first_page_end(payload, sizeof(payload) - 1);
    ASSERT_TRUE(pages.protect_second_page());

    std::array<int16_t, 2> out {};
    const Ret ret = parse_vector_spaces(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::i16, out.size(),
        line, pages.first_page_end());
    ASSERT_EQ(ret.code(), 0) << ret.message();
    EXPECT_EQ(out[0], 1);
    EXPECT_EQ(out[1], 2);
}

TEST(string_utils, parse_vector_spaces_f32_scientific_notation) {
    std::array<float, 3> out {};
    const Ret ret = parse_vector_spaces(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::f32, out.size(),
        "1.5e2 -2.5e-1 3.0e0");
    ASSERT_EQ(ret.code(), 0) << ret.message();
    EXPECT_FLOAT_EQ(out[0], 150.0f);
    EXPECT_FLOAT_EQ(out[1], -0.25f);
    EXPECT_FLOAT_EQ(out[2], 3.0f);
}

TEST(string_utils, convert_vector_f16_success) {
    const std::array<float, 3> input {1.5f, -2.0f, 3.25f};
    std::array<float16, 3> out {};
    const Ret ret = convert_vector(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::f16,
        input.size(), input.data(), input.size());
    ASSERT_EQ(0, ret.code()) << ret.message();
    EXPECT_FLOAT_EQ(static_cast<float>(out[0]), 1.5f);
    EXPECT_FLOAT_EQ(static_cast<float>(out[1]), -2.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(out[2]), 3.25f);
}

TEST(string_utils, convert_vector_i16_non_integral_fails) {
    const std::array<float, 2> input {1.0f, 2.5f};
    std::array<int16_t, 2> out {};
    const Ret ret = convert_vector(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::i16,
        input.size(), input.data(), input.size());
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("Query vector contains non-integral i16 value", ret.message());
}

// ── load_vector ─────────────────────────────────────────────────────────────

class LoadVectorTest : public ::testing::Test {
protected:
    std::filesystem::path tmp_path_;

    void SetUp() override {
        tmp_path_ = std::filesystem::temp_directory_path()
            / ("sketch2_utest_load_vector_" + std::to_string(getpid()));
    }

    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove(tmp_path_, ec);
    }

    void write_file(const std::string& content) {
        std::ofstream out(tmp_path_, std::ios::binary);
        out << content;
    }
};

TEST_F(LoadVectorTest, LoadsFileContentIntoString) {
    write_file("1.0, 2.0, 3.0");
    std::string vec;
    const Ret ret = load_vector(tmp_path_.c_str(), vec);
    ASSERT_EQ(0, ret.code()) << ret.message();
    EXPECT_EQ("1.0, 2.0, 3.0", vec);
}

TEST_F(LoadVectorTest, StripsTrailingNewline) {
    write_file("1.0, 2.0, 3.0\n");
    std::string vec;
    const Ret ret = load_vector(tmp_path_.c_str(), vec);
    ASSERT_EQ(0, ret.code()) << ret.message();
    EXPECT_EQ("1.0, 2.0, 3.0", vec);
}

TEST_F(LoadVectorTest, StripsTrailingCarriageReturnAndNewline) {
    write_file("1.0, 2.0, 3.0\r\n");
    std::string vec;
    const Ret ret = load_vector(tmp_path_.c_str(), vec);
    ASSERT_EQ(0, ret.code()) << ret.message();
    EXPECT_EQ("1.0, 2.0, 3.0", vec);
}

TEST_F(LoadVectorTest, LoadsSpaceDelimitedContent) {
    write_file("1.0 2.0 3.0\n");
    std::string vec;
    const Ret ret = load_vector(tmp_path_.c_str(), vec);
    ASSERT_EQ(0, ret.code()) << ret.message();
    EXPECT_EQ("1.0 2.0 3.0", vec);
}

TEST_F(LoadVectorTest, EmptyFileProducesEmptyString) {
    write_file("");
    std::string vec = "old";
    const Ret ret = load_vector(tmp_path_.c_str(), vec);
    ASSERT_EQ(0, ret.code()) << ret.message();
    EXPECT_TRUE(vec.empty());
}

TEST_F(LoadVectorTest, NullPathFails) {
    std::string vec;
    EXPECT_NE(0, load_vector(nullptr, vec).code());
}

TEST_F(LoadVectorTest, NonExistentPathFails) {
    std::string vec;
    const Ret ret = load_vector("/nonexistent/path/vector.txt", vec);
    EXPECT_NE(0, ret.code());
}

TEST_F(LoadVectorTest, ResultCanBePassedToParseVector) {
    write_file("1.25, -2.5, 3.0\n");
    std::string vec;
    ASSERT_EQ(0, load_vector(tmp_path_.c_str(), vec).code());

    std::array<float, 3> out {};
    const Ret ret = parse_vector(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::f32, out.size(),
        vec.c_str());
    ASSERT_EQ(0, ret.code()) << ret.message();
    EXPECT_FLOAT_EQ(out[0], 1.25f);
    EXPECT_FLOAT_EQ(out[1], -2.5f);
    EXPECT_FLOAT_EQ(out[2], 3.0f);
}

TEST_F(LoadVectorTest, ResultCanBePassedToParseVectorSpaces) {
    write_file("1.25 -2.5 3.0\n");
    std::string vec;
    ASSERT_EQ(0, load_vector(tmp_path_.c_str(), vec).code());

    std::array<float, 3> out {};
    const Ret ret = parse_vector_spaces(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::f32, out.size(),
        vec.c_str());
    ASSERT_EQ(0, ret.code()) << ret.message();
    EXPECT_FLOAT_EQ(out[0], 1.25f);
    EXPECT_FLOAT_EQ(out[1], -2.5f);
    EXPECT_FLOAT_EQ(out[2], 3.0f);
}

} // namespace sketch2
