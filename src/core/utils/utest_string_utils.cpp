// Unit tests for textual vector parsing and formatting helpers.

#include "string_utils.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>

namespace sketch2 {

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

TEST(string_utils, parse_vector_f16_unsupported_platform_fails) {
    if (supports_f16()) {
        GTEST_SKIP() << "f16 is supported on this platform";
    }
    std::array<float16, 2> out {};
    const Ret ret = parse_vector(
        reinterpret_cast<uint8_t*>(out.data()), sizeof(out), DataType::f16, out.size(),
        "1.0, 2.0");
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("f16 is not supported.", ret.message());
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
    const Ret ret = print_vector(reinterpret_cast<uint8_t*>(vec.data()), DataType::f32, vec.size(), out, sizeof(out));
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

TEST(string_utils, data_type_from_int_invalid_values_throw) {
    EXPECT_THROW(data_type_from_int(-1), std::runtime_error);
    EXPECT_THROW(data_type_from_int(3), std::runtime_error);
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
    EXPECT_EQ(DataType::i16, data_type_from_string("i16"));
    EXPECT_THROW(data_type_from_string("bad"), std::runtime_error);
    EXPECT_STREQ("f32", data_type_to_string(DataType::f32));
    EXPECT_STREQ("i16", data_type_to_string(DataType::i16));
}

} // namespace sketch2
