#include "string_utils.h"

#include <array>
#include <cstdint>

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

} // namespace sketch2

