// Unit tests for shared utility types.

#include "utils/shared_types.h"

#include <gtest/gtest.h>

#include <type_traits>

namespace sketch2 {

static_assert(std::is_move_constructible_v<Ret>);
static_assert(std::is_move_assignable_v<Ret>);

TEST(shared_types, ret_copy_preserves_content_flag) {
    const Ret original(0, "payload", true);

    const Ret copied(original);

    EXPECT_EQ(0, copied.code());
    EXPECT_EQ("payload", copied.message());
    EXPECT_TRUE(copied.is_content());
}

TEST(shared_types, data_type_on_disk_mapping_is_locked) {
    // These are absolute file-format values, not enum-order expectations.
    EXPECT_EQ(0, data_type_to_int(DataType::f16));
    EXPECT_EQ(1, data_type_to_int(DataType::f32));
    EXPECT_EQ(2, data_type_to_int(DataType::i16));
    EXPECT_EQ(3, data_type_to_int(DataType::f8));

    EXPECT_EQ(DataType::f16, data_type_from_int(0));
    EXPECT_EQ(DataType::f32, data_type_from_int(1));
    EXPECT_EQ(DataType::i16, data_type_from_int(2));
    EXPECT_EQ(DataType::f8, data_type_from_int(3));
    EXPECT_THROW(data_type_from_int(4), std::runtime_error);
}

TEST(shared_types, data_type_strings_and_sizes_are_fixed) {
    EXPECT_STREQ("f16", data_type_to_string(DataType::f16));
    EXPECT_STREQ("f32", data_type_to_string(DataType::f32));
    EXPECT_STREQ("i16", data_type_to_string(DataType::i16));
    EXPECT_STREQ("f8", data_type_to_string(DataType::f8));

    EXPECT_EQ(2u, data_type_size(DataType::f16));
    EXPECT_EQ(4u, data_type_size(DataType::f32));
    EXPECT_EQ(2u, data_type_size(DataType::i16));
    EXPECT_EQ(1u, data_type_size(DataType::f8));
}

} // namespace sketch2
