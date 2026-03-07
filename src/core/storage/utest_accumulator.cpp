#include "core/storage/accumulator.h"

#include <array>

#include <gtest/gtest.h>

namespace sketch2 {

namespace {

template <typename T, size_t N>
const uint8_t* as_bytes(const std::array<T, N>& data) {
    return reinterpret_cast<const uint8_t*>(data.data());
}

} // namespace

TEST(AccumulatorTest, AddAndReadVectorsById) {
    Accumulator accumulator;
    ASSERT_EQ(0, accumulator.init(256, DataType::f32, 4).code());

    const std::array<float, 4> v0 {1.0f, 2.0f, 3.0f, 4.0f};
    const std::array<float, 4> v1 {5.0f, 6.0f, 7.0f, 8.0f};

    ASSERT_EQ(0, accumulator.add_vector(20, as_bytes(v1)).code());
    ASSERT_EQ(0, accumulator.add_vector(10, as_bytes(v0)).code());

    EXPECT_EQ(accumulator.vectors_count(), 2U);
    EXPECT_EQ(accumulator.get_vector_ids(), (std::vector<uint64_t> {10, 20}));

    const float* f0 = reinterpret_cast<const float*>(accumulator.get_vector(10));
    const float* f1 = reinterpret_cast<const float*>(accumulator.get_vector(20));
    ASSERT_NE(nullptr, f0);
    ASSERT_NE(nullptr, f1);
    EXPECT_FLOAT_EQ(f0[0], 1.0f);
    EXPECT_FLOAT_EQ(f0[3], 4.0f);
    EXPECT_FLOAT_EQ(f1[0], 5.0f);
    EXPECT_FLOAT_EQ(f1[3], 8.0f);
}

TEST(AccumulatorTest, DeleteVectorTracksSortedDeletedIds) {
    Accumulator accumulator;
    ASSERT_EQ(0, accumulator.init(128, DataType::i16, 4).code());

    ASSERT_EQ(0, accumulator.delete_vector(11).code());
    ASSERT_EQ(0, accumulator.delete_vector(7).code());
    ASSERT_EQ(0, accumulator.delete_vector(9).code());

    EXPECT_EQ(accumulator.deleted_count(), 3U);
    EXPECT_EQ(accumulator.get_deleted_ids(), (std::vector<uint64_t> {7, 9, 11}));
}

TEST(AccumulatorTest, DeleteRemovesVectorAndAddRemovesDeletedId) {
    Accumulator accumulator;
    ASSERT_EQ(0, accumulator.init(128, DataType::i16, 4).code());

    const std::array<int16_t, 4> vec0 {10, 20, 30, 40};
    const std::array<int16_t, 4> vec1 {50, 60, 70, 80};

    ASSERT_EQ(0, accumulator.add_vector(5, as_bytes(vec0)).code());
    ASSERT_EQ(0, accumulator.delete_vector(5).code());
    EXPECT_EQ(accumulator.vectors_count(), 0U);
    EXPECT_EQ(accumulator.deleted_count(), 1U);
    EXPECT_EQ(nullptr, accumulator.get_vector(5));
    EXPECT_EQ(accumulator.get_deleted_ids(), (std::vector<uint64_t> {5}));

    ASSERT_EQ(0, accumulator.add_vector(5, as_bytes(vec1)).code());
    EXPECT_EQ(accumulator.vectors_count(), 1U);
    EXPECT_EQ(accumulator.deleted_count(), 0U);
    EXPECT_EQ(accumulator.get_vector_ids(), (std::vector<uint64_t> {5}));
    EXPECT_TRUE(accumulator.get_deleted_ids().empty());

    const int16_t* values = reinterpret_cast<const int16_t*>(accumulator.get_vector(5));
    ASSERT_NE(nullptr, values);
    EXPECT_EQ(values[0], 50);
    EXPECT_EQ(values[3], 80);
}

TEST(AccumulatorTest, AddVectorReplacesExistingValue) {
    Accumulator accumulator;
    ASSERT_EQ(0, accumulator.init(64, DataType::i16, 4).code());

    const std::array<int16_t, 4> vec0 {1, 2, 3, 4};
    const std::array<int16_t, 4> vec1 {5, 6, 7, 8};
    ASSERT_EQ(0, accumulator.add_vector(3, as_bytes(vec0)).code());
    ASSERT_EQ(0, accumulator.add_vector(3, as_bytes(vec1)).code());

    EXPECT_EQ(accumulator.vectors_count(), 1U);
    const int16_t* values = reinterpret_cast<const int16_t*>(accumulator.get_vector(3));
    ASSERT_NE(nullptr, values);
    EXPECT_EQ(values[0], 5);
    EXPECT_EQ(values[3], 8);
}

TEST(AccumulatorTest, CanAddVectorReportsCapacity) {
    Accumulator accumulator;
    ASSERT_EQ(0, accumulator.init(24, DataType::f32, 4).code());

    EXPECT_TRUE(accumulator.can_add_vector(1));
    const std::array<float, 4> vec {1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT_EQ(0, accumulator.add_vector(1, as_bytes(vec)).code());
    EXPECT_FALSE(accumulator.can_add_vector(2));
    EXPECT_TRUE(accumulator.can_add_vector(1));
}

TEST(AccumulatorTest, CanDeleteVectorReportsCapacity) {
    Accumulator accumulator;
    ASSERT_EQ(0, accumulator.init(8, DataType::i16, 4).code());

    EXPECT_TRUE(accumulator.can_delete_vector(1));
    ASSERT_EQ(0, accumulator.delete_vector(1).code());
    EXPECT_FALSE(accumulator.can_delete_vector(2));
    EXPECT_TRUE(accumulator.can_delete_vector(1));
}

TEST(AccumulatorTest, ClearRemovesAllStoredState) {
    Accumulator accumulator;
    ASSERT_EQ(0, accumulator.init(64, DataType::i16, 4).code());

    const std::array<int16_t, 4> vec {1, 2, 3, 4};
    ASSERT_EQ(0, accumulator.add_vector(3, as_bytes(vec)).code());
    ASSERT_EQ(0, accumulator.delete_vector(5).code());

    accumulator.clear();

    EXPECT_EQ(0u, accumulator.vectors_count());
    EXPECT_EQ(0u, accumulator.deleted_count());
    EXPECT_TRUE(accumulator.get_vector_ids().empty());
    EXPECT_TRUE(accumulator.get_deleted_ids().empty());
    EXPECT_EQ(nullptr, accumulator.get_vector(3));
    EXPECT_TRUE(accumulator.can_add_vector(7));
    EXPECT_TRUE(accumulator.can_delete_vector(8));
}

TEST(AccumulatorTest, BufferFullForVectorReturnsError) {
    Accumulator accumulator;
    ASSERT_EQ(0, accumulator.init(23, DataType::f32, 4).code());

    const std::array<float, 4> vec {1.0f, 2.0f, 3.0f, 4.0f};
    const Ret ret = accumulator.add_vector(1, as_bytes(vec));
    EXPECT_NE(ret.code(), 0);
    EXPECT_EQ(ret.message(), "Accumulator: buffer full");
}

TEST(AccumulatorTest, BufferFullForDeletedIdReturnsError) {
    Accumulator accumulator;
    ASSERT_EQ(0, accumulator.init(7, DataType::i16, 4).code());

    const Ret ret = accumulator.delete_vector(2);
    EXPECT_NE(ret.code(), 0);
    EXPECT_EQ(ret.message(), "Accumulator: buffer full");
}

TEST(AccumulatorTest, DeleteCanFreeSpaceForDeletedId) {
    Accumulator accumulator;
    ASSERT_EQ(0, accumulator.init(24, DataType::i16, 4).code());

    const std::array<int16_t, 4> vec {1, 2, 3, 4};
    ASSERT_EQ(0, accumulator.add_vector(9, as_bytes(vec)).code());
    ASSERT_EQ(0, accumulator.delete_vector(9).code());

    EXPECT_EQ(accumulator.vectors_count(), 0U);
    EXPECT_EQ(accumulator.deleted_count(), 1U);
    EXPECT_EQ(accumulator.get_deleted_ids(), (std::vector<uint64_t> {9}));
}

TEST(AccumulatorTest, MissingVectorReturnsNull) {
    Accumulator accumulator;
    ASSERT_EQ(0, accumulator.init(128, DataType::i16, 4).code());
    EXPECT_EQ(nullptr, accumulator.get_vector(123));
}

} // namespace sketch2
