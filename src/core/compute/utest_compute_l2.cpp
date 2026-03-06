#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "core/compute/compute_l2.h"

using namespace sketch2;

namespace {

TEST(ComputeL2Test, DistF32ComputesSquaredDistance) {
    const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> b = {1.0f, 0.0f, 1.0f, 10.0f};
    ComputeL2 l2;

    const double got = l2.dist(reinterpret_cast<const uint8_t*>(a.data()),
                               reinterpret_cast<const uint8_t*>(b.data()),
                               DataType::f32, a.size());
    // (0)^2 + (2)^2 + (2)^2 + (-6)^2 = 44
    EXPECT_DOUBLE_EQ(44.0, got);
}

TEST(ComputeL2Test, DistI16ComputesSquaredDistance) {
    const std::vector<int16_t> a = {10, -2, 7, -8};
    const std::vector<int16_t> b = {4, -5, 10, -8};
    ComputeL2 l2;

    const double got = l2.dist(reinterpret_cast<const uint8_t*>(a.data()),
                               reinterpret_cast<const uint8_t*>(b.data()),
                               DataType::i16, a.size());
    // (6)^2 + (3)^2 + (-3)^2 + (0)^2 = 54
    EXPECT_DOUBLE_EQ(54.0, got);
}

TEST(ComputeL2Test, DistF16ComputesSquaredDistance) {
    if (!supports_f16()) {
        GTEST_SKIP() << "f16 is not supported on this build";
    }

    const std::vector<float16> a = {float16(1.0f), float16(2.0f), float16(3.0f), float16(4.0f)};
    const std::vector<float16> b = {float16(1.0f), float16(0.0f), float16(1.0f), float16(10.0f)};
    ComputeL2 l2;

    const double got = l2.dist(reinterpret_cast<const uint8_t*>(a.data()),
                               reinterpret_cast<const uint8_t*>(b.data()),
                               DataType::f16, a.size());
    EXPECT_NEAR(44.0, got, 1e-3);
}

TEST(ComputeL2Test, ResolveDistReturnsFunctionForAllTypes) {
    EXPECT_NE(nullptr, ComputeL2::resolve_dist(DataType::f32));
    if (supports_f16()) {
        EXPECT_NE(nullptr, ComputeL2::resolve_dist(DataType::f16));
    }
    EXPECT_NE(nullptr, ComputeL2::resolve_dist(DataType::i16));
}

} // namespace
