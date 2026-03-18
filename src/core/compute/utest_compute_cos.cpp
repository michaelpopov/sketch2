// Unit tests for cosine-distance compute implementations.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/compute/utest_compute_helpers.h"
#include "core/compute/compute_cos.h"
#include "core/compute/compute_cos_avx2.h"
#include "core/compute/compute_cos_avx512.h"
#include "core/compute/compute_cos_neon.h"

using namespace sketch2;
using namespace sketch2::test;

namespace {

#if SKETCH_HAS_AVX2
#define SKETCH2_COMPUTE_AVX2_TESTS 1
#endif

#if SKETCH_HAS_AVX512F
#define SKETCH2_COMPUTE_AVX512F_TESTS 1
#endif

#if SKETCH_HAS_AVX512VNNI
#define SKETCH2_COMPUTE_AVX512VNNI_TESTS 1
#endif

TEST(ComputeCosTest, DistF32ComputesDistance) {
    const std::vector<float> a = {1.0f, 0.0f, 0.0f, 0.0f};
    const std::vector<float> b = {1.0f, 1.0f, 0.0f, 0.0f};
    ComputeCos cos;
    const double got = cos.dist(reinterpret_cast<const uint8_t*>(a.data()),
                                reinterpret_cast<const uint8_t*>(b.data()),
                                DataType::f32, a.size());
    EXPECT_NEAR(reference_cosine_distance(a.data(), b.data(), a.size()), got, 1e-12);
}

TEST(ComputeCosTest, DistI16ComputesDistance) {
    const std::vector<int16_t> a = {1, 2, -1, 0};
    const std::vector<int16_t> b = {2, 1, -2, 0};
    ComputeCos cos;
    const double got = cos.dist(reinterpret_cast<const uint8_t*>(a.data()),
                                reinterpret_cast<const uint8_t*>(b.data()),
                                DataType::i16, a.size());
    EXPECT_NEAR(reference_cosine_distance(a.data(), b.data(), a.size()), got, 1e-12);
}

TEST(ComputeCosTest, DistI16NearParallelRetainsSmallPositiveDistance) {
    const size_t dim = 256;
    std::vector<int16_t> a(dim, INT16_MAX);
    std::vector<int16_t> b(dim, INT16_MAX);
    b.back() = INT16_MAX - 1;

    ComputeCos cos;
    const double ref = reference_cosine_distance(a.data(), b.data(), dim);
    const double got = cos.dist(reinterpret_cast<const uint8_t*>(a.data()),
                                reinterpret_cast<const uint8_t*>(b.data()),
                                DataType::i16, dim);
    ASSERT_GT(ref, 0.0);
    EXPECT_GT(got, 0.0);
    EXPECT_NEAR(ref, got, 1e-12);
}

TEST(ComputeCosTest, DistF16ComputesDistance) {
    if (!supports_f16()) {
        GTEST_SKIP() << "f16 is not supported on this build";
    }

    const std::vector<float16> a = {float16(1.0f), float16(0.0f), float16(0.0f), float16(0.0f)};
    const std::vector<float16> b = {float16(1.0f), float16(1.0f), float16(0.0f), float16(0.0f)};
    ComputeCos cos;
    const double got = cos.dist(reinterpret_cast<const uint8_t*>(a.data()),
                                reinterpret_cast<const uint8_t*>(b.data()),
                                DataType::f16, a.size());
    EXPECT_NEAR(reference_cosine_distance(a.data(), b.data(), a.size()), got, 1e-3);
}

TEST(ComputeCosTest, IdenticalVectorsYieldZeroDistance) {
    const std::vector<float> a = {1.0f, 2.0f, -3.0f, 4.0f};
    ComputeCos cos;
    const double got = cos.dist(reinterpret_cast<const uint8_t*>(a.data()),
                                reinterpret_cast<const uint8_t*>(a.data()),
                                DataType::f32, a.size());
    EXPECT_NEAR(0.0, got, 1e-12);
}

TEST(ComputeCosTest, OrthogonalVectorsYieldUnitDistance) {
    // Axis-aligned unit vectors are orthogonal: cos(90°) = 0, distance = 1 - 0 = 1.
    const std::vector<float> a = {1.0f, 0.0f, 0.0f, 0.0f};
    const std::vector<float> b = {0.0f, 1.0f, 0.0f, 0.0f};
    ComputeCos cos;
    const double got = cos.dist(reinterpret_cast<const uint8_t*>(a.data()),
                                reinterpret_cast<const uint8_t*>(b.data()),
                                DataType::f32, a.size());
    EXPECT_NEAR(1.0, got, 1e-12);
}

TEST(ComputeCosTest, AntiparallelVectorsYieldTwoDistance) {
    // b = -a: cos(180°) = -1, distance = 1 - (-1) = 2.
    const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> b = {-1.0f, -2.0f, -3.0f, -4.0f};
    ComputeCos cos;
    const double got = cos.dist(reinterpret_cast<const uint8_t*>(a.data()),
                                reinterpret_cast<const uint8_t*>(b.data()),
                                DataType::f32, a.size());
    EXPECT_NEAR(2.0, got, 1e-12);
}

TEST(ComputeCosTest, ZeroNormHandlingMatchesContract) {
    const std::vector<float> zero = {0.0f, 0.0f, 0.0f, 0.0f};
    const std::vector<float> nonzero = {1.0f, 2.0f, 0.0f, 0.0f};
    ComputeCos cos;

    const double both_zero = cos.dist(reinterpret_cast<const uint8_t*>(zero.data()),
                                      reinterpret_cast<const uint8_t*>(zero.data()),
                                      DataType::f32, zero.size());
    const double one_zero = cos.dist(reinterpret_cast<const uint8_t*>(zero.data()),
                                     reinterpret_cast<const uint8_t*>(nonzero.data()),
                                     DataType::f32, zero.size());
    EXPECT_DOUBLE_EQ(0.0, both_zero);
    EXPECT_DOUBLE_EQ(1.0, one_zero);
}

TEST(ComputeCosTest, ResolveDistReturnsFunctionForAllTypes) {
    EXPECT_NE(nullptr, ComputeCos::resolve_dist(DataType::f32));
    if (supports_f16()) {
        EXPECT_NE(nullptr, ComputeCos::resolve_dist(DataType::f16));
    }
    EXPECT_NE(nullptr, ComputeCos::resolve_dist(DataType::i16));
}

TEST(ComputeCosTest, ResolveDotReturnsFunctionForAllTypes) {
    EXPECT_NE(nullptr, ComputeCos::resolve_dot(DataType::f32));
    if (supports_f16()) {
        EXPECT_NE(nullptr, ComputeCos::resolve_dot(DataType::f16));
    }
    EXPECT_NE(nullptr, ComputeCos::resolve_dot(DataType::i16));
}

TEST(ComputeCosTest, ResolveSquaredNormReturnsFunctionForAllTypes) {
    EXPECT_NE(nullptr, ComputeCos::resolve_squared_norm(DataType::f32));
    if (supports_f16()) {
        EXPECT_NE(nullptr, ComputeCos::resolve_squared_norm(DataType::f16));
    }
    EXPECT_NE(nullptr, ComputeCos::resolve_squared_norm(DataType::i16));
}

TEST(ComputeCosTest, ResolveDotComputesKnownValues) {
    const std::vector<float> a_f32 = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> b_f32 = {5.0f, 6.0f, 7.0f, 8.0f};
    const double got_f32 = ComputeCos::resolve_dot(DataType::f32)(
        reinterpret_cast<const uint8_t*>(a_f32.data()),
        reinterpret_cast<const uint8_t*>(b_f32.data()),
        a_f32.size());
    EXPECT_DOUBLE_EQ(70.0, got_f32);

    const std::vector<int16_t> a_i16 = {1, -2, 3, -4};
    const std::vector<int16_t> b_i16 = {5, 6, -7, -8};
    const double got_i16 = ComputeCos::resolve_dot(DataType::i16)(
        reinterpret_cast<const uint8_t*>(a_i16.data()),
        reinterpret_cast<const uint8_t*>(b_i16.data()),
        a_i16.size());
    EXPECT_DOUBLE_EQ(4.0, got_i16);

#if defined(__FLT16_MANT_DIG__)
    if (supports_f16()) {
        const std::vector<float16> a_f16 = {float16(1.0f), float16(2.0f), float16(3.0f), float16(4.0f)};
        const std::vector<float16> b_f16 = {float16(5.0f), float16(6.0f), float16(7.0f), float16(8.0f)};
        const double got_f16 = ComputeCos::resolve_dot(DataType::f16)(
            reinterpret_cast<const uint8_t*>(a_f16.data()),
            reinterpret_cast<const uint8_t*>(b_f16.data()),
            a_f16.size());
        EXPECT_DOUBLE_EQ(70.0, got_f16);
    }
#endif
}

TEST(ComputeCosTest, ResolveSquaredNormComputesKnownValues) {
    const std::vector<float> a_f32 = {1.0f, 2.0f, 3.0f, 4.0f};
    const double got_f32 = ComputeCos::resolve_squared_norm(DataType::f32)(
        reinterpret_cast<const uint8_t*>(a_f32.data()), a_f32.size());
    EXPECT_DOUBLE_EQ(30.0, got_f32);

    const std::vector<int16_t> a_i16 = {1, -2, 3, -4};
    const double got_i16 = ComputeCos::resolve_squared_norm(DataType::i16)(
        reinterpret_cast<const uint8_t*>(a_i16.data()), a_i16.size());
    EXPECT_DOUBLE_EQ(30.0, got_i16);

#if defined(__FLT16_MANT_DIG__)
    if (supports_f16()) {
        const std::vector<float16> a_f16 = {float16(1.0f), float16(2.0f), float16(3.0f), float16(4.0f)};
        const double got_f16 = ComputeCos::resolve_squared_norm(DataType::f16)(
            reinterpret_cast<const uint8_t*>(a_f16.data()), a_f16.size());
        EXPECT_DOUBLE_EQ(30.0, got_f16);
    }
#endif
}

TEST(ComputeCosScalar, DistI16UsesScalarFallback) {
    if (get_singleton().compute_unit().kind() != ComputeBackendKind::scalar) {
        GTEST_SKIP() << "scalar fallback is not the active runtime implementation";
    }
    const std::vector<int16_t> a = {1, 2, -1, 0};
    const std::vector<int16_t> b = {2, 1, -2, 0};
    ComputeCos cos;
    const double got = cos.dist(reinterpret_cast<const uint8_t*>(a.data()),
                                reinterpret_cast<const uint8_t*>(b.data()),
                                DataType::i16, a.size());
    EXPECT_NEAR(reference_cosine_distance(a.data(), b.data(), a.size()), got, 1e-12);
}

#if defined(__aarch64__)
TEST(ComputeCosNeon, DotF32MatchesReference) {
    const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, -5.0f};
    const std::vector<float> b = {0.0f, 2.5f, 1.0f, 0.0f, -1.0f};
    const double got = ComputeCos_Neon::dot_f32(reinterpret_cast<const uint8_t*>(a.data()),
                                                reinterpret_cast<const uint8_t*>(b.data()),
                                                a.size());
    EXPECT_NEAR(reference_dot(a.data(), b.data(), a.size()), got, 1e-6);

    // Also verify across multiple SIMD widths (f32 SIMD width = 4).
    for (size_t dim : {size_t(16), size_t(32)}) {
        std::vector<float> a2(dim), b2(dim);
        for (size_t i = 0; i < dim; ++i) {
            a2[i] = static_cast<float>(i % 7 + 1) * 0.5f - 2.0f;
            b2[i] = static_cast<float>(i % 5) * 0.3f - 0.75f;
        }
        const double ref = reference_dot(a2.data(), b2.data(), dim);
        const double got2 = ComputeCos_Neon::dot_f32(
            reinterpret_cast<const uint8_t*>(a2.data()),
            reinterpret_cast<const uint8_t*>(b2.data()), dim);
        EXPECT_NEAR(ref, got2, std::max(1e-5, std::abs(ref) * 5e-5)) << "dim=" << dim;
    }
}

TEST(ComputeCosNeon, DistF32MatchesReference) {
    const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, -5.0f};
    const std::vector<float> b = {0.0f, 2.5f, 1.0f, 0.0f, -1.0f};
    const double got = ComputeCos_Neon::dist_f32(reinterpret_cast<const uint8_t*>(a.data()),
                                                 reinterpret_cast<const uint8_t*>(b.data()),
                                                 a.size());
    EXPECT_NEAR(reference_cosine_distance(a.data(), b.data(), a.size()), got, 1e-6);

    // Also verify across multiple SIMD widths (f32 SIMD width = 4).
    for (size_t dim : {size_t(16), size_t(32)}) {
        std::vector<float> a2(dim), b2(dim);
        for (size_t i = 0; i < dim; ++i) {
            a2[i] = static_cast<float>(i % 7 + 1) * 0.5f - 2.0f;
            b2[i] = static_cast<float>(i % 5) * 0.3f - 0.75f;
        }
        const double ref = reference_cosine_distance(a2.data(), b2.data(), dim);
        const double got2 = ComputeCos_Neon::dist_f32(
            reinterpret_cast<const uint8_t*>(a2.data()),
            reinterpret_cast<const uint8_t*>(b2.data()), dim);
        EXPECT_NEAR(ref, got2, 5e-5) << "dim=" << dim;
    }
}

#if defined(__FLT16_MANT_DIG__)
TEST(ComputeCosNeon, DotF16MatchesReference) {
    if (!supports_f16()) {
        GTEST_SKIP() << "f16 is not supported on this build";
    }

    const std::vector<float16> a = {float16(1.0f), float16(2.0f), float16(3.0f), float16(4.0f)};
    const std::vector<float16> b = {float16(5.0f), float16(6.0f), float16(7.0f), float16(8.0f)};
    const double got = ComputeCos_Neon::dot_f16(reinterpret_cast<const uint8_t*>(a.data()),
                                                reinterpret_cast<const uint8_t*>(b.data()),
                                                a.size());
    EXPECT_NEAR(reference_dot(a.data(), b.data(), a.size()), got, 1e-6);

    // Also verify across multiple SIMD widths (f16 SIMD width = 4 or 8).
    for (size_t dim : {size_t(16), size_t(32)}) {
        std::vector<float16> a2(dim), b2(dim);
        for (size_t i = 0; i < dim; ++i) {
            a2[i] = static_cast<float16>(static_cast<float>(i % 7 + 1) * 0.5f - 2.0f);
            b2[i] = static_cast<float16>(static_cast<float>(i % 5) * 0.3f - 0.75f);
        }
        const double ref = reference_dot(a2.data(), b2.data(), dim);
        const double got2 = ComputeCos_Neon::dot_f16(
            reinterpret_cast<const uint8_t*>(a2.data()),
            reinterpret_cast<const uint8_t*>(b2.data()), dim);
        EXPECT_NEAR(ref, got2, std::max(1e-2, std::abs(ref) * 1e-2)) << "dim=" << dim;
    }
}
#endif

TEST(ComputeCosNeon, DotI16MatchesReference) {
    const std::vector<int16_t> a = {10, -2, 7, -8, 20};
    const std::vector<int16_t> b = {4, -5, 10, -8, 18};
    const double got = ComputeCos_Neon::dot_i16(reinterpret_cast<const uint8_t*>(a.data()),
                                                reinterpret_cast<const uint8_t*>(b.data()),
                                                a.size());
    EXPECT_NEAR(reference_dot(a.data(), b.data(), a.size()), got, 1e-6);

    // Also verify across multiple SIMD widths (i16 SIMD width = 8).
    for (size_t dim : {size_t(16), size_t(32)}) {
        std::vector<int16_t> a2(dim), b2(dim);
        for (size_t i = 0; i < dim; ++i) {
            a2[i] = static_cast<int16_t>(static_cast<int>((i * 13 + 7) % 200) - 100);
            b2[i] = static_cast<int16_t>(static_cast<int>((i * 17 + 3) % 200) - 100);
        }
        const double ref = reference_dot(a2.data(), b2.data(), dim);
        const double got2 = ComputeCos_Neon::dot_i16(
            reinterpret_cast<const uint8_t*>(a2.data()),
            reinterpret_cast<const uint8_t*>(b2.data()), dim);
        EXPECT_DOUBLE_EQ(ref, got2) << "dim=" << dim;
    }
}

TEST(ComputeCosNeon, DistI16MatchesReference) {
    const std::vector<int16_t> a = {10, -2, 7, -8, 20};
    const std::vector<int16_t> b = {4, -5, 10, -8, 18};
    const double got = ComputeCos_Neon::dist_i16(reinterpret_cast<const uint8_t*>(a.data()),
                                                 reinterpret_cast<const uint8_t*>(b.data()),
                                                 a.size());
    EXPECT_NEAR(reference_cosine_distance(a.data(), b.data(), a.size()), got, 1e-6);

    // Also verify across multiple SIMD widths (i16 SIMD width = 8).
    for (size_t dim : {size_t(16), size_t(32)}) {
        std::vector<int16_t> a2(dim), b2(dim);
        for (size_t i = 0; i < dim; ++i) {
            a2[i] = static_cast<int16_t>(static_cast<int>((i * 13 + 7) % 200) - 100);
            b2[i] = static_cast<int16_t>(static_cast<int>((i * 17 + 3) % 200) - 100);
        }
        const double ref = reference_cosine_distance(a2.data(), b2.data(), dim);
        const double got2 = ComputeCos_Neon::dist_i16(
            reinterpret_cast<const uint8_t*>(a2.data()),
            reinterpret_cast<const uint8_t*>(b2.data()), dim);
        EXPECT_NEAR(ref, got2, 2e-4) << "dim=" << dim;
    }
}

// Semantic contract tests: identical, orthogonal, and anti-parallel vectors.
TEST(ComputeCosNeon, DistF32IdenticalVectorsYieldZero) {
    const std::vector<float> a = {1.0f, 2.0f, -3.0f, 4.0f, -5.0f};
    const double got = ComputeCos_Neon::dist_f32(reinterpret_cast<const uint8_t*>(a.data()),
                                                 reinterpret_cast<const uint8_t*>(a.data()),
                                                 a.size());
    EXPECT_NEAR(0.0, got, 1e-6);
}

TEST(ComputeCosNeon, DistF32OrthogonalVectorsYieldOne) {
    const std::vector<float> a = {1.0f, 0.0f, 0.0f, 0.0f};
    const std::vector<float> b = {0.0f, 1.0f, 0.0f, 0.0f};
    const double got = ComputeCos_Neon::dist_f32(reinterpret_cast<const uint8_t*>(a.data()),
                                                 reinterpret_cast<const uint8_t*>(b.data()),
                                                 a.size());
    EXPECT_NEAR(1.0, got, 1e-6);
}

TEST(ComputeCosNeon, DistF32AntiparallelVectorsYieldTwo) {
    const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> b = {-1.0f, -2.0f, -3.0f, -4.0f};
    const double got = ComputeCos_Neon::dist_f32(reinterpret_cast<const uint8_t*>(a.data()),
                                                 reinterpret_cast<const uint8_t*>(b.data()),
                                                 a.size());
    EXPECT_NEAR(2.0, got, 1e-6);
}

TEST(ComputeCosNeon, DistI16IdenticalVectorsYieldZero) {
    const std::vector<int16_t> a = {10, -2, 7, -8, 20};
    const double got = ComputeCos_Neon::dist_i16(reinterpret_cast<const uint8_t*>(a.data()),
                                                 reinterpret_cast<const uint8_t*>(a.data()),
                                                 a.size());
    EXPECT_NEAR(0.0, got, 1e-6);
}

TEST(ComputeCosNeon, DistI16OrthogonalVectorsYieldOne) {
    const std::vector<int16_t> a = {1, 0, 0, 0, 0, 0, 0, 0};
    const std::vector<int16_t> b = {0, 1, 0, 0, 0, 0, 0, 0};
    const double got = ComputeCos_Neon::dist_i16(reinterpret_cast<const uint8_t*>(a.data()),
                                                 reinterpret_cast<const uint8_t*>(b.data()),
                                                 a.size());
    EXPECT_NEAR(1.0, got, 1e-6);
}

#if defined(__FLT16_MANT_DIG__)
TEST(ComputeCosNeon, DistF16IdenticalVectorsYieldZero) {
    if (!supports_f16()) {
        GTEST_SKIP() << "f16 is not supported on this build";
    }
    const std::vector<float16> a = {float16(1.0f), float16(2.0f), float16(-3.0f), float16(4.0f)};
    const double got = ComputeCos_Neon::dist_f16(reinterpret_cast<const uint8_t*>(a.data()),
                                                 reinterpret_cast<const uint8_t*>(a.data()),
                                                 a.size());
    EXPECT_NEAR(0.0, got, 1e-3);
}

TEST(ComputeCosNeon, DistF16OrthogonalVectorsYieldOne) {
    if (!supports_f16()) {
        GTEST_SKIP() << "f16 is not supported on this build";
    }
    const std::vector<float16> a = {float16(1.0f), float16(0.0f), float16(0.0f), float16(0.0f)};
    const std::vector<float16> b = {float16(0.0f), float16(1.0f), float16(0.0f), float16(0.0f)};
    const double got = ComputeCos_Neon::dist_f16(reinterpret_cast<const uint8_t*>(a.data()),
                                                 reinterpret_cast<const uint8_t*>(b.data()),
                                                 a.size());
    EXPECT_NEAR(1.0, got, 1e-3);
}
#endif

TEST(ComputeCosNeon, SquaredNormF32MatchesReference) {
    const std::vector<float> a = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f};
    // 1 + 4 + 9 + 16 + 25 = 55
    const double got = ComputeCos_Neon::squared_norm_f32(
        reinterpret_cast<const uint8_t*>(a.data()), a.size());
    EXPECT_DOUBLE_EQ(55.0, got);

    // Also verify across multiple SIMD widths (f32 SIMD width = 4).
    for (size_t dim : {size_t(16), size_t(32)}) {
        std::vector<float> a2(dim);
        for (size_t i = 0; i < dim; ++i)
            a2[i] = static_cast<float>(i % 7 + 1) * 0.5f - 2.0f;
        double ref = reference_squared_norm(a2.data(), dim);
        const double got2 = ComputeCos_Neon::squared_norm_f32(
            reinterpret_cast<const uint8_t*>(a2.data()), dim);
        EXPECT_NEAR(ref, got2, std::max(1e-5, ref * 1e-5)) << "dim=" << dim;
    }
}

TEST(ComputeCosNeon, SquaredNormI16MatchesReference) {
    const std::vector<int16_t> a = {10, -20, 30, -40, 50};
    // 100 + 400 + 900 + 1600 + 2500 = 5500
    const double got = ComputeCos_Neon::squared_norm_i16(
        reinterpret_cast<const uint8_t*>(a.data()), a.size());
    EXPECT_DOUBLE_EQ(5500.0, got);

    // Also verify across multiple SIMD widths (i16 SIMD width = 8).
    for (size_t dim : {size_t(16), size_t(32)}) {
        std::vector<int16_t> a2(dim);
        for (size_t i = 0; i < dim; ++i)
            a2[i] = static_cast<int16_t>(static_cast<int>((i * 13 + 7) % 200) - 100);
        double ref = reference_squared_norm(a2.data(), dim);
        const double got2 = ComputeCos_Neon::squared_norm_i16(
            reinterpret_cast<const uint8_t*>(a2.data()), dim);
        EXPECT_DOUBLE_EQ(ref, got2) << "dim=" << dim;
    }
}

#if defined(__FLT16_MANT_DIG__)
TEST(ComputeCosNeon, SquaredNormF16MatchesReference) {
    if (!supports_f16()) {
        GTEST_SKIP() << "f16 is not supported on this build";
    }
    const std::vector<float16> a = {float16(1.0f), float16(-2.0f), float16(3.0f), float16(-4.0f)};
    // 1 + 4 + 9 + 16 = 30
    const double got = ComputeCos_Neon::squared_norm_f16(
        reinterpret_cast<const uint8_t*>(a.data()), a.size());
    EXPECT_NEAR(30.0, got, 1e-2);

    // Also verify across multiple SIMD widths (f16 SIMD width = 4 or 8).
    for (size_t dim : {size_t(16), size_t(32)}) {
        std::vector<float16> a2(dim);
        for (size_t i = 0; i < dim; ++i)
            a2[i] = static_cast<float16>(static_cast<float>(i % 7 + 1) * 0.5f - 2.0f);
        double ref = reference_squared_norm(a2.data(), dim);
        const double got2 = ComputeCos_Neon::squared_norm_f16(
            reinterpret_cast<const uint8_t*>(a2.data()), dim);
        EXPECT_NEAR(ref, got2, std::max(1e-2, ref * 1e-2)) << "dim=" << dim;
    }
}
#endif

TEST(ComputeCosNeon, DistF32WithQueryNormMatchesReference) {
    const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, -5.0f};
    const std::vector<float> b = {0.0f, 2.5f, 1.0f, 0.0f, -1.0f};
    // b_norm_sq = 0 + 6.25 + 1 + 0 + 1 = 8.25
    const double got = ComputeCos_Neon::dist_f32_with_query_norm(
        reinterpret_cast<const uint8_t*>(a.data()),
        reinterpret_cast<const uint8_t*>(b.data()),
        a.size(), 8.25);
    EXPECT_NEAR(reference_cosine_distance(a.data(), b.data(), a.size()), got, 1e-6);

    // Also verify across multiple SIMD widths (f32 SIMD width = 4).
    for (size_t dim : {size_t(16), size_t(32)}) {
        std::vector<float> a2(dim), b2(dim);
        for (size_t i = 0; i < dim; ++i) {
            a2[i] = static_cast<float>(i % 7 + 1) * 0.5f - 2.0f;
            b2[i] = static_cast<float>(i % 5) * 0.3f - 0.75f;
        }
        double b_norm_sq = reference_squared_norm(b2.data(), dim);
        const double ref = reference_cosine_distance(a2.data(), b2.data(), dim);
        const double got2 = ComputeCos_Neon::dist_f32_with_query_norm(
            reinterpret_cast<const uint8_t*>(a2.data()),
            reinterpret_cast<const uint8_t*>(b2.data()), dim, b_norm_sq);
        EXPECT_NEAR(ref, got2, 5e-5) << "dim=" << dim;
    }
}

TEST(ComputeCosNeon, DistI16WithQueryNormMatchesReference) {
    const std::vector<int16_t> a = {10, -2, 7, -8, 20};
    const std::vector<int16_t> b = {4, -5, 10, -8, 18};
    // b_norm_sq = 16 + 25 + 100 + 64 + 324 = 529
    const double got = ComputeCos_Neon::dist_i16_with_query_norm(
        reinterpret_cast<const uint8_t*>(a.data()),
        reinterpret_cast<const uint8_t*>(b.data()),
        a.size(), 529.0);
    EXPECT_NEAR(reference_cosine_distance(a.data(), b.data(), a.size()), got, 1e-6);

    // Also verify across multiple SIMD widths (i16 SIMD width = 8).
    for (size_t dim : {size_t(16), size_t(32)}) {
        std::vector<int16_t> a2(dim), b2(dim);
        for (size_t i = 0; i < dim; ++i) {
            a2[i] = static_cast<int16_t>(static_cast<int>((i * 13 + 7) % 200) - 100);
            b2[i] = static_cast<int16_t>(static_cast<int>((i * 17 + 3) % 200) - 100);
        }
        double b_norm_sq = reference_squared_norm(b2.data(), dim);
        const double ref = reference_cosine_distance(a2.data(), b2.data(), dim);
        const double got2 = ComputeCos_Neon::dist_i16_with_query_norm(
            reinterpret_cast<const uint8_t*>(a2.data()),
            reinterpret_cast<const uint8_t*>(b2.data()), dim, b_norm_sq);
        EXPECT_NEAR(ref, got2, 2e-4) << "dim=" << dim;
    }
}

#if defined(__FLT16_MANT_DIG__)
TEST(ComputeCosNeon, DistF16WithQueryNormMatchesReference) {
    if (!supports_f16()) {
        GTEST_SKIP() << "f16 is not supported on this build";
    }
    const std::vector<float16> a = {float16(1.0f), float16(2.0f), float16(3.0f), float16(4.0f)};
    const std::vector<float16> b = {float16(0.5f), float16(1.0f), float16(-1.5f), float16(2.0f)};
    // b_norm_sq = 0.25 + 1 + 2.25 + 4 = 7.5
    const double got = ComputeCos_Neon::dist_f16_with_query_norm(
        reinterpret_cast<const uint8_t*>(a.data()),
        reinterpret_cast<const uint8_t*>(b.data()),
        a.size(), 7.5);
    EXPECT_NEAR(reference_cosine_distance(a.data(), b.data(), a.size()), got, 1e-2);

    // Also verify across multiple SIMD widths (f16 SIMD width = 4 or 8).
    for (size_t dim : {size_t(16), size_t(32)}) {
        std::vector<float16> a2(dim), b2(dim);
        for (size_t i = 0; i < dim; ++i) {
            a2[i] = static_cast<float16>(static_cast<float>(i % 7 + 1) * 0.5f - 2.0f);
            b2[i] = static_cast<float16>(static_cast<float>(i % 5) * 0.3f - 0.75f);
        }
        double b_norm_sq = reference_squared_norm(b2.data(), dim);
        const double ref = reference_cosine_distance(a2.data(), b2.data(), dim);
        const double got2 = ComputeCos_Neon::dist_f16_with_query_norm(
            reinterpret_cast<const uint8_t*>(a2.data()),
            reinterpret_cast<const uint8_t*>(b2.data()), dim, b_norm_sq);
        EXPECT_NEAR(ref, got2, 1e-2) << "dim=" << dim;
    }
}
#endif

// Tail handling: exercise the scalar tail loop for dims not a multiple of the SIMD width.
TEST(ComputeCosNeon, DotF32TailHandling) {
    const std::vector<size_t> dims = {1, 2, 3, 5, 6, 7, 9, 11, 13, 15, 17};
    for (size_t dim : dims) {
        auto a = make_buffer<float>(dim, 0);
        auto b = make_buffer<float>(dim, 0);
        fill_f32(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + 7));
        const double ref = reference_dot(a.ptr, b.ptr, dim);
        const double got = ComputeCos_Neon::dot_f32(
            reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
        EXPECT_NEAR(ref, got, std::max(1e-5, std::abs(ref) * 5e-5)) << "dim=" << dim;
    }
}

TEST(ComputeCosNeon, DistF32TailHandling) {
    const std::vector<size_t> dims = {1, 2, 3, 5, 6, 7, 9, 11, 13, 15, 17};
    for (size_t dim : dims) {
        auto a = make_buffer<float>(dim, 0);
        auto b = make_buffer<float>(dim, 0);
        fill_f32(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + 11));
        const double ref = reference_cosine_distance(a.ptr, b.ptr, dim);
        const double got = ComputeCos_Neon::dist_f32(
            reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
        EXPECT_NEAR(ref, got, 5e-5) << "dim=" << dim;
    }
}

TEST(ComputeCosNeon, DotI16TailHandling) {
    // dot_i16 SIMD width is 8; test dims with remainders 1-7.
    const std::vector<size_t> dims = {1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 15, 17};
    for (size_t dim : dims) {
        auto a = make_buffer<int16_t>(dim, 0);
        auto b = make_buffer<int16_t>(dim, 0);
        fill_i16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + 3));
        const double ref = reference_dot(a.ptr, b.ptr, dim);
        const double got = ComputeCos_Neon::dot_i16(
            reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
        EXPECT_DOUBLE_EQ(ref, got) << "dim=" << dim;
    }
}

TEST(ComputeCosNeon, DistI16TailHandling) {
    const std::vector<size_t> dims = {1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 15, 17};
    for (size_t dim : dims) {
        auto a = make_buffer<int16_t>(dim, 0);
        auto b = make_buffer<int16_t>(dim, 0);
        fill_i16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + 19));
        const double ref = reference_cosine_distance(a.ptr, b.ptr, dim);
        const double got = ComputeCos_Neon::dist_i16(
            reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
        EXPECT_NEAR(ref, got, 2e-4) << "dim=" << dim;
    }
}

#if defined(__FLT16_MANT_DIG__)
TEST(ComputeCosNeon, DotF16TailHandling) {
    if (!supports_f16()) {
        GTEST_SKIP() << "f16 is not supported on this build";
    }
    // dot_f16 SIMD width is 4 (or 8 with FP16_VECTOR_ARITHMETIC); test tail dims.
    const std::vector<size_t> dims = {1, 2, 3, 5, 6, 7, 9, 11, 13, 15, 17};
    for (size_t dim : dims) {
        auto a = make_buffer<float16>(dim, 0);
        auto b = make_buffer<float16>(dim, 0);
        fill_f16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + 5));
        const double ref = reference_dot(a.ptr, b.ptr, dim);
        const double got = ComputeCos_Neon::dot_f16(
            reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
        EXPECT_NEAR(ref, got, std::max(1e-2, std::abs(ref) * 1e-2)) << "dim=" << dim;
    }
}

TEST(ComputeCosNeon, DistF16TailHandling) {
    if (!supports_f16()) {
        GTEST_SKIP() << "f16 is not supported on this build";
    }
    const std::vector<size_t> dims = {1, 2, 3, 5, 6, 7, 9, 11, 13, 15, 17};
    for (size_t dim : dims) {
        auto a = make_buffer<float16>(dim, 0);
        auto b = make_buffer<float16>(dim, 0);
        fill_f16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + 13));
        const double ref = reference_cosine_distance(a.ptr, b.ptr, dim);
        const double got = ComputeCos_Neon::dist_f16(
            reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
        EXPECT_NEAR(ref, got, 1e-2) << "dim=" << dim;
    }
}
#endif

TEST(ComputeCosNeon, DistF32ZeroDim) {
    auto a = make_buffer<float>(1, 0);
    auto b = make_buffer<float>(1, 0);
    const double got = ComputeCos_Neon::dist_f32(
        reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), 0);
    EXPECT_DOUBLE_EQ(0.0, got);
}

TEST(ComputeCosNeon, DistI16ZeroDim) {
    auto a = make_buffer<int16_t>(1, 0);
    auto b = make_buffer<int16_t>(1, 0);
    const double got = ComputeCos_Neon::dist_i16(
        reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), 0);
    EXPECT_DOUBLE_EQ(0.0, got);
}

#if defined(__FLT16_MANT_DIG__)
TEST(ComputeCosNeon, DistF16ZeroDim) {
    if (!supports_f16()) {
        GTEST_SKIP() << "f16 is not supported on this build";
    }
    auto a = make_buffer<float16>(1, 0);
    auto b = make_buffer<float16>(1, 0);
    const double got = ComputeCos_Neon::dist_f16(
        reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), 0);
    EXPECT_DOUBLE_EQ(0.0, got);
}
#endif

TEST(ComputeCosNeon, DistI16HandlesExtremes) {
    const size_t dim = 16;
    auto a = make_buffer<int16_t>(dim, 0);
    auto b = make_buffer<int16_t>(dim, 0);
    for (size_t i = 0; i < dim; ++i) {
        a.ptr[i] = (i % 2 == 0) ? INT16_MIN : INT16_MAX;
        b.ptr[i] = (i % 2 == 0) ? INT16_MAX : INT16_MIN;
    }
    const double ref = reference_cosine_distance(a.ptr, b.ptr, dim);
    const double got = ComputeCos_Neon::dist_i16(
        reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
    EXPECT_NEAR(ref, got, 2e-4);
}

#if defined(__FLT16_MANT_DIG__)
TEST(ComputeCosNeon, DistF16HandlesExtremes) {
    if (!supports_f16()) {
        GTEST_SKIP() << "f16 is not supported on this build";
    }
    // b = -a: dot(a,b) = -||a||^2, cosine = -1, distance = 2.0.
    // All arithmetic in f32 after vcvt; 65504^2 is exactly representable in f32.
    const size_t dim = 8;
    std::vector<float16> a(dim), b(dim);
    for (size_t i = 0; i < dim; ++i) {
        a[i] = (i % 2 == 0) ? float16(65504.0f) : float16(-65504.0f);
        b[i] = (i % 2 == 0) ? float16(-65504.0f) : float16(65504.0f);
    }
    const double ref = reference_cosine_distance(a.data(), b.data(), dim);
    const double got = ComputeCos_Neon::dist_f16(reinterpret_cast<const uint8_t*>(a.data()),
                                                 reinterpret_cast<const uint8_t*>(b.data()),
                                                 dim);
    EXPECT_NEAR(2.0, ref, 1e-10);   // sanity-check the reference itself
    EXPECT_NEAR(ref, got, 1e-4);
}

TEST(ComputeCosNeon, DotF16HandlesExtremes) {
    if (!supports_f16()) {
        GTEST_SKIP() << "f16 is not supported on this build";
    }
    // dot(a, -a) = -||a||^2; all products computed in f32 after vcvt.
    const size_t dim = 8;
    std::vector<float16> a(dim), b(dim);
    for (size_t i = 0; i < dim; ++i) {
        a[i] = float16(65504.0f);
        b[i] = float16(-65504.0f);
    }
    const double ref = reference_dot(a.data(), b.data(), dim);
    const double got = ComputeCos_Neon::dot_f16(reinterpret_cast<const uint8_t*>(a.data()),
                                                reinterpret_cast<const uint8_t*>(b.data()),
                                                dim);
    EXPECT_NEAR(ref, got, std::max(1.0, std::abs(ref) * 1e-5));
}
#endif

TEST(ComputeCosNeon, DotF32LargeDim) {
    const size_t dim = 128;
    auto a = make_buffer<float>(dim, 0);
    auto b = make_buffer<float>(dim, 0);
    fill_f32(a.ptr, b.ptr, dim, 9999);
    const double ref = reference_dot(a.ptr, b.ptr, dim);
    const double got = ComputeCos_Neon::dot_f32(
        reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
    EXPECT_NEAR(ref, got, std::max(1e-4, std::abs(ref) * 5e-5));
}

TEST(ComputeCosNeon, DistF32LargeDim) {
    const size_t dim = 128;
    auto a = make_buffer<float>(dim, 0);
    auto b = make_buffer<float>(dim, 0);
    fill_f32(a.ptr, b.ptr, dim, 12345);
    const double ref = reference_cosine_distance(a.ptr, b.ptr, dim);
    const double got = ComputeCos_Neon::dist_f32(
        reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
    EXPECT_NEAR(ref, got, 5e-5);
}

TEST(ComputeCosNeon, DistI16LargeDim) {
    const size_t dim = 128;
    auto a = make_buffer<int16_t>(dim, 0);
    auto b = make_buffer<int16_t>(dim, 0);
    fill_i16(a.ptr, b.ptr, dim, 54321);
    const double ref = reference_cosine_distance(a.ptr, b.ptr, dim);
    const double got = ComputeCos_Neon::dist_i16(
        reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
    EXPECT_NEAR(ref, got, 2e-4);
}

#if defined(__FLT16_MANT_DIG__)
TEST(ComputeCosNeon, DotF16LargeDim) {
    if (!supports_f16()) {
        GTEST_SKIP() << "f16 is not supported on this build";
    }
    const size_t dim = 128;
    auto a = make_buffer<float16>(dim, 0);
    auto b = make_buffer<float16>(dim, 0);
    fill_f16(a.ptr, b.ptr, dim, 9999);
    const double ref = reference_dot(a.ptr, b.ptr, dim);
    const double got = ComputeCos_Neon::dot_f16(
        reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
    EXPECT_NEAR(ref, got, std::max(1e-2, std::abs(ref) * 1e-2));
}

TEST(ComputeCosNeon, DistF16LargeDim) {
    if (!supports_f16()) {
        GTEST_SKIP() << "f16 is not supported on this build";
    }
    const size_t dim = 128;
    auto a = make_buffer<float16>(dim, 0);
    auto b = make_buffer<float16>(dim, 0);
    fill_f16(a.ptr, b.ptr, dim, 12345);
    const double ref = reference_cosine_distance(a.ptr, b.ptr, dim);
    const double got = ComputeCos_Neon::dist_f16(
        reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
    EXPECT_NEAR(ref, got, 1e-2);
}
#endif

// Misalignment: vld1q handles unaligned loads, but test explicitly to guard against
// compiler or linker changes that might introduce alignment assumptions.
TEST(ComputeCosNeon, DotF32MisalignedMatchesReference) {
    const std::vector<size_t> dims = {1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(4)}) {
            for (size_t misalign_b : {size_t(0), size_t(8)}) {
                auto a = make_buffer<float>(dim, misalign_a);
                auto b = make_buffer<float>(dim, misalign_b);
                fill_f32(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 101));
                const double ref = reference_dot(a.ptr, b.ptr, dim);
                const double got = ComputeCos_Neon::dot_f32(
                    reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
                EXPECT_NEAR(ref, got, std::max(1e-5, std::abs(ref) * 5e-5))
                    << "dim=" << dim << " misalign_a=" << misalign_a << " misalign_b=" << misalign_b;
            }
        }
    }
}

TEST(ComputeCosNeon, DistF32MisalignedMatchesReference) {
    const std::vector<size_t> dims = {1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(4)}) {
            for (size_t misalign_b : {size_t(0), size_t(8)}) {
                auto a = make_buffer<float>(dim, misalign_a);
                auto b = make_buffer<float>(dim, misalign_b);
                fill_f32(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 3));
                const double ref = reference_cosine_distance(a.ptr, b.ptr, dim);
                const double got = ComputeCos_Neon::dist_f32(
                    reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
                EXPECT_NEAR(ref, got, 5e-5)
                    << "dim=" << dim << " misalign_a=" << misalign_a << " misalign_b=" << misalign_b;
            }
        }
    }
}

TEST(ComputeCosNeon, DotI16MisalignedMatchesReference) {
    const std::vector<size_t> dims = {1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(2)}) {
            for (size_t misalign_b : {size_t(0), size_t(6)}) {
                auto a = make_buffer<int16_t>(dim, misalign_a);
                auto b = make_buffer<int16_t>(dim, misalign_b);
                fill_i16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 151));
                const double ref = reference_dot(a.ptr, b.ptr, dim);
                const double got = ComputeCos_Neon::dot_i16(
                    reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
                EXPECT_DOUBLE_EQ(ref, got)
                    << "dim=" << dim << " misalign_a=" << misalign_a << " misalign_b=" << misalign_b;
            }
        }
    }
}

TEST(ComputeCosNeon, DistI16MisalignedMatchesReference) {
    const std::vector<size_t> dims = {1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(2)}) {
            for (size_t misalign_b : {size_t(0), size_t(6)}) {
                auto a = make_buffer<int16_t>(dim, misalign_a);
                auto b = make_buffer<int16_t>(dim, misalign_b);
                fill_i16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 19));
                const double ref = reference_cosine_distance(a.ptr, b.ptr, dim);
                const double got = ComputeCos_Neon::dist_i16(
                    reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
                EXPECT_NEAR(ref, got, 2e-4)
                    << "dim=" << dim << " misalign_a=" << misalign_a << " misalign_b=" << misalign_b;
            }
        }
    }
}

#if defined(__FLT16_MANT_DIG__)
TEST(ComputeCosNeon, DotF16MisalignedMatchesReference) {
    if (!supports_f16()) {
        GTEST_SKIP() << "f16 is not supported on this build";
    }
    const std::vector<size_t> dims = {1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(2)}) {
            for (size_t misalign_b : {size_t(0), size_t(6)}) {
                auto a = make_buffer<float16>(dim, misalign_a);
                auto b = make_buffer<float16>(dim, misalign_b);
                fill_f16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 131));
                const double ref = reference_dot(a.ptr, b.ptr, dim);
                const double got = ComputeCos_Neon::dot_f16(
                    reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
                EXPECT_NEAR(ref, got, std::max(1e-2, std::abs(ref) * 1e-2))
                    << "dim=" << dim << " misalign_a=" << misalign_a << " misalign_b=" << misalign_b;
            }
        }
    }
}

TEST(ComputeCosNeon, DistF16MisalignedMatchesReference) {
    if (!supports_f16()) {
        GTEST_SKIP() << "f16 is not supported on this build";
    }
    const std::vector<size_t> dims = {1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(2)}) {
            for (size_t misalign_b : {size_t(0), size_t(6)}) {
                auto a = make_buffer<float16>(dim, misalign_a);
                auto b = make_buffer<float16>(dim, misalign_b);
                fill_f16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 11));
                const double ref = reference_cosine_distance(a.ptr, b.ptr, dim);
                const double got = ComputeCos_Neon::dist_f16(
                    reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
                EXPECT_NEAR(ref, got, 1e-2)
                    << "dim=" << dim << " misalign_a=" << misalign_a << " misalign_b=" << misalign_b;
            }
        }
    }
}
#endif

// Dispatch verification: on aarch64, resolve_* returns NEON function pointers.
TEST(ComputeCosNeon, ResolveDistUsesNeonF32Path) {
    EXPECT_EQ(&ComputeCos_Neon::dist_f32, ComputeCos::resolve_dist(DataType::f32));
}

TEST(ComputeCosNeon, ResolveDistUsesNeonI16Path) {
    EXPECT_EQ(&ComputeCos_Neon::dist_i16, ComputeCos::resolve_dist(DataType::i16));
}

#if defined(__FLT16_MANT_DIG__)
TEST(ComputeCosNeon, ResolveDistUsesNeonF16Path) {
    EXPECT_EQ(&ComputeCos_Neon::dist_f16, ComputeCos::resolve_dist(DataType::f16));
}
#endif

TEST(ComputeCosNeon, ResolveDotUsesNeonF32Path) {
    EXPECT_EQ(&ComputeCos_Neon::dot_f32, ComputeCos::resolve_dot(DataType::f32));
}

TEST(ComputeCosNeon, ResolveDotUsesNeonI16Path) {
    EXPECT_EQ(&ComputeCos_Neon::dot_i16, ComputeCos::resolve_dot(DataType::i16));
}

TEST(ComputeCosNeon, ResolveDistWithQueryNormUsesNeonF32Path) {
    EXPECT_EQ(&ComputeCos_Neon::dist_f32_with_query_norm,
              ComputeCos::resolve_dist_with_query_norm(DataType::f32));
}

TEST(ComputeCosNeon, ResolveDistWithQueryNormUsesNeonI16Path) {
    EXPECT_EQ(&ComputeCos_Neon::dist_i16_with_query_norm,
              ComputeCos::resolve_dist_with_query_norm(DataType::i16));
}

TEST(ComputeCosNeon, ResolveSquaredNormUsesNeonF32Path) {
    EXPECT_EQ(&ComputeCos_Neon::squared_norm_f32, ComputeCos::resolve_squared_norm(DataType::f32));
}

TEST(ComputeCosNeon, ResolveSquaredNormUsesNeonI16Path) {
    EXPECT_EQ(&ComputeCos_Neon::squared_norm_i16, ComputeCos::resolve_squared_norm(DataType::i16));
}

#if defined(__FLT16_MANT_DIG__)
TEST(ComputeCosNeon, ResolveDistWithQueryNormUsesNeonF16Path) {
    EXPECT_EQ(&ComputeCos_Neon::dist_f16_with_query_norm,
              ComputeCos::resolve_dist_with_query_norm(DataType::f16));
}

TEST(ComputeCosNeon, ResolveSquaredNormUsesNeonF16Path) {
    EXPECT_EQ(&ComputeCos_Neon::squared_norm_f16, ComputeCos::resolve_squared_norm(DataType::f16));
}
#endif

#else
TEST(ComputeCosNeon, NotBuiltForThisTarget) {
    GTEST_SKIP() << "NEON is not enabled for this target";
}
#endif

#if defined(SKETCH2_COMPUTE_AVX2_TESTS)

class ComputeCosAVX2 : public ::testing::Test {
protected:
    void SetUp() override {
        if (!ComputeUnit::is_supported(ComputeBackendKind::avx2)) {
            GTEST_SKIP() << "AVX2 is not supported on this CPU";
        }
        original_ = get_singleton().compute_unit().kind();
        ASSERT_TRUE(Singleton::force_compute_unit_for_testing(ComputeBackendKind::avx2));
        overridden_ = true;
    }

    void TearDown() override {
        if (overridden_) {
            EXPECT_TRUE(Singleton::force_compute_unit_for_testing(original_));
        }
    }

private:
    ComputeBackendKind original_ = ComputeBackendKind::scalar;
    bool overridden_ = false;
};

TEST_F(ComputeCosAVX2, DotF32MatchesReferenceAlignedAndUnaligned) {
    const std::vector<size_t> dims = {1, 7, 8, 9, 31, 32, 33, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(4)}) {
            for (size_t misalign_b : {size_t(0), size_t(12)}) {
                auto a = make_buffer<float>(dim, misalign_a);
                auto b = make_buffer<float>(dim, misalign_b);
                fill_f32(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 101));

                const double ref = reference_dot(a.ptr, b.ptr, dim);
                const double got = ComputeCos_AVX2::dot_f32(reinterpret_cast<uint8_t *>(a.ptr),
                                                            reinterpret_cast<uint8_t *>(b.ptr), dim);
                EXPECT_NEAR(ref, got, 5e-5) << "dim=" << dim << " misalign_a=" << misalign_a
                                            << " misalign_b=" << misalign_b;
            }
        }
    }
}

TEST_F(ComputeCosAVX2, DistF32ZeroDimIsZero) {
    auto a = make_buffer<float>(1, 0);
    auto b = make_buffer<float>(1, 0);
    const double got = ComputeCos_AVX2::dist_f32(reinterpret_cast<uint8_t *>(a.ptr),
                                                 reinterpret_cast<uint8_t *>(b.ptr), 0);
    EXPECT_DOUBLE_EQ(0.0, got);
}

TEST_F(ComputeCosAVX2, DistF32MatchesReferenceAlignedAndUnaligned) {
    const std::vector<size_t> dims = {1, 7, 8, 9, 15, 16, 31, 32, 33, 63, 64, 65, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(4)}) {
            for (size_t misalign_b : {size_t(0), size_t(12)}) {
                auto a = make_buffer<float>(dim, misalign_a);
                auto b = make_buffer<float>(dim, misalign_b);
                fill_f32(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 3));

                const double ref = reference_cosine_distance(a.ptr, b.ptr, dim);
                const double got = ComputeCos_AVX2::dist_f32(reinterpret_cast<uint8_t *>(a.ptr),
                                                             reinterpret_cast<uint8_t *>(b.ptr), dim);
                EXPECT_NEAR(ref, got, 5e-5) << "dim=" << dim << " misalign_a=" << misalign_a
                                            << " misalign_b=" << misalign_b;
            }
        }
    }
}

#if defined(__FLT16_MANT_DIG__)
TEST_F(ComputeCosAVX2, DotF16MatchesReferenceAlignedAndUnaligned) {
    const std::vector<size_t> dims = {1, 7, 8, 9, 31, 32, 33, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(2)}) {
            for (size_t misalign_b : {size_t(0), size_t(6)}) {
                auto a = make_buffer<float16>(dim, misalign_a);
                auto b = make_buffer<float16>(dim, misalign_b);
                fill_f16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 131));

                const double ref = reference_dot(a.ptr, b.ptr, dim);
                const double got = ComputeCos_AVX2::dot_f16(reinterpret_cast<uint8_t *>(a.ptr),
                                                            reinterpret_cast<uint8_t *>(b.ptr), dim);
                EXPECT_NEAR(ref, got, 1e-2) << "dim=" << dim << " misalign_a=" << misalign_a
                                            << " misalign_b=" << misalign_b;
            }
        }
    }
}

TEST_F(ComputeCosAVX2, DistF16ZeroDimIsZero) {
    auto a = make_buffer<float16>(1, 0);
    auto b = make_buffer<float16>(1, 0);
    const double got = ComputeCos_AVX2::dist_f16(reinterpret_cast<uint8_t *>(a.ptr),
                                                 reinterpret_cast<uint8_t *>(b.ptr), 0);
    EXPECT_DOUBLE_EQ(0.0, got);
}

TEST_F(ComputeCosAVX2, DistF16MatchesReferenceAlignedAndUnaligned) {
    const std::vector<size_t> dims = {1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(2)}) {
            for (size_t misalign_b : {size_t(0), size_t(6)}) {
                auto a = make_buffer<float16>(dim, misalign_a);
                auto b = make_buffer<float16>(dim, misalign_b);
                fill_f16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 11));

                const double ref = reference_cosine_distance(a.ptr, b.ptr, dim);
                const double got = ComputeCos_AVX2::dist_f16(reinterpret_cast<uint8_t *>(a.ptr),
                                                             reinterpret_cast<uint8_t *>(b.ptr), dim);
                EXPECT_NEAR(ref, got, 1e-2) << "dim=" << dim << " misalign_a=" << misalign_a
                                            << " misalign_b=" << misalign_b;
            }
        }
    }
}

TEST_F(ComputeCosAVX2, ResolveDistUsesAVX2F16Path) {
    EXPECT_EQ(&ComputeCos_AVX2::dist_f16, ComputeCos::resolve_dist(DataType::f16));
}

TEST_F(ComputeCosAVX2, ResolveSquaredNormUsesAVX2F16Path) {
    EXPECT_EQ(&ComputeCos_AVX2::squared_norm_f16, ComputeCos::resolve_squared_norm(DataType::f16));
}
#endif

TEST_F(ComputeCosAVX2, DistI16ZeroDimIsZero) {
    auto a = make_buffer<int16_t>(1, 0);
    auto b = make_buffer<int16_t>(1, 0);
    const double got = ComputeCos_AVX2::dist_i16(reinterpret_cast<uint8_t *>(a.ptr),
                                                 reinterpret_cast<uint8_t *>(b.ptr), 0);
    EXPECT_DOUBLE_EQ(0.0, got);
}

TEST_F(ComputeCosAVX2, DotI16MatchesReferenceAlignedAndUnaligned) {
    const std::vector<size_t> dims = {1, 15, 16, 17, 31, 32, 33, 96, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(2)}) {
            for (size_t misalign_b : {size_t(0), size_t(6)}) {
                auto a = make_buffer<int16_t>(dim, misalign_a);
                auto b = make_buffer<int16_t>(dim, misalign_b);
                fill_i16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 151));

                const double ref = reference_dot(a.ptr, b.ptr, dim);
                const double got = ComputeCos_AVX2::dot_i16(reinterpret_cast<uint8_t *>(a.ptr),
                                                            reinterpret_cast<uint8_t *>(b.ptr), dim);
                EXPECT_NEAR(ref, got, 1e-9) << "dim=" << dim << " misalign_a=" << misalign_a
                                            << " misalign_b=" << misalign_b;
            }
        }
    }
}

TEST_F(ComputeCosAVX2, DistI16MatchesReferenceAlignedAndUnaligned) {
    const std::vector<size_t> dims = {1, 15, 16, 17, 31, 32, 33, 47, 48, 49, 96, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(2)}) {
            for (size_t misalign_b : {size_t(0), size_t(6)}) {
                auto a = make_buffer<int16_t>(dim, misalign_a);
                auto b = make_buffer<int16_t>(dim, misalign_b);
                fill_i16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 19));

                const double ref = reference_cosine_distance(a.ptr, b.ptr, dim);
                const double got = ComputeCos_AVX2::dist_i16(reinterpret_cast<uint8_t *>(a.ptr),
                                                             reinterpret_cast<uint8_t *>(b.ptr), dim);
                EXPECT_NEAR(ref, got, 2e-4) << "dim=" << dim << " misalign_a=" << misalign_a
                                            << " misalign_b=" << misalign_b;
            }
        }
    }
}

TEST_F(ComputeCosAVX2, ResolveSquaredNormUsesAVX2F32Path) {
    EXPECT_EQ(&ComputeCos_AVX2::squared_norm_f32, ComputeCos::resolve_squared_norm(DataType::f32));
}

TEST_F(ComputeCosAVX2, ResolveSquaredNormUsesAVX2I16Path) {
    EXPECT_EQ(&ComputeCos_AVX2::squared_norm_i16, ComputeCos::resolve_squared_norm(DataType::i16));
}

#else

TEST(ComputeCosAVX2, NotBuiltForThisTarget) {
    GTEST_SKIP() << "AVX2 is not enabled for this target";
}

#endif

} // namespace

#if defined(SKETCH2_COMPUTE_AVX512F_TESTS)

namespace {

class ComputeCosAVX512F : public ::testing::Test {
protected:
    void SetUp() override {
        if (!ComputeUnit::is_supported(ComputeBackendKind::avx512f)) {
            GTEST_SKIP() << "AVX-512F is not supported on this CPU";
        }
    }
};

TEST_F(ComputeCosAVX512F, DotF32MatchesReferenceAlignedAndUnaligned) {
    const std::vector<size_t> dims = {1, 15, 16, 17, 31, 32, 33, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(4)}) {
            for (size_t misalign_b : {size_t(0), size_t(12)}) {
                auto a = make_buffer<float>(dim, misalign_a);
                auto b = make_buffer<float>(dim, misalign_b);
                fill_f32(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 211));

                const double ref = reference_dot(a.ptr, b.ptr, dim);
                const double got = ComputeCos_AVX512::dot_f32(reinterpret_cast<uint8_t *>(a.ptr),
                                                              reinterpret_cast<uint8_t *>(b.ptr), dim);
                EXPECT_NEAR(ref, got, 5e-5) << "dim=" << dim << " misalign_a=" << misalign_a
                                            << " misalign_b=" << misalign_b;
            }
        }
    }
}

TEST_F(ComputeCosAVX512F, DistF32MatchesReferenceAlignedAndUnaligned) {
    const std::vector<size_t> dims = {1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(4)}) {
            for (size_t misalign_b : {size_t(0), size_t(12)}) {
                auto a = make_buffer<float>(dim, misalign_a);
                auto b = make_buffer<float>(dim, misalign_b);
                fill_f32(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 223));

                const double ref = reference_cosine_distance(a.ptr, b.ptr, dim);
                const double got = ComputeCos_AVX512::dist_f32(reinterpret_cast<uint8_t *>(a.ptr),
                                                               reinterpret_cast<uint8_t *>(b.ptr), dim);
                EXPECT_NEAR(ref, got, 5e-5) << "dim=" << dim << " misalign_a=" << misalign_a
                                            << " misalign_b=" << misalign_b;
            }
        }
    }
}

#if defined(__FLT16_MANT_DIG__)
TEST_F(ComputeCosAVX512F, DotF16MatchesReferenceAlignedAndUnaligned) {
    const std::vector<size_t> dims = {1, 15, 16, 17, 31, 32, 33, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(2)}) {
            for (size_t misalign_b : {size_t(0), size_t(6)}) {
                auto a = make_buffer<float16>(dim, misalign_a);
                auto b = make_buffer<float16>(dim, misalign_b);
                fill_f16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 227));

                const double ref = reference_dot(a.ptr, b.ptr, dim);
                const double got = ComputeCos_AVX512::dot_f16(reinterpret_cast<uint8_t *>(a.ptr),
                                                              reinterpret_cast<uint8_t *>(b.ptr), dim);
                EXPECT_NEAR(ref, got, 1e-2) << "dim=" << dim << " misalign_a=" << misalign_a
                                            << " misalign_b=" << misalign_b;
            }
        }
    }
}

TEST_F(ComputeCosAVX512F, DistF16MatchesReferenceAlignedAndUnaligned) {
    const std::vector<size_t> dims = {1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(2)}) {
            for (size_t misalign_b : {size_t(0), size_t(6)}) {
                auto a = make_buffer<float16>(dim, misalign_a);
                auto b = make_buffer<float16>(dim, misalign_b);
                fill_f16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 229));

                const double ref = reference_cosine_distance(a.ptr, b.ptr, dim);
                const double got = ComputeCos_AVX512::dist_f16(reinterpret_cast<uint8_t *>(a.ptr),
                                                               reinterpret_cast<uint8_t *>(b.ptr), dim);
                EXPECT_NEAR(ref, got, 1e-2) << "dim=" << dim << " misalign_a=" << misalign_a
                                            << " misalign_b=" << misalign_b;
            }
        }
    }
}
#endif

TEST_F(ComputeCosAVX512F, DotI16MatchesReferenceAlignedAndUnaligned) {
    const std::vector<size_t> dims = {1, 15, 16, 17, 31, 32, 33, 96, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(2)}) {
            for (size_t misalign_b : {size_t(0), size_t(6)}) {
                auto a = make_buffer<int16_t>(dim, misalign_a);
                auto b = make_buffer<int16_t>(dim, misalign_b);
                fill_i16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 233));

                const double ref = reference_dot(a.ptr, b.ptr, dim);
                const double got = ComputeCos_AVX512::dot_i16(reinterpret_cast<uint8_t *>(a.ptr),
                                                              reinterpret_cast<uint8_t *>(b.ptr), dim);
                EXPECT_NEAR(ref, got, 1e-9) << "dim=" << dim << " misalign_a=" << misalign_a
                                            << " misalign_b=" << misalign_b;
            }
        }
    }
}

TEST_F(ComputeCosAVX512F, DistI16MatchesReferenceAlignedAndUnaligned) {
    const std::vector<size_t> dims = {1, 15, 16, 17, 31, 32, 33, 47, 48, 49, 96, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(2)}) {
            for (size_t misalign_b : {size_t(0), size_t(6)}) {
                auto a = make_buffer<int16_t>(dim, misalign_a);
                auto b = make_buffer<int16_t>(dim, misalign_b);
                fill_i16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 239));

                const double ref = reference_cosine_distance(a.ptr, b.ptr, dim);
                const double got = ComputeCos_AVX512::dist_i16(reinterpret_cast<uint8_t *>(a.ptr),
                                                               reinterpret_cast<uint8_t *>(b.ptr), dim);
                EXPECT_NEAR(ref, got, 2e-4) << "dim=" << dim << " misalign_a=" << misalign_a
                                            << " misalign_b=" << misalign_b;
            }
        }
    }
}

} // namespace

#endif

#if defined(SKETCH2_COMPUTE_AVX512VNNI_TESTS)

namespace {

class ComputeCosAVX512VNNI : public ::testing::Test {
protected:
    void SetUp() override {
        if (!ComputeUnit::is_supported(ComputeBackendKind::avx512_vnni)) {
            GTEST_SKIP() << "AVX-512 VNNI is not supported on this CPU";
        }
    }
};

TEST_F(ComputeCosAVX512VNNI, DotI16MatchesReferenceAlignedAndUnaligned) {
    const std::vector<size_t> dims = {1, 15, 16, 17, 31, 32, 33, 96, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(2)}) {
            for (size_t misalign_b : {size_t(0), size_t(6)}) {
                auto a = make_buffer<int16_t>(dim, misalign_a);
                auto b = make_buffer<int16_t>(dim, misalign_b);
                fill_i16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 241));

                const double ref = reference_dot(a.ptr, b.ptr, dim);
                const double got = ComputeCos_AVX512_VNNI::dot_i16(reinterpret_cast<uint8_t *>(a.ptr),
                                                                   reinterpret_cast<uint8_t *>(b.ptr), dim);
                EXPECT_NEAR(ref, got, 1e-9) << "dim=" << dim << " misalign_a=" << misalign_a
                                            << " misalign_b=" << misalign_b;
            }
        }
    }
}

TEST_F(ComputeCosAVX512VNNI, DistI16MatchesReferenceAlignedAndUnaligned) {
    const std::vector<size_t> dims = {1, 15, 16, 17, 31, 32, 33, 47, 48, 49, 96, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(2)}) {
            for (size_t misalign_b : {size_t(0), size_t(6)}) {
                auto a = make_buffer<int16_t>(dim, misalign_a);
                auto b = make_buffer<int16_t>(dim, misalign_b);
                fill_i16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 251));

                const double ref = reference_cosine_distance(a.ptr, b.ptr, dim);
                const double got = ComputeCos_AVX512_VNNI::dist_i16(reinterpret_cast<uint8_t *>(a.ptr),
                                                                    reinterpret_cast<uint8_t *>(b.ptr), dim);
                EXPECT_NEAR(ref, got, 2e-4) << "dim=" << dim << " misalign_a=" << misalign_a
                                            << " misalign_b=" << misalign_b;
            }
        }
    }
}

} // namespace

#endif
