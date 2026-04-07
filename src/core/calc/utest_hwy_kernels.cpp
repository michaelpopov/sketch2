// Unit tests for Highway distance kernels.

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
#include "core/calc/calc_engine.h"
#include "core/calc/hwy_kernels.h"
#include "core/compute/utest_compute_helpers.h"

using namespace sketch2;
using namespace sketch2::test;

// ---------------------------------------------------------------------------
// DOT tests
// ---------------------------------------------------------------------------

TEST(HwyKernelsTest, DOTF32MatchesReference) {
    for (size_t dim : {1, 3, 7, 15, 17, 33, 100, 128, 257}) {
        auto ba = make_buffer<float>(dim, 0);
        auto bb = make_buffer<float>(dim, 0);
        fill_f32(ba.ptr, bb.ptr, dim, 42);
        const double expected = reference_dot(ba.ptr, bb.ptr, dim);
        const CalcKernels k = resolve_hwy_kernels(DistFunc::DOT, DataType::f32);
        const double got = k.dist(reinterpret_cast<const uint8_t*>(ba.ptr),
                                  reinterpret_cast<const uint8_t*>(bb.ptr), dim);
        EXPECT_NEAR(expected, got, 1e-4) << "dim=" << dim;
    }
}

TEST(HwyKernelsTest, DOTF16MatchesReference) {
    for (size_t dim : {1, 3, 7, 15, 17, 33, 100, 128, 257}) {
        auto ba = make_buffer<float16>(dim, 0);
        auto bb = make_buffer<float16>(dim, 0);
        fill_f16(ba.ptr, bb.ptr, dim, 42);
        const double expected = reference_dot(ba.ptr, bb.ptr, dim);
        const CalcKernels k = resolve_hwy_kernels(DistFunc::DOT, DataType::f16);
        const double got = k.dist(reinterpret_cast<const uint8_t*>(ba.ptr),
                                  reinterpret_cast<const uint8_t*>(bb.ptr), dim);
        EXPECT_NEAR(expected, got, 1e-1) << "dim=" << dim;
    }
}

TEST(HwyKernelsTest, DOTI16MatchesReference) {
    for (size_t dim : {1, 3, 7, 15, 17, 33, 100, 128, 257}) {
        auto ba = make_buffer<int16_t>(dim, 0);
        auto bb = make_buffer<int16_t>(dim, 0);
        fill_i16(ba.ptr, bb.ptr, dim, 42);
        const double expected = reference_dot(ba.ptr, bb.ptr, dim);
        const CalcKernels k = resolve_hwy_kernels(DistFunc::DOT, DataType::i16);
        const double got = k.dist(reinterpret_cast<const uint8_t*>(ba.ptr),
                                  reinterpret_cast<const uint8_t*>(bb.ptr), dim);
        EXPECT_DOUBLE_EQ(expected, got) << "dim=" << dim;
    }
}

// ---------------------------------------------------------------------------
// L2 tests
// ---------------------------------------------------------------------------

TEST(HwyKernelsTest, L2F32MatchesReference) {
    for (size_t dim : {1, 3, 7, 15, 17, 33, 100, 128, 257}) {
        auto ba = make_buffer<float>(dim, 0);
        auto bb = make_buffer<float>(dim, 0);
        fill_f32(ba.ptr, bb.ptr, dim, 42);
        const double expected = reference_l2(ba.ptr, bb.ptr, dim);
        const CalcKernels k = resolve_hwy_kernels(DistFunc::L2, DataType::f32);
        const double got = k.dist(reinterpret_cast<const uint8_t*>(ba.ptr),
                                  reinterpret_cast<const uint8_t*>(bb.ptr), dim);
        // SIMD accumulation order differs from scalar, allowing small rounding differences.
        EXPECT_NEAR(expected, got, std::max(1e-4, std::abs(expected) * 1e-6)) << "dim=" << dim;
    }
}

TEST(HwyKernelsTest, L2F16MatchesReference) {
    for (size_t dim : {1, 3, 7, 15, 17, 33, 100, 128, 257}) {
        auto ba = make_buffer<float16>(dim, 0);
        auto bb = make_buffer<float16>(dim, 0);
        fill_f16(ba.ptr, bb.ptr, dim, 42);
        const double expected = reference_l2(ba.ptr, bb.ptr, dim);
        const CalcKernels k = resolve_hwy_kernels(DistFunc::L2, DataType::f16);
        const double got = k.dist(reinterpret_cast<const uint8_t*>(ba.ptr),
                                  reinterpret_cast<const uint8_t*>(bb.ptr), dim);
        EXPECT_NEAR(expected, got, 1e-1) << "dim=" << dim;
    }
}

TEST(HwyKernelsTest, L2I16MatchesReference) {
    for (size_t dim : {1, 3, 7, 15, 17, 33, 100, 128, 257}) {
        auto ba = make_buffer<int16_t>(dim, 0);
        auto bb = make_buffer<int16_t>(dim, 0);
        fill_i16(ba.ptr, bb.ptr, dim, 42);
        const double expected = reference_l2(ba.ptr, bb.ptr, dim);
        const CalcKernels k = resolve_hwy_kernels(DistFunc::L2, DataType::i16);
        const double got = k.dist(reinterpret_cast<const uint8_t*>(ba.ptr),
                                  reinterpret_cast<const uint8_t*>(bb.ptr), dim);
        // i16 squared diffs accumulated in float may lose precision for large sums.
        EXPECT_NEAR(expected, got, std::max(1.0, std::abs(expected) * 1e-5)) << "dim=" << dim;
    }
}

// ---------------------------------------------------------------------------
// Cosine tests
// ---------------------------------------------------------------------------

TEST(HwyKernelsTest, CosDotF32MatchesReference) {
    for (size_t dim : {1, 3, 7, 15, 17, 33, 100, 128, 257}) {
        auto ba = make_buffer<float>(dim, 0);
        auto bb = make_buffer<float>(dim, 0);
        fill_f32(ba.ptr, bb.ptr, dim, 42);
        const double expected = reference_dot(ba.ptr, bb.ptr, dim);
        const CalcKernels k = resolve_hwy_kernels(DistFunc::COS, DataType::f32);
        const double got = k.dot(reinterpret_cast<const uint8_t*>(ba.ptr),
                                 reinterpret_cast<const uint8_t*>(bb.ptr), dim);
        EXPECT_NEAR(expected, got, 1e-4) << "dim=" << dim;
    }
}

TEST(HwyKernelsTest, CosSquaredNormF32MatchesReference) {
    for (size_t dim : {1, 3, 7, 15, 17, 33, 100, 128, 257}) {
        auto ba = make_buffer<float>(dim, 0);
        auto bb = make_buffer<float>(dim, 0);
        fill_f32(ba.ptr, bb.ptr, dim, 42);
        const double expected = reference_squared_norm(ba.ptr, dim);
        const CalcKernels k = resolve_hwy_kernels(DistFunc::COS, DataType::f32);
        const double got = k.squared_norm(reinterpret_cast<const uint8_t*>(ba.ptr), dim);
        EXPECT_NEAR(expected, got, std::max(1e-4, std::abs(expected) * 1e-6)) << "dim=" << dim;
    }
}

TEST(HwyKernelsTest, CosDistF32MatchesReference) {
    for (size_t dim : {1, 3, 7, 15, 17, 33, 100, 128, 257}) {
        auto ba = make_buffer<float>(dim, 0);
        auto bb = make_buffer<float>(dim, 0);
        fill_f32(ba.ptr, bb.ptr, dim, 42);
        const double expected = reference_cosine_distance(ba.ptr, bb.ptr, dim);
        const CalcKernels k = resolve_hwy_kernels(DistFunc::COS, DataType::f32);
        const double got = k.dist(reinterpret_cast<const uint8_t*>(ba.ptr),
                                  reinterpret_cast<const uint8_t*>(bb.ptr), dim);
        EXPECT_NEAR(expected, got, 1e-6) << "dim=" << dim;
    }
}

TEST(HwyKernelsTest, CosDistF16MatchesReference) {
    for (size_t dim : {1, 3, 7, 15, 17, 33, 100, 128, 257}) {
        auto ba = make_buffer<float16>(dim, 0);
        auto bb = make_buffer<float16>(dim, 0);
        fill_f16(ba.ptr, bb.ptr, dim, 42);
        const double expected = reference_cosine_distance(ba.ptr, bb.ptr, dim);
        const CalcKernels k = resolve_hwy_kernels(DistFunc::COS, DataType::f16);
        const double got = k.dist(reinterpret_cast<const uint8_t*>(ba.ptr),
                                  reinterpret_cast<const uint8_t*>(bb.ptr), dim);
        EXPECT_NEAR(expected, got, 1e-2) << "dim=" << dim;
    }
}

TEST(HwyKernelsTest, CosDistI16MatchesReference) {
    for (size_t dim : {1, 3, 7, 15, 17, 33, 100, 128, 257}) {
        auto ba = make_buffer<int16_t>(dim, 0);
        auto bb = make_buffer<int16_t>(dim, 0);
        fill_i16(ba.ptr, bb.ptr, dim, 42);
        const double expected = reference_cosine_distance(ba.ptr, bb.ptr, dim);
        const CalcKernels k = resolve_hwy_kernels(DistFunc::COS, DataType::i16);
        const double got = k.dist(reinterpret_cast<const uint8_t*>(ba.ptr),
                                  reinterpret_cast<const uint8_t*>(bb.ptr), dim);
        EXPECT_NEAR(expected, got, 1e-4) << "dim=" << dim;
    }
}

TEST(HwyKernelsTest, CosDistWithQueryNormF32MatchesReference) {
    for (size_t dim : {1, 3, 7, 15, 17, 33, 100, 128, 257}) {
        auto ba = make_buffer<float>(dim, 0);
        auto bb = make_buffer<float>(dim, 0);
        fill_f32(ba.ptr, bb.ptr, dim, 42);
        const double expected = reference_cosine_distance(ba.ptr, bb.ptr, dim);
        const CalcKernels k = resolve_hwy_kernels(DistFunc::COS, DataType::f32);
        const double query_norm_sq = k.squared_norm(reinterpret_cast<const uint8_t*>(bb.ptr), dim);
        const double got = k.dist_with_query_norm(
            reinterpret_cast<const uint8_t*>(ba.ptr),
            reinterpret_cast<const uint8_t*>(bb.ptr), dim, query_norm_sq);
        EXPECT_NEAR(expected, got, 1e-6) << "dim=" << dim;
    }
}

// ---------------------------------------------------------------------------
// resolve_calc_kernels dispatches correctly
// ---------------------------------------------------------------------------

TEST(HwyKernelsTest, ResolveCalcKernelsReturnsNonNull) {
    for (DistFunc func : {DistFunc::DOT, DistFunc::L2, DistFunc::COS}) {
        for (DataType type : {DataType::f32, DataType::f16, DataType::i16}) {
            const CalcKernels k = resolve_calc_kernels(CalcEngine::highway, func, type);
            ASSERT_NE(k.dist, nullptr) << "func=" << static_cast<int>(func)
                                       << " type=" << static_cast<int>(type);
            if (func == DistFunc::COS) {
                ASSERT_NE(k.dot, nullptr);
                ASSERT_NE(k.squared_norm, nullptr);
                ASSERT_NE(k.dist_with_query_norm, nullptr);
            }
        }
    }
}
