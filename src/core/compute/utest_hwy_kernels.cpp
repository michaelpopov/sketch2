// Unit tests for Highway distance kernels.

#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <vector>
#include "core/compute/highway.h"
#include "core/compute/utest_compute_helpers.h"

using namespace sketch2;
using namespace sketch2::test;

namespace {

template <typename T>
constexpr size_t misalign_a() {
    return alignof(T);
}

template <typename T>
constexpr size_t misalign_b() {
    return alignof(T) * 3;
}

template <typename T>
void expect_cos_zero_contract(DataType type) {
    for (size_t dim : {1, 3, 17, 128, 257}) {
        auto zero = make_buffer<T>(dim, misalign_a<T>());
        auto nonzero = make_buffer<T>(dim, misalign_b<T>());
        std::fill_n(zero.ptr, dim, T{0});
        std::fill_n(nonzero.ptr, dim, T{1});

        const ComputeKernels k = resolve_hwy_kernels(DistFunc::COS, type);
        const auto* z = reinterpret_cast<const uint8_t*>(zero.ptr);
        const auto* n = reinterpret_cast<const uint8_t*>(nonzero.ptr);
        const double nonzero_norm_sq = k.squared_norm(n, dim);

        EXPECT_DOUBLE_EQ(0.0, k.dist(z, z, dim)) << "dim=" << dim;
        EXPECT_DOUBLE_EQ(1.0, k.dist(z, n, dim)) << "dim=" << dim;
        EXPECT_DOUBLE_EQ(1.0, k.dist(n, z, dim)) << "dim=" << dim;

        EXPECT_DOUBLE_EQ(0.0, k.dist_with_query_norm(z, z, dim, 0.0)) << "dim=" << dim;
        EXPECT_DOUBLE_EQ(1.0, k.dist_with_query_norm(z, n, dim, nonzero_norm_sq))
            << "dim=" << dim;
        EXPECT_DOUBLE_EQ(1.0, k.dist_with_query_norm(n, z, dim, 0.0)) << "dim=" << dim;
    }
}

} // namespace

// ---------------------------------------------------------------------------
// DOT tests
// ---------------------------------------------------------------------------

TEST(HwyKernelsTest, DOTF32MatchesReference) {
    for (size_t dim : {1, 3, 7, 15, 17, 33, 100, 128, 257}) {
        auto ba = make_buffer<float>(dim, misalign_a<float>());
        auto bb = make_buffer<float>(dim, misalign_b<float>());
        fill_f32(ba.ptr, bb.ptr, dim, 42);
        const double expected = reference_dot(ba.ptr, bb.ptr, dim);
        const ComputeKernels k = resolve_hwy_kernels(DistFunc::DOT, DataType::f32);
        const double got = k.dist(reinterpret_cast<const uint8_t*>(ba.ptr),
                                  reinterpret_cast<const uint8_t*>(bb.ptr), dim);
        EXPECT_NEAR(expected, got, 1e-4) << "dim=" << dim;
    }
}

TEST(HwyKernelsTest, DOTF16MatchesReference) {
    for (size_t dim : {1, 3, 7, 15, 17, 33, 100, 128, 257}) {
        auto ba = make_buffer<float16>(dim, misalign_a<float16>());
        auto bb = make_buffer<float16>(dim, misalign_b<float16>());
        fill_f16(ba.ptr, bb.ptr, dim, 42);
        const double expected = reference_dot(ba.ptr, bb.ptr, dim);
        const ComputeKernels k = resolve_hwy_kernels(DistFunc::DOT, DataType::f16);
        const double got = k.dist(reinterpret_cast<const uint8_t*>(ba.ptr),
                                  reinterpret_cast<const uint8_t*>(bb.ptr), dim);
        EXPECT_NEAR(expected, got, 1e-1) << "dim=" << dim;
    }
}

TEST(HwyKernelsTest, DOTI16MatchesReference) {
    for (size_t dim : {1, 3, 7, 15, 17, 33, 100, 128, 257}) {
        auto ba = make_buffer<int16_t>(dim, misalign_a<int16_t>());
        auto bb = make_buffer<int16_t>(dim, misalign_b<int16_t>());
        fill_i16(ba.ptr, bb.ptr, dim, 42);
        const double expected = reference_dot(ba.ptr, bb.ptr, dim);
        const ComputeKernels k = resolve_hwy_kernels(DistFunc::DOT, DataType::i16);
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
        auto ba = make_buffer<float>(dim, misalign_a<float>());
        auto bb = make_buffer<float>(dim, misalign_b<float>());
        fill_f32(ba.ptr, bb.ptr, dim, 42);
        const double expected = reference_l2(ba.ptr, bb.ptr, dim);
        const ComputeKernels k = resolve_hwy_kernels(DistFunc::L2, DataType::f32);
        const double got = k.dist(reinterpret_cast<const uint8_t*>(ba.ptr),
                                  reinterpret_cast<const uint8_t*>(bb.ptr), dim);
        // SIMD accumulation order differs from scalar, allowing small rounding differences.
        EXPECT_NEAR(expected, got, std::max(1e-4, std::abs(expected) * 1e-6)) << "dim=" << dim;
    }
}

TEST(HwyKernelsTest, L2F16MatchesReference) {
    for (size_t dim : {1, 3, 7, 15, 17, 33, 100, 128, 257}) {
        auto ba = make_buffer<float16>(dim, misalign_a<float16>());
        auto bb = make_buffer<float16>(dim, misalign_b<float16>());
        fill_f16(ba.ptr, bb.ptr, dim, 42);
        const double expected = reference_l2(ba.ptr, bb.ptr, dim);
        const ComputeKernels k = resolve_hwy_kernels(DistFunc::L2, DataType::f16);
        const double got = k.dist(reinterpret_cast<const uint8_t*>(ba.ptr),
                                  reinterpret_cast<const uint8_t*>(bb.ptr), dim);
        EXPECT_NEAR(expected, got, 1e-1) << "dim=" << dim;
    }
}

TEST(HwyKernelsTest, L2I16MatchesReference) {
    for (size_t dim : {1, 3, 7, 15, 17, 33, 100, 128, 257}) {
        auto ba = make_buffer<int16_t>(dim, misalign_a<int16_t>());
        auto bb = make_buffer<int16_t>(dim, misalign_b<int16_t>());
        fill_i16(ba.ptr, bb.ptr, dim, 42);
        const double expected = reference_l2(ba.ptr, bb.ptr, dim);
        const ComputeKernels k = resolve_hwy_kernels(DistFunc::L2, DataType::i16);
        const double got = k.dist(reinterpret_cast<const uint8_t*>(ba.ptr),
                                  reinterpret_cast<const uint8_t*>(bb.ptr), dim);
        // i16 squared diffs are integers accumulated exactly in int64, so the
        // result matches the reference bit-for-bit.
        EXPECT_DOUBLE_EQ(expected, got) << "dim=" << dim;
    }
}

// ---------------------------------------------------------------------------
// Cosine tests
// ---------------------------------------------------------------------------

TEST(HwyKernelsTest, CosDotF32MatchesReference) {
    for (size_t dim : {1, 3, 7, 15, 17, 33, 100, 128, 257}) {
        auto ba = make_buffer<float>(dim, misalign_a<float>());
        auto bb = make_buffer<float>(dim, misalign_b<float>());
        fill_f32(ba.ptr, bb.ptr, dim, 42);
        const double expected = reference_dot(ba.ptr, bb.ptr, dim);
        const ComputeKernels k = resolve_hwy_kernels(DistFunc::COS, DataType::f32);
        const double got = k.dot(reinterpret_cast<const uint8_t*>(ba.ptr),
                                 reinterpret_cast<const uint8_t*>(bb.ptr), dim);
        EXPECT_NEAR(expected, got, 1e-4) << "dim=" << dim;
    }
}

TEST(HwyKernelsTest, CosSquaredNormF32MatchesReference) {
    for (size_t dim : {1, 3, 7, 15, 17, 33, 100, 128, 257}) {
        auto ba = make_buffer<float>(dim, misalign_a<float>());
        auto bb = make_buffer<float>(dim, misalign_b<float>());
        fill_f32(ba.ptr, bb.ptr, dim, 42);
        const double expected = reference_squared_norm(ba.ptr, dim);
        const ComputeKernels k = resolve_hwy_kernels(DistFunc::COS, DataType::f32);
        const double got = k.squared_norm(reinterpret_cast<const uint8_t*>(ba.ptr), dim);
        EXPECT_NEAR(expected, got, std::max(1e-4, std::abs(expected) * 1e-6)) << "dim=" << dim;
    }
}

TEST(HwyKernelsTest, CosDistF32MatchesReference) {
    for (size_t dim : {1, 3, 7, 15, 17, 33, 100, 128, 257}) {
        auto ba = make_buffer<float>(dim, misalign_a<float>());
        auto bb = make_buffer<float>(dim, misalign_b<float>());
        fill_f32(ba.ptr, bb.ptr, dim, 42);
        const double expected = reference_cosine_distance(ba.ptr, bb.ptr, dim);
        const ComputeKernels k = resolve_hwy_kernels(DistFunc::COS, DataType::f32);
        const double got = k.dist(reinterpret_cast<const uint8_t*>(ba.ptr),
                                  reinterpret_cast<const uint8_t*>(bb.ptr), dim);
        EXPECT_NEAR(expected, got, 1e-6) << "dim=" << dim;
    }
}

TEST(HwyKernelsTest, CosDistF16MatchesReference) {
    for (size_t dim : {1, 3, 7, 15, 17, 33, 100, 128, 257}) {
        auto ba = make_buffer<float16>(dim, misalign_a<float16>());
        auto bb = make_buffer<float16>(dim, misalign_b<float16>());
        fill_f16(ba.ptr, bb.ptr, dim, 42);
        const double expected = reference_cosine_distance(ba.ptr, bb.ptr, dim);
        const ComputeKernels k = resolve_hwy_kernels(DistFunc::COS, DataType::f16);
        const double got = k.dist(reinterpret_cast<const uint8_t*>(ba.ptr),
                                  reinterpret_cast<const uint8_t*>(bb.ptr), dim);
        EXPECT_NEAR(expected, got, 1e-2) << "dim=" << dim;
    }
}

TEST(HwyKernelsTest, CosDistI16MatchesReference) {
    for (size_t dim : {1, 3, 7, 15, 17, 33, 100, 128, 257}) {
        auto ba = make_buffer<int16_t>(dim, misalign_a<int16_t>());
        auto bb = make_buffer<int16_t>(dim, misalign_b<int16_t>());
        fill_i16(ba.ptr, bb.ptr, dim, 42);
        const double expected = reference_cosine_distance(ba.ptr, bb.ptr, dim);
        const ComputeKernels k = resolve_hwy_kernels(DistFunc::COS, DataType::i16);
        const double got = k.dist(reinterpret_cast<const uint8_t*>(ba.ptr),
                                  reinterpret_cast<const uint8_t*>(bb.ptr), dim);
        EXPECT_NEAR(expected, got, 1e-4) << "dim=" << dim;
    }
}

TEST(HwyKernelsTest, CosDistF16ZeroVectorContract) {
    expect_cos_zero_contract<float16>(DataType::f16);
}

TEST(HwyKernelsTest, CosDistI16ZeroVectorContract) {
    expect_cos_zero_contract<int16_t>(DataType::i16);
}

TEST(HwyKernelsTest, CosDistWithQueryNormF32MatchesReference) {
    for (size_t dim : {1, 3, 7, 15, 17, 33, 100, 128, 257}) {
        auto ba = make_buffer<float>(dim, misalign_a<float>());
        auto bb = make_buffer<float>(dim, misalign_b<float>());
        fill_f32(ba.ptr, bb.ptr, dim, 42);
        const double expected = reference_cosine_distance(ba.ptr, bb.ptr, dim);
        const ComputeKernels k = resolve_hwy_kernels(DistFunc::COS, DataType::f32);
        const double query_norm_sq = k.squared_norm(reinterpret_cast<const uint8_t*>(bb.ptr), dim);
        const double got = k.dist_with_query_norm(
            reinterpret_cast<const uint8_t*>(ba.ptr),
            reinterpret_cast<const uint8_t*>(bb.ptr), dim, query_norm_sq);
        EXPECT_NEAR(expected, got, 1e-6) << "dim=" << dim;
    }
}

// ---------------------------------------------------------------------------
// resolve_hwy_kernels exposes the Highway-backed kernels
// ---------------------------------------------------------------------------

TEST(HwyKernelsTest, ResolveHwyKernelsReturnsNonNull) {
    for (DistFunc func : {DistFunc::DOT, DistFunc::L2, DistFunc::COS}) {
        for (DataType type : {DataType::f32, DataType::f16, DataType::i16}) {
            const ComputeKernels k = resolve_hwy_kernels(func, type);
            ASSERT_NE(k.dist, nullptr) << "func=" << static_cast<int>(func)
                                       << " type=" << static_cast<int>(type);
            if (func == DistFunc::L2 || func == DistFunc::COS) {
                ASSERT_NE(k.dot, nullptr);
                ASSERT_NE(k.squared_norm, nullptr);
            }
            if (func == DistFunc::COS) {
                ASSERT_NE(k.dist_with_query_norm, nullptr);
            }
        }
    }
}
