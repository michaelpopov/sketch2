// Unit tests for Highway distance kernels.

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "core/compute/highway.h"
#include "core/compute/utest_compute_helpers.h"
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#include "hwy/targets.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

using namespace sketch2;
using namespace sketch2::test;

namespace {

// Covers scalar-only paths, N-1/N/N+1 around every fixed f32 width emitted by
// the configured x86 targets (4, 8, 16), and several multi-vector/tail cases.
constexpr std::array<size_t, 20> kF8Dimensions = {
    1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257,
};

class RestoreHighwayTargets {
public:
    ~RestoreHighwayTargets() {
        hwy::SetSupportedTargetsForTest(0);
        hwy::GetChosenTarget().Update(hwy::SupportedTargets());
    }
};

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

TEST(HwyKernelsTest, F8FillUsesTheCanonicalBoundedCodebook) {
    constexpr size_t dim = float8_codebook::kSize;
    auto ba = make_buffer<float8>(dim, misalign_a<float8>());
    auto bb = make_buffer<float8>(dim, misalign_b<float8>());
    fill_f8(ba.ptr, bb.ptr, dim, 42);

    // f8 is byte-addressable and the buffers deliberately start at 1/3-byte
    // offsets from a 32-byte boundary, exercising unaligned vector loads.
    EXPECT_NE(0U, reinterpret_cast<uintptr_t>(ba.ptr) % 16U);
    EXPECT_NE(0U, reinterpret_cast<uintptr_t>(bb.ptr) % 16U);

    const auto& codebook = float8_codebook::bits();
    std::array<bool, float8_codebook::kSize> seen_a {};
    std::array<bool, float8_codebook::kSize> seen_b {};
    for (size_t i = 0; i < dim; ++i) {
        for (size_t index = 0; index < codebook.size(); ++index) {
            if (ba.ptr[i].to_bits() == codebook[index]) {
                seen_a[index] = true;
            }
            if (bb.ptr[i].to_bits() == codebook[index]) {
                seen_b[index] = true;
            }
        }
    }
    for (size_t index = 0; index < codebook.size(); ++index) {
        EXPECT_TRUE(seen_a[index]) << "codebook index=" << index;
        EXPECT_TRUE(seen_b[index]) << "codebook index=" << index;
    }
}

TEST(HwyKernelsTest, DOTF8MatchesDecodedReference) {
    for (size_t dim : kF8Dimensions) {
        auto ba = make_buffer<float8>(dim, misalign_a<float8>());
        auto bb = make_buffer<float8>(dim, misalign_b<float8>());
        fill_f8(ba.ptr, bb.ptr, dim, 42);
        const auto* a = reinterpret_cast<const uint8_t*>(ba.ptr);
        const auto* b = reinterpret_cast<const uint8_t*>(bb.ptr);
        const double expected = reference_f8_dot(a, b, dim);
        const ComputeKernels k = resolve_hwy_kernels(DistFunc::DOT, DataType::f8);
        const double got = k.dist(a, b, dim);
        EXPECT_NEAR(expected, got, f8_f32_accumulation_tolerance(
            reference_f8_dot_abs_sum(a, b, dim), dim)) << "dim=" << dim;
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

TEST(HwyKernelsTest, L2F8MatchesDecodedReference) {
    for (size_t dim : kF8Dimensions) {
        auto ba = make_buffer<float8>(dim, misalign_a<float8>());
        auto bb = make_buffer<float8>(dim, misalign_b<float8>());
        fill_f8(ba.ptr, bb.ptr, dim, 42);
        const auto* a = reinterpret_cast<const uint8_t*>(ba.ptr);
        const auto* b = reinterpret_cast<const uint8_t*>(bb.ptr);
        const double expected = reference_f8_l2(a, b, dim);
        const ComputeKernels k = resolve_hwy_kernels(DistFunc::L2, DataType::f8);
        const double got = k.dist(a, b, dim);
        EXPECT_NEAR(expected, got, f8_f32_accumulation_tolerance(expected, dim))
            << "dim=" << dim;
    }
}

TEST(HwyKernelsTest, L2AndCosDotF8MatchDecodedReference) {
    for (size_t dim : kF8Dimensions) {
        auto ba = make_buffer<float8>(dim, misalign_a<float8>());
        auto bb = make_buffer<float8>(dim, misalign_b<float8>());
        fill_f8(ba.ptr, bb.ptr, dim, 42);
        const auto* a = reinterpret_cast<const uint8_t*>(ba.ptr);
        const auto* b = reinterpret_cast<const uint8_t*>(bb.ptr);
        const double expected = reference_f8_dot(a, b, dim);
        const double tolerance = f8_f32_accumulation_tolerance(
            reference_f8_dot_abs_sum(a, b, dim), dim);
        const ComputeKernels l2 = resolve_hwy_kernels(DistFunc::L2, DataType::f8);
        const ComputeKernels cos = resolve_hwy_kernels(DistFunc::COS, DataType::f8);
        EXPECT_NEAR(expected, l2.dot(a, b, dim), tolerance) << "L2 dim=" << dim;
        EXPECT_NEAR(expected, cos.dot(a, b, dim), tolerance) << "COS dim=" << dim;
    }
}

TEST(HwyKernelsTest, SquaredNormF8MatchesDecodedReference) {
    for (size_t dim : kF8Dimensions) {
        auto ba = make_buffer<float8>(dim, misalign_a<float8>());
        auto bb = make_buffer<float8>(dim, misalign_b<float8>());
        fill_f8(ba.ptr, bb.ptr, dim, 42);
        const auto* a = reinterpret_cast<const uint8_t*>(ba.ptr);
        const double expected = reference_f8_squared_norm(a, dim);
        const ComputeKernels k = resolve_hwy_kernels(DistFunc::L2, DataType::f8);
        EXPECT_NEAR(expected, k.squared_norm(a, dim), f8_f32_accumulation_tolerance(expected, dim))
            << "dim=" << dim;
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

TEST(HwyKernelsTest, CosDistF8MatchesDecodedReference) {
    for (size_t dim : kF8Dimensions) {
        auto ba = make_buffer<float8>(dim, misalign_a<float8>());
        auto bb = make_buffer<float8>(dim, misalign_b<float8>());
        fill_f8(ba.ptr, bb.ptr, dim, 42);
        const auto* a = reinterpret_cast<const uint8_t*>(ba.ptr);
        const auto* b = reinterpret_cast<const uint8_t*>(bb.ptr);
        const double expected = reference_f8_cosine_distance(a, b, dim);
        const ComputeKernels k = resolve_hwy_kernels(DistFunc::COS, DataType::f8);
        EXPECT_NEAR(expected, k.dist(a, b, dim), f8_cosine_tolerance(a, b, dim))
            << "dim=" << dim;
    }
}

TEST(HwyKernelsTest, CosDistF16ZeroVectorContract) {
    expect_cos_zero_contract<float16>(DataType::f16);
}

TEST(HwyKernelsTest, CosDistI16ZeroVectorContract) {
    expect_cos_zero_contract<int16_t>(DataType::i16);
}

TEST(HwyKernelsTest, CosDistF8ZeroVectorContract) {
    expect_cos_zero_contract<float8>(DataType::f8);
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

TEST(HwyKernelsTest, CosDistWithQueryNormF8MatchesDecodedReference) {
    for (size_t dim : kF8Dimensions) {
        auto ba = make_buffer<float8>(dim, misalign_a<float8>());
        auto bb = make_buffer<float8>(dim, misalign_b<float8>());
        fill_f8(ba.ptr, bb.ptr, dim, 42);
        const auto* a = reinterpret_cast<const uint8_t*>(ba.ptr);
        const auto* b = reinterpret_cast<const uint8_t*>(bb.ptr);
        const double expected = reference_f8_cosine_distance(a, b, dim);
        const double query_norm_sq = reference_f8_squared_norm(b, dim);
        const ComputeKernels k = resolve_hwy_kernels(DistFunc::COS, DataType::f8);
        const double got = k.dist_with_query_norm(a, b, dim, query_norm_sq);
        EXPECT_NEAR(expected, got, f8_cosine_tolerance(a, b, dim)) << "dim=" << dim;
    }
}

TEST(HwyKernelsTest, F8KernelsRunOnEverySupportedGeneratedHighwayTarget) {
    constexpr size_t dim = 129;
    auto ba = make_buffer<float8>(dim, misalign_a<float8>());
    auto bb = make_buffer<float8>(dim, misalign_b<float8>());
    fill_f8(ba.ptr, bb.ptr, dim, 42);
    const auto* a = reinterpret_cast<const uint8_t*>(ba.ptr);
    const auto* b = reinterpret_cast<const uint8_t*>(bb.ptr);
    const double expected_dot = reference_f8_dot(a, b, dim);
    const double expected_l2 = reference_f8_l2(a, b, dim);
    const double expected_norm = reference_f8_squared_norm(a, dim);
    const double expected_cos = reference_f8_cosine_distance(a, b, dim);
    const double dot_tolerance = f8_f32_accumulation_tolerance(
        reference_f8_dot_abs_sum(a, b, dim), dim);
    const double l2_tolerance = f8_f32_accumulation_tolerance(expected_l2, dim);
    const double norm_tolerance = f8_f32_accumulation_tolerance(expected_norm, dim);
    const double cos_tolerance = f8_cosine_tolerance(a, b, dim);
    const double query_norm_sq = reference_f8_squared_norm(b, dim);

    hwy::SetSupportedTargetsForTest(0);
    const std::vector<int64_t> targets = hwy::SupportedAndGeneratedTargets();
    ASSERT_FALSE(targets.empty());
    RestoreHighwayTargets restore_targets;
    for (const int64_t target : targets) {
        SCOPED_TRACE(hwy::TargetName(target));
        hwy::SetSupportedTargetsForTest(target);
        hwy::GetChosenTarget().Update(hwy::SupportedTargets());

        const ComputeKernels dot = resolve_hwy_kernels(DistFunc::DOT, DataType::f8);
        const ComputeKernels l2 = resolve_hwy_kernels(DistFunc::L2, DataType::f8);
        const ComputeKernels cos = resolve_hwy_kernels(DistFunc::COS, DataType::f8);
        ASSERT_NE(dot.dist, nullptr);
        ASSERT_NE(l2.dist, nullptr);
        ASSERT_NE(l2.dot, nullptr);
        ASSERT_NE(l2.squared_norm, nullptr);
        ASSERT_NE(cos.dist, nullptr);
        ASSERT_NE(cos.dot, nullptr);
        ASSERT_NE(cos.dist_with_query_norm, nullptr);

        EXPECT_NEAR(expected_dot, dot.dist(a, b, dim), dot_tolerance);
        EXPECT_NEAR(expected_l2, l2.dist(a, b, dim), l2_tolerance);
        EXPECT_NEAR(expected_dot, l2.dot(a, b, dim), dot_tolerance);
        EXPECT_NEAR(expected_norm, l2.squared_norm(a, dim), norm_tolerance);
        EXPECT_NEAR(expected_cos, cos.dist(a, b, dim), cos_tolerance);
        EXPECT_NEAR(expected_dot, cos.dot(a, b, dim), dot_tolerance);
        EXPECT_NEAR(expected_cos, cos.dist_with_query_norm(a, b, dim, query_norm_sq),
            cos_tolerance);
    }
}

// ---------------------------------------------------------------------------
// resolve_hwy_kernels exposes the Highway-backed kernels
// ---------------------------------------------------------------------------

TEST(HwyKernelsTest, ResolveHwyKernelsReturnsNonNull) {
    for (DistFunc func : {DistFunc::DOT, DistFunc::L2, DistFunc::COS}) {
        for (DataType type : {DataType::f32, DataType::f16, DataType::i16, DataType::f8}) {
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
