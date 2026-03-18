// Unit tests for L2-distance compute implementations.

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/compute/utest_compute_helpers.h"
#include "core/compute/compute_l2.h"
#include "core/compute/compute_l2_avx2.h"
#include "core/compute/compute_l2_avx512.h"
#include "core/compute/compute_l2_neon.h"

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

#if defined(__aarch64__)
TEST(ComputeL2Neon, DistF32MatchesReference) {
    const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, -5.0f};
    const std::vector<float> b = {0.0f, 2.5f, 1.0f, 0.0f, -1.0f};
    const double got = ComputeL2_Neon::dist_f32(reinterpret_cast<const uint8_t*>(a.data()),
                                                reinterpret_cast<const uint8_t*>(b.data()),
                                                a.size());
    EXPECT_DOUBLE_EQ(37.25, got);

    // Also verify across multiple SIMD widths (f32 SIMD width = 4).
    for (size_t dim : {size_t(16), size_t(32)}) {
        std::vector<float> a2(dim), b2(dim);
        for (size_t i = 0; i < dim; ++i) {
            a2[i] = static_cast<float>(i % 7 + 1) * 0.5f - 2.0f;
            b2[i] = static_cast<float>(i % 5) * 0.3f - 0.75f;
        }
        double ref = reference_l2(a2.data(), b2.data(), dim);
        const double got2 = ComputeL2_Neon::dist_f32(
            reinterpret_cast<const uint8_t*>(a2.data()),
            reinterpret_cast<const uint8_t*>(b2.data()), dim);
        EXPECT_NEAR(ref, got2, std::max(1e-5, ref * 1e-5)) << "dim=" << dim;
    }
}

TEST(ComputeL2Neon, DistF16MatchesReference) {
    if (!supports_f16()) {
        GTEST_SKIP() << "f16 is not supported on this build";
    }

    const std::vector<float16> a = {
        float16(1.0f), float16(2.5f), float16(-3.0f), float16(4.25f),
        float16(-5.5f), float16(6.0f), float16(7.5f), float16(-8.0f),
        float16(9.0f)
    };
    const std::vector<float16> b = {
        float16(0.5f), float16(2.0f), float16(-1.0f), float16(0.25f),
        float16(-1.5f), float16(5.0f), float16(8.5f), float16(-9.0f),
        float16(7.0f)
    };
    const double got = ComputeL2_Neon::dist_f16(reinterpret_cast<const uint8_t*>(a.data()),
                                                reinterpret_cast<const uint8_t*>(b.data()),
                                                a.size());
    const double ref = reference_l2(a.data(), b.data(), a.size());
    EXPECT_NEAR(ref, got, std::max(1e-3, ref * 2e-3));

    // Also verify across multiple SIMD widths (f16 SIMD width = 4 or 8).
    for (size_t dim : {size_t(16), size_t(32)}) {
        std::vector<float16> a2(dim), b2(dim);
        for (size_t i = 0; i < dim; ++i) {
            a2[i] = static_cast<float16>(static_cast<float>(i % 7 + 1) * 0.5f - 2.0f);
            b2[i] = static_cast<float16>(static_cast<float>(i % 5) * 0.3f - 0.75f);
        }
        const double ref2 = reference_l2(a2.data(), b2.data(), dim);
        const double got2 = ComputeL2_Neon::dist_f16(
            reinterpret_cast<const uint8_t*>(a2.data()),
            reinterpret_cast<const uint8_t*>(b2.data()), dim);
        EXPECT_NEAR(ref2, got2, std::max(1e-2, ref2 * 2e-3)) << "dim=" << dim;
    }
}

TEST(ComputeL2Neon, DistI16MatchesReference) {
    const std::vector<int16_t> a = {10, -2, 7, -8, 20};
    const std::vector<int16_t> b = {4, -5, 10, -8, 18};
    const double got = ComputeL2_Neon::dist_i16(reinterpret_cast<const uint8_t*>(a.data()),
                                                reinterpret_cast<const uint8_t*>(b.data()),
                                                a.size());
    EXPECT_DOUBLE_EQ(58.0, got);

    // Also verify across multiple SIMD widths (i16 SIMD width = 8).
    for (size_t dim : {size_t(16), size_t(32)}) {
        std::vector<int16_t> a2(dim), b2(dim);
        for (size_t i = 0; i < dim; ++i) {
            a2[i] = static_cast<int16_t>(static_cast<int>((i * 13 + 7) % 200) - 100);
            b2[i] = static_cast<int16_t>(static_cast<int>((i * 17 + 3) % 200) - 100);
        }
        double ref = reference_l2(a2.data(), b2.data(), dim);
        const double got2 = ComputeL2_Neon::dist_i16(
            reinterpret_cast<const uint8_t*>(a2.data()),
            reinterpret_cast<const uint8_t*>(b2.data()), dim);
        EXPECT_DOUBLE_EQ(ref, got2) << "dim=" << dim;
    }
}

// Tail handling: exercise the scalar tail loop for dims not a multiple of SIMD width.
TEST(ComputeL2Neon, DistF32TailHandling) {
    const std::vector<size_t> dims = {1, 2, 3, 5, 6, 7, 9, 11, 13, 15, 17};
    for (size_t dim : dims) {
        auto a = make_buffer<float>(dim, 0);
        auto b = make_buffer<float>(dim, 0);
        fill_f32(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + 7));
        const double ref = reference_l2(a.ptr, b.ptr, dim);
        const double got = ComputeL2_Neon::dist_f32(
            reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
        EXPECT_NEAR(ref, got, std::max(1e-5, ref * 1e-5)) << "dim=" << dim;
    }
}

TEST(ComputeL2Neon, DistI16TailHandling) {
    // dist_i16 SIMD width is 8; test dims with remainders 1-7.
    const std::vector<size_t> dims = {1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 15, 17};
    for (size_t dim : dims) {
        auto a = make_buffer<int16_t>(dim, 0);
        auto b = make_buffer<int16_t>(dim, 0);
        fill_i16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + 3));
        const double ref = reference_l2(a.ptr, b.ptr, dim);
        const double got = ComputeL2_Neon::dist_i16(
            reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
        EXPECT_DOUBLE_EQ(ref, got) << "dim=" << dim;
    }
}

#if defined(__FLT16_MANT_DIG__)
TEST(ComputeL2Neon, DistF16TailHandling) {
    if (!supports_f16()) {
        GTEST_SKIP() << "f16 is not supported on this build";
    }
    // dist_f16 SIMD width is 4 (or 8 with FP16_VECTOR_ARITHMETIC); test tail dims.
    const std::vector<size_t> dims = {1, 2, 3, 5, 6, 7, 9, 11, 13, 15, 17};
    for (size_t dim : dims) {
        auto a = make_buffer<float16>(dim, 0);
        auto b = make_buffer<float16>(dim, 0);
        fill_f16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + 5));
        const double ref = reference_l2(a.ptr, b.ptr, dim);
        const double got = ComputeL2_Neon::dist_f16(
            reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
        EXPECT_NEAR(ref, got, std::max(1e-2, ref * 2e-3)) << "dim=" << dim;
    }
}
#endif

TEST(ComputeL2Neon, DistF32ZeroDim) {
    auto a = make_buffer<float>(1, 0);
    auto b = make_buffer<float>(1, 0);
    const double got = ComputeL2_Neon::dist_f32(
        reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), 0);
    EXPECT_DOUBLE_EQ(0.0, got);
}

TEST(ComputeL2Neon, DistI16ZeroDim) {
    auto a = make_buffer<int16_t>(1, 0);
    auto b = make_buffer<int16_t>(1, 0);
    const double got = ComputeL2_Neon::dist_i16(
        reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), 0);
    EXPECT_DOUBLE_EQ(0.0, got);
}

#if defined(__FLT16_MANT_DIG__)
TEST(ComputeL2Neon, DistF16ZeroDim) {
    if (!supports_f16()) {
        GTEST_SKIP() << "f16 is not supported on this build";
    }
    auto a = make_buffer<float16>(1, 0);
    auto b = make_buffer<float16>(1, 0);
    const double got = ComputeL2_Neon::dist_f16(
        reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), 0);
    EXPECT_DOUBLE_EQ(0.0, got);
}
#endif

TEST(ComputeL2Neon, DistI16HandlesExtremes) {
    const size_t dim = 16;
    auto a = make_buffer<int16_t>(dim, 0);
    auto b = make_buffer<int16_t>(dim, 0);
    for (size_t i = 0; i < dim; ++i) {
        a.ptr[i] = (i % 2 == 0) ? INT16_MIN : INT16_MAX;
        b.ptr[i] = (i % 2 == 0) ? INT16_MAX : INT16_MIN;
    }
    const double ref = reference_l2(a.ptr, b.ptr, dim);
    const double got = ComputeL2_Neon::dist_i16(
        reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
    EXPECT_DOUBLE_EQ(ref, got);
}

#if defined(__FLT16_MANT_DIG__)
TEST(ComputeL2Neon, DistF16HandlesExtremes) {
    if (!supports_f16()) {
        GTEST_SKIP() << "f16 is not supported on this build";
    }
    // Use a[i] = ±32752, b[i] = ∓32752 so the f16 difference is ±65504 (f16 max),
    // which stays in range for vsubq_f16 in the __ARM_FEATURE_FP16_VECTOR_ARITHMETIC path.
    // 32752 = 2^15 - 16 is exactly representable in f16 (spacing at 2^14 is 16).
    const size_t dim = 8;
    std::vector<float16> a(dim), b(dim);
    for (size_t i = 0; i < dim; ++i) {
        a[i] = (i % 2 == 0) ? float16(32752.0f) : float16(-32752.0f);
        b[i] = (i % 2 == 0) ? float16(-32752.0f) : float16(32752.0f);
    }
    const double ref = reference_l2(a.data(), b.data(), dim);
    const double got = ComputeL2_Neon::dist_f16(reinterpret_cast<const uint8_t*>(a.data()),
                                                reinterpret_cast<const uint8_t*>(b.data()),
                                                dim);
    EXPECT_NEAR(ref, got, std::max(1.0, ref * 1e-5));
}
#endif

#if defined(__FLT16_MANT_DIG__)
TEST(ComputeL2Neon, DistF16LargeDim) {
    if (!supports_f16()) {
        GTEST_SKIP() << "f16 is not supported on this build";
    }
    const size_t dim = 512;
    auto a = make_buffer<float16>(dim, 0);
    auto b = make_buffer<float16>(dim, 0);
    fill_f16(a.ptr, b.ptr, dim, 9012);
    const double ref = reference_l2(a.ptr, b.ptr, dim);
    const double got = ComputeL2_Neon::dist_f16(
        reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
    EXPECT_NEAR(ref, got, std::max(1e-2, ref * 2e-3));
}
#endif

TEST(ComputeL2Neon, DistF32LargeDim) {
    const size_t dim = 512;
    auto a = make_buffer<float>(dim, 0);
    auto b = make_buffer<float>(dim, 0);
    fill_f32(a.ptr, b.ptr, dim, 1234);
    const double ref = reference_l2(a.ptr, b.ptr, dim);
    const double got = ComputeL2_Neon::dist_f32(
        reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
    EXPECT_NEAR(ref, got, std::max(1e-4, ref * 1e-5));
}

TEST(ComputeL2Neon, DistI16LargeDim) {
    const size_t dim = 512;
    auto a = make_buffer<int16_t>(dim, 0);
    auto b = make_buffer<int16_t>(dim, 0);
    fill_i16(a.ptr, b.ptr, dim, 5678);
    const double ref = reference_l2(a.ptr, b.ptr, dim);
    const double got = ComputeL2_Neon::dist_i16(
        reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
    EXPECT_DOUBLE_EQ(ref, got);
}

// Misalignment: vld1q handles unaligned loads, but test explicitly to guard against
// compiler or linker changes that might introduce alignment assumptions.
TEST(ComputeL2Neon, DistF32MisalignedMatchesReference) {
    const std::vector<size_t> dims = {1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(4)}) {
            for (size_t misalign_b : {size_t(0), size_t(8)}) {
                auto a = make_buffer<float>(dim, misalign_a);
                auto b = make_buffer<float>(dim, misalign_b);
                fill_f32(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 29));
                const double ref = reference_l2(a.ptr, b.ptr, dim);
                const double got = ComputeL2_Neon::dist_f32(
                    reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
                EXPECT_NEAR(ref, got, std::max(1e-5, ref * 1e-5))
                    << "dim=" << dim << " misalign_a=" << misalign_a << " misalign_b=" << misalign_b;
            }
        }
    }
}

TEST(ComputeL2Neon, DistI16MisalignedMatchesReference) {
    const std::vector<size_t> dims = {1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(2)}) {
            for (size_t misalign_b : {size_t(0), size_t(6)}) {
                auto a = make_buffer<int16_t>(dim, misalign_a);
                auto b = make_buffer<int16_t>(dim, misalign_b);
                fill_i16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 23));
                const double ref = reference_l2(a.ptr, b.ptr, dim);
                const double got = ComputeL2_Neon::dist_i16(
                    reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
                EXPECT_DOUBLE_EQ(ref, got)
                    << "dim=" << dim << " misalign_a=" << misalign_a << " misalign_b=" << misalign_b;
            }
        }
    }
}

#if defined(__FLT16_MANT_DIG__)
TEST(ComputeL2Neon, DistF16MisalignedMatchesReference) {
    if (!supports_f16()) {
        GTEST_SKIP() << "f16 is not supported on this build";
    }
    const std::vector<size_t> dims = {1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(2)}) {
            for (size_t misalign_b : {size_t(0), size_t(6)}) {
                auto a = make_buffer<float16>(dim, misalign_a);
                auto b = make_buffer<float16>(dim, misalign_b);
                fill_f16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 17));
                const double ref = reference_l2(a.ptr, b.ptr, dim);
                const double got = ComputeL2_Neon::dist_f16(
                    reinterpret_cast<uint8_t*>(a.ptr), reinterpret_cast<uint8_t*>(b.ptr), dim);
                EXPECT_NEAR(ref, got, std::max(1e-2, ref * 2e-3))
                    << "dim=" << dim << " misalign_a=" << misalign_a << " misalign_b=" << misalign_b;
            }
        }
    }
}
#endif

// Dispatch verification: on aarch64, resolve_dist returns NEON function pointers.
TEST(ComputeL2Neon, ResolveDistUsesNeonF32Path) {
    EXPECT_EQ(&ComputeL2_Neon::dist_f32, ComputeL2::resolve_dist(DataType::f32));
}

TEST(ComputeL2Neon, ResolveDistUsesNeonI16Path) {
    EXPECT_EQ(&ComputeL2_Neon::dist_i16, ComputeL2::resolve_dist(DataType::i16));
}

#if defined(__FLT16_MANT_DIG__)
TEST(ComputeL2Neon, ResolveDistUsesNeonF16Path) {
    EXPECT_EQ(&ComputeL2_Neon::dist_f16, ComputeL2::resolve_dist(DataType::f16));
}
#endif

#else
TEST(ComputeL2Neon, NotBuiltForThisTarget) {
    GTEST_SKIP() << "NEON is not enabled for this target";
}
#endif

#if defined(SKETCH2_COMPUTE_AVX2_TESTS)

class ComputeL2AVX2 : public ::testing::Test {
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

#if defined(__FLT16_MANT_DIG__)
TEST_F(ComputeL2AVX2, DistF16ZeroDimIsZero) {
    auto a = make_buffer<float16>(1, 0);
    auto b = make_buffer<float16>(1, 0);
    const double got = ComputeL2_AVX2::dist_f16(reinterpret_cast<uint8_t *>(a.ptr),
                                                reinterpret_cast<uint8_t *>(b.ptr), 0);
    EXPECT_DOUBLE_EQ(0.0, got);
}

TEST_F(ComputeL2AVX2, DistF16MatchesReferenceAlignedAndUnaligned) {
    const std::vector<size_t> dims = {1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(2)}) {
            for (size_t misalign_b : {size_t(0), size_t(6)}) {
                auto a = make_buffer<float16>(dim, misalign_a);
                auto b = make_buffer<float16>(dim, misalign_b);
                fill_f16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 17));

                const double ref = reference_l2(a.ptr, b.ptr, dim);
                const double got = ComputeL2_AVX2::dist_f16(reinterpret_cast<uint8_t *>(a.ptr),
                                                            reinterpret_cast<uint8_t *>(b.ptr), dim);
                EXPECT_NEAR(ref, got, std::max(1e-2, ref * 2e-3)) << "dim=" << dim
                                                                   << " misalign_a=" << misalign_a
                                                                   << " misalign_b=" << misalign_b;
            }
        }
    }
}

TEST_F(ComputeL2AVX2, ResolveDistUsesAVX2F16Path) {
    EXPECT_EQ(&ComputeL2_AVX2::dist_f16, ComputeL2::resolve_dist(DataType::f16));
}
#endif

TEST_F(ComputeL2AVX2, DistF32MatchesReferenceAlignedAndUnaligned) {
    const std::vector<size_t> dims = {1, 7, 8, 9, 15, 16, 31, 32, 33, 63, 64, 65, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(4)}) {
            for (size_t misalign_b : {size_t(0), size_t(12)}) {
                auto a = make_buffer<float>(dim, misalign_a);
                auto b = make_buffer<float>(dim, misalign_b);
                fill_f32(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 29));

                const double ref = reference_l2(a.ptr, b.ptr, dim);
                const double got = ComputeL2_AVX2::dist_f32(reinterpret_cast<uint8_t *>(a.ptr),
                                                            reinterpret_cast<uint8_t *>(b.ptr), dim);
                EXPECT_NEAR(ref, got, std::max(1e-5, ref * 1e-5)) << "dim=" << dim
                                                                   << " misalign_a=" << misalign_a
                                                                   << " misalign_b=" << misalign_b;
            }
        }
    }
}

TEST_F(ComputeL2AVX2, ResolveDistUsesAVX2F32Path) {
    EXPECT_EQ(&ComputeL2_AVX2::dist_f32, ComputeL2::resolve_dist(DataType::f32));
}

TEST_F(ComputeL2AVX2, DistI16ZeroDimIsZero) {
    auto a = make_buffer<int16_t>(1, 0);
    auto b = make_buffer<int16_t>(1, 0);
    const double got = ComputeL2_AVX2::dist_i16(reinterpret_cast<uint8_t *>(a.ptr),
                                                reinterpret_cast<uint8_t *>(b.ptr), 0);
    EXPECT_DOUBLE_EQ(0.0, got);
}

TEST_F(ComputeL2AVX2, DistI16MatchesReferenceAlignedAndUnaligned) {
    const std::vector<size_t> dims = {1, 15, 16, 17, 31, 32, 33, 47, 48, 49, 96, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(2)}) {
            for (size_t misalign_b : {size_t(0), size_t(6)}) {
                auto a = make_buffer<int16_t>(dim, misalign_a);
                auto b = make_buffer<int16_t>(dim, misalign_b);
                fill_i16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 23));

                const double ref = reference_l2(a.ptr, b.ptr, dim);
                const double got = ComputeL2_AVX2::dist_i16(reinterpret_cast<uint8_t *>(a.ptr),
                                                            reinterpret_cast<uint8_t *>(b.ptr), dim);
                EXPECT_DOUBLE_EQ(ref, got) << "dim=" << dim << " misalign_a=" << misalign_a
                                           << " misalign_b=" << misalign_b;
            }
        }
    }
}

TEST_F(ComputeL2AVX2, DistI16HandlesExtremes) {
    const size_t dim = 64;
    auto a = make_buffer<int16_t>(dim, 0);
    auto b = make_buffer<int16_t>(dim, 0);
    for (size_t i = 0; i < dim; ++i) {
        a.ptr[i] = (i % 2 == 0) ? INT16_MIN : INT16_MAX;
        b.ptr[i] = (i % 2 == 0) ? INT16_MAX : INT16_MIN;
    }

    const double ref = reference_l2(a.ptr, b.ptr, dim);
    const double got = ComputeL2_AVX2::dist_i16(reinterpret_cast<uint8_t *>(a.ptr),
                                                reinterpret_cast<uint8_t *>(b.ptr), dim);
    EXPECT_DOUBLE_EQ(ref, got);
}

#else

TEST(ComputeL2AVX2, NotBuiltForThisTarget) {
    GTEST_SKIP() << "AVX2 is not enabled for this target";
}

#endif

} // namespace

#if defined(SKETCH2_COMPUTE_AVX512F_TESTS)

namespace {

class ComputeL2AVX512F : public ::testing::Test {
protected:
    void SetUp() override {
        if (!ComputeUnit::is_supported(ComputeBackendKind::avx512f)) {
            GTEST_SKIP() << "AVX-512F is not supported on this CPU";
        }
    }
};

TEST_F(ComputeL2AVX512F, DistF32MatchesReferenceAlignedAndUnaligned) {
    const std::vector<size_t> dims = {1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(4)}) {
            for (size_t misalign_b : {size_t(0), size_t(12)}) {
                auto a = make_buffer<float>(dim, misalign_a);
                auto b = make_buffer<float>(dim, misalign_b);
                fill_f32(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 71));

                const double ref = reference_l2(a.ptr, b.ptr, dim);
                const double got = ComputeL2_AVX512::dist_f32(reinterpret_cast<uint8_t *>(a.ptr),
                                                              reinterpret_cast<uint8_t *>(b.ptr), dim);
                EXPECT_NEAR(ref, got, std::max(1e-5, ref * 1e-5)) << "dim=" << dim
                                                                   << " misalign_a=" << misalign_a
                                                                   << " misalign_b=" << misalign_b;
            }
        }
    }
}

#if defined(__FLT16_MANT_DIG__)
TEST_F(ComputeL2AVX512F, DistF16MatchesReferenceAlignedAndUnaligned) {
    const std::vector<size_t> dims = {1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(2)}) {
            for (size_t misalign_b : {size_t(0), size_t(6)}) {
                auto a = make_buffer<float16>(dim, misalign_a);
                auto b = make_buffer<float16>(dim, misalign_b);
                fill_f16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 73));

                const double ref = reference_l2(a.ptr, b.ptr, dim);
                const double got = ComputeL2_AVX512::dist_f16(reinterpret_cast<uint8_t *>(a.ptr),
                                                              reinterpret_cast<uint8_t *>(b.ptr), dim);
                EXPECT_NEAR(ref, got, std::max(1e-2, ref * 2e-3)) << "dim=" << dim
                                                                   << " misalign_a=" << misalign_a
                                                                   << " misalign_b=" << misalign_b;
            }
        }
    }
}
#endif

TEST_F(ComputeL2AVX512F, DistI16MatchesReferenceAlignedAndUnaligned) {
    const std::vector<size_t> dims = {1, 15, 16, 17, 31, 32, 33, 47, 48, 49, 96, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(2)}) {
            for (size_t misalign_b : {size_t(0), size_t(6)}) {
                auto a = make_buffer<int16_t>(dim, misalign_a);
                auto b = make_buffer<int16_t>(dim, misalign_b);
                fill_i16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 79));

                const double ref = reference_l2(a.ptr, b.ptr, dim);
                const double got = ComputeL2_AVX512::dist_i16(reinterpret_cast<uint8_t *>(a.ptr),
                                                              reinterpret_cast<uint8_t *>(b.ptr), dim);
                EXPECT_DOUBLE_EQ(ref, got) << "dim=" << dim << " misalign_a=" << misalign_a
                                           << " misalign_b=" << misalign_b;
            }
        }
    }
}

TEST_F(ComputeL2AVX512F, DistI16HandlesExtremes) {
    const size_t dim = 64;
    auto a = make_buffer<int16_t>(dim, 0);
    auto b = make_buffer<int16_t>(dim, 0);
    for (size_t i = 0; i < dim; ++i) {
        a.ptr[i] = (i % 2 == 0) ? INT16_MIN : INT16_MAX;
        b.ptr[i] = (i % 2 == 0) ? INT16_MAX : INT16_MIN;
    }

    const double ref = reference_l2(a.ptr, b.ptr, dim);
    const double got = ComputeL2_AVX512::dist_i16(reinterpret_cast<uint8_t *>(a.ptr),
                                                  reinterpret_cast<uint8_t *>(b.ptr), dim);
    EXPECT_DOUBLE_EQ(ref, got);
}

} // namespace

#endif

#if defined(SKETCH2_COMPUTE_AVX512VNNI_TESTS)

namespace {

class ComputeL2AVX512VNNI : public ::testing::Test {
protected:
    void SetUp() override {
        if (!ComputeUnit::is_supported(ComputeBackendKind::avx512_vnni)) {
            GTEST_SKIP() << "AVX-512 VNNI is not supported on this CPU";
        }
    }
};

TEST_F(ComputeL2AVX512VNNI, DistI16MatchesReferenceAlignedAndUnaligned) {
    const std::vector<size_t> dims = {1, 15, 16, 17, 31, 32, 33, 47, 48, 49, 96, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(2)}) {
            for (size_t misalign_b : {size_t(0), size_t(6)}) {
                auto a = make_buffer<int16_t>(dim, misalign_a);
                auto b = make_buffer<int16_t>(dim, misalign_b);
                fill_i16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 83));

                const double ref = reference_l2(a.ptr, b.ptr, dim);
                const double got = ComputeL2_AVX512_VNNI::dist_i16(reinterpret_cast<uint8_t *>(a.ptr),
                                                                   reinterpret_cast<uint8_t *>(b.ptr), dim);
                EXPECT_DOUBLE_EQ(ref, got) << "dim=" << dim << " misalign_a=" << misalign_a
                                           << " misalign_b=" << misalign_b;
            }
        }
    }
}

TEST_F(ComputeL2AVX512VNNI, DistI16HandlesExtremes) {
    const size_t dim = 64;
    auto a = make_buffer<int16_t>(dim, 0);
    auto b = make_buffer<int16_t>(dim, 0);
    for (size_t i = 0; i < dim; ++i) {
        a.ptr[i] = (i % 2 == 0) ? INT16_MIN : INT16_MAX;
        b.ptr[i] = (i % 2 == 0) ? INT16_MAX : INT16_MIN;
    }

    const double ref = reference_l2(a.ptr, b.ptr, dim);
    const double got = ComputeL2_AVX512_VNNI::dist_i16(reinterpret_cast<uint8_t *>(a.ptr),
                                                       reinterpret_cast<uint8_t *>(b.ptr), dim);
    EXPECT_DOUBLE_EQ(ref, got);
}

} // namespace

#endif
