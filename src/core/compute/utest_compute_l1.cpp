// Unit tests for L1-distance compute implementations.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/compute/compute_l1.h"
#include "core/compute/compute_l1_avx2.h"
#include "core/compute/compute_l1_neon.h"

using namespace sketch2;

namespace {

TEST(ComputeL1Test, DistF32ComputesDistance) {
    const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> b = {1.5f, 0.0f, 1.0f, 10.0f};
    ComputeL1 l1;
    const double got = l1.dist(reinterpret_cast<const uint8_t*>(a.data()),
                               reinterpret_cast<const uint8_t*>(b.data()),
                               DataType::f32, a.size());
    EXPECT_DOUBLE_EQ(10.5, got);
}

TEST(ComputeL1Test, DistF16ComputesDistance) {
    if (!supports_f16()) {
        GTEST_SKIP() << "f16 is not supported on this build";
    }

    const std::vector<float16> a = {float16(1.0f), float16(2.0f), float16(3.0f), float16(4.0f)};
    const std::vector<float16> b = {float16(1.5f), float16(0.0f), float16(1.0f), float16(10.0f)};
    ComputeL1 l1;
    const double got = l1.dist(reinterpret_cast<const uint8_t*>(a.data()),
                               reinterpret_cast<const uint8_t*>(b.data()),
                               DataType::f16, a.size());
    EXPECT_NEAR(10.5, got, 1e-3);
}

TEST(ComputeL1Test, ResolveDistReturnsFunctionForAllTypes) {
    EXPECT_NE(nullptr, ComputeL1::resolve_dist(DataType::f32));
    if (supports_f16()) {
        EXPECT_NE(nullptr, ComputeL1::resolve_dist(DataType::f16));
    }
    EXPECT_NE(nullptr, ComputeL1::resolve_dist(DataType::i16));
}

#if !defined(__AVX2__) && !defined(__aarch64__)
TEST(ComputeL1Scalar, DistI16UsesScalarFallback) {
    const std::vector<int16_t> a = {10, -2, 7, -8};
    const std::vector<int16_t> b = {4, -5, 10, -8};
    ComputeL1 l1;
    const double got = l1.dist(reinterpret_cast<const uint8_t*>(a.data()),
                               reinterpret_cast<const uint8_t*>(b.data()),
                               DataType::i16, a.size());
    EXPECT_DOUBLE_EQ(12.0, got);
}
#else
TEST(ComputeL1Scalar, NotBuiltForThisTarget) {
    GTEST_SKIP() << "scalar fallback is not the active implementation on this target";
}
#endif

#if defined(__aarch64__)
TEST(ComputeL1Neon, DistF32MatchesReference) {
    const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, -5.0f};
    const std::vector<float> b = {0.0f, 2.5f, 1.0f, 0.0f, -1.0f};
    const double got = ComputeL1_Neon::dist_f32(reinterpret_cast<const uint8_t*>(a.data()),
                                                reinterpret_cast<const uint8_t*>(b.data()),
                                                a.size());
    EXPECT_DOUBLE_EQ(11.5, got);
}

TEST(ComputeL1Neon, DistI16MatchesReference) {
    const std::vector<int16_t> a = {10, -2, 7, -8, 20};
    const std::vector<int16_t> b = {4, -5, 10, -8, 18};
    const double got = ComputeL1_Neon::dist_i16(reinterpret_cast<const uint8_t*>(a.data()),
                                                reinterpret_cast<const uint8_t*>(b.data()),
                                                a.size());
    EXPECT_DOUBLE_EQ(14.0, got);
}
#else
TEST(ComputeL1Neon, NotBuiltForThisTarget) {
    GTEST_SKIP() << "NEON is not enabled for this target";
}
#endif

template <typename T>
struct TestBuffer {
    std::vector<uint8_t> storage;
    T *ptr = nullptr;
};

template <typename T>
TestBuffer<T> make_buffer(size_t dim, size_t misalign_bytes) {
    TestBuffer<T> out;
    out.storage.resize(dim * sizeof(T) + 64 + misalign_bytes);
    uintptr_t p = reinterpret_cast<uintptr_t>(out.storage.data());
    p = (p + 31u) & ~uintptr_t(31u);
    p += misalign_bytes;
    out.ptr = reinterpret_cast<T *>(p);
    return out;
}

#if defined(__AVX2__)

double reference_l1_f32(const float *a, const float *b, size_t dim) {
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        sum += std::abs(a[i] - b[i]);
    }
    return sum;
}

double reference_l1_i16(const int16_t *a, const int16_t *b, size_t dim) {
    int64_t sum = 0;
    for (size_t i = 0; i < dim; ++i) {
        const int diff = static_cast<int>(a[i]) - static_cast<int>(b[i]);
        sum += std::abs(diff);
    }
    return static_cast<double>(sum);
}

void fill_f32(float *a, float *b, size_t dim, uint32_t seed) {
    for (size_t i = 0; i < dim; ++i) {
        const int32_t ai = static_cast<int32_t>((i * 17 + seed * 13) % 401) - 200;
        const int32_t bi = static_cast<int32_t>((i * 29 + seed * 7) % 401) - 200;
        a[i] = static_cast<float>(ai) * 0.125f + static_cast<float>(i % 5) * 0.03125f;
        b[i] = static_cast<float>(bi) * 0.125f - static_cast<float>(i % 3) * 0.0625f;
    }
}

void fill_i16(int16_t *a, int16_t *b, size_t dim, uint32_t seed) {
    for (size_t i = 0; i < dim; ++i) {
        const int32_t ai = static_cast<int32_t>((i * 977 + seed * 131) % 65536) - 32768;
        const int32_t bi = static_cast<int32_t>((i * 733 + seed * 191) % 65536) - 32768;
        a[i] = static_cast<int16_t>(ai);
        b[i] = static_cast<int16_t>(bi);
    }
}

#if defined(__FLT16_MANT_DIG__)
double reference_l1_f16(const float16 *a, const float16 *b, size_t dim) {
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        sum += std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    }
    return sum;
}

void fill_f16(float16 *a, float16 *b, size_t dim, uint32_t seed) {
    for (size_t i = 0; i < dim; ++i) {
        const int32_t ai = static_cast<int32_t>((i * 17 + seed * 13) % 401) - 200;
        const int32_t bi = static_cast<int32_t>((i * 29 + seed * 7) % 401) - 200;
        a[i] = static_cast<float16>(static_cast<float>(ai) * 0.125f + static_cast<float>(i % 5) * 0.03125f);
        b[i] = static_cast<float16>(static_cast<float>(bi) * 0.125f - static_cast<float>(i % 3) * 0.0625f);
    }
}
#endif

TEST(ComputeL1AVX2, DistF32ZeroDimIsZero) {
    auto a = make_buffer<float>(1, 0);
    auto b = make_buffer<float>(1, 0);
    const double got = ComputeL1_AVX2::dist_f32(reinterpret_cast<uint8_t *>(a.ptr),
                                                reinterpret_cast<uint8_t *>(b.ptr), 0);
    EXPECT_DOUBLE_EQ(0.0, got);
}

TEST(ComputeL1AVX2, DistF32MatchesReferenceAlignedAndUnaligned) {
    const std::vector<size_t> dims = {1, 7, 8, 9, 15, 16, 31, 32, 33, 63, 64, 65, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(4)}) {
            for (size_t misalign_b : {size_t(0), size_t(12)}) {
                auto a = make_buffer<float>(dim, misalign_a);
                auto b = make_buffer<float>(dim, misalign_b);
                fill_f32(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 3));

                const double ref = reference_l1_f32(a.ptr, b.ptr, dim);
                const double got = ComputeL1_AVX2::dist_f32(reinterpret_cast<uint8_t *>(a.ptr),
                                                            reinterpret_cast<uint8_t *>(b.ptr), dim);
                const double tol = std::max(1e-5, ref * 1e-5);
                EXPECT_NEAR(ref, got, tol) << "dim=" << dim << " misalign_a=" << misalign_a
                                           << " misalign_b=" << misalign_b;
            }
        }
    }
}

TEST(ComputeL1AVX2, DistF32LargeMultipleOf8MatchesReference) {
    const size_t dim = 1024;
    auto a = make_buffer<float>(dim, 0);
    auto b = make_buffer<float>(dim, 0);
    fill_f32(a.ptr, b.ptr, dim, 1234);

    const double ref = reference_l1_f32(a.ptr, b.ptr, dim);
    const double got = ComputeL1_AVX2::dist_f32(reinterpret_cast<uint8_t *>(a.ptr),
                                                reinterpret_cast<uint8_t *>(b.ptr), dim);
    EXPECT_NEAR(ref, got, std::max(1e-5, ref * 1e-5));
}

#if defined(__FLT16_MANT_DIG__)
TEST(ComputeL1AVX2, DistF16ZeroDimIsZero) {
    auto a = make_buffer<float16>(1, 0);
    auto b = make_buffer<float16>(1, 0);
    const double got = ComputeL1_AVX2::dist_f16(reinterpret_cast<uint8_t *>(a.ptr),
                                                reinterpret_cast<uint8_t *>(b.ptr), 0);
    EXPECT_DOUBLE_EQ(0.0, got);
}

TEST(ComputeL1AVX2, DistF16MatchesReferenceAlignedAndUnaligned) {
    const std::vector<size_t> dims = {1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(2)}) {
            for (size_t misalign_b : {size_t(0), size_t(6)}) {
                auto a = make_buffer<float16>(dim, misalign_a);
                auto b = make_buffer<float16>(dim, misalign_b);
                fill_f16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 11));

                const double ref = reference_l1_f16(a.ptr, b.ptr, dim);
                const double got = ComputeL1_AVX2::dist_f16(reinterpret_cast<uint8_t *>(a.ptr),
                                                            reinterpret_cast<uint8_t *>(b.ptr), dim);
                EXPECT_NEAR(ref, got, std::max(1e-2, ref * 2e-3)) << "dim=" << dim
                                                                   << " misalign_a=" << misalign_a
                                                                   << " misalign_b=" << misalign_b;
            }
        }
    }
}

TEST(ComputeL1AVX2, ResolveDistUsesAVX2F16Path) {
    EXPECT_EQ(&ComputeL1_AVX2::dist_f16, ComputeL1::resolve_dist(DataType::f16));
}
#endif

TEST(ComputeL1AVX2, DistI16ZeroDimIsZero) {
    auto a = make_buffer<int16_t>(1, 0);
    auto b = make_buffer<int16_t>(1, 0);
    const double got = ComputeL1_AVX2::dist_i16(reinterpret_cast<uint8_t *>(a.ptr),
                                                reinterpret_cast<uint8_t *>(b.ptr), 0);
    EXPECT_DOUBLE_EQ(0.0, got);
}

TEST(ComputeL1AVX2, DistI16MatchesReferenceAlignedAndUnaligned) {
    const std::vector<size_t> dims = {1, 15, 16, 17, 31, 32, 33, 47, 48, 49, 96, 127};
    for (size_t dim : dims) {
        for (size_t misalign_a : {size_t(0), size_t(2)}) {
            for (size_t misalign_b : {size_t(0), size_t(6)}) {
                auto a = make_buffer<int16_t>(dim, misalign_a);
                auto b = make_buffer<int16_t>(dim, misalign_b);
                fill_i16(a.ptr, b.ptr, dim, static_cast<uint32_t>(dim + misalign_a + misalign_b + 5));

                const double ref = reference_l1_i16(a.ptr, b.ptr, dim);
                const double got = ComputeL1_AVX2::dist_i16(reinterpret_cast<uint8_t *>(a.ptr),
                                                            reinterpret_cast<uint8_t *>(b.ptr), dim);
                EXPECT_DOUBLE_EQ(ref, got) << "dim=" << dim << " misalign_a=" << misalign_a
                                           << " misalign_b=" << misalign_b;
            }
        }
    }
}

TEST(ComputeL1AVX2, DistI16HandlesExtremes) {
    const size_t dim = 64;
    auto a = make_buffer<int16_t>(dim, 0);
    auto b = make_buffer<int16_t>(dim, 0);
    for (size_t i = 0; i < dim; ++i) {
        a.ptr[i] = (i % 2 == 0) ? INT16_MIN : INT16_MAX;
        b.ptr[i] = (i % 2 == 0) ? INT16_MAX : INT16_MIN;
    }

    const double ref = reference_l1_i16(a.ptr, b.ptr, dim);
    const double got = ComputeL1_AVX2::dist_i16(reinterpret_cast<uint8_t *>(a.ptr),
                                                reinterpret_cast<uint8_t *>(b.ptr), dim);
    EXPECT_DOUBLE_EQ(ref, got);
}

#else

TEST(ComputeL1AVX2, NotBuiltForThisTarget) {
    GTEST_SKIP() << "AVX2 is not enabled for this target";
}

#endif

} // namespace
