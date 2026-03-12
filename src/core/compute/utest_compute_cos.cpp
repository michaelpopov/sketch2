#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/compute/compute_cos.h"
#include "core/compute/compute_cos_avx2.h"
#include "core/compute/compute_cos_neon.h"

using namespace sketch2;

namespace {

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

template <typename T>
double reference_cosine_distance(const T *a, const T *b, size_t dim) {
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double ai = static_cast<double>(a[i]);
        const double bi = static_cast<double>(b[i]);
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    if (norm_a == 0.0 && norm_b == 0.0) {
        return 0.0;
    }
    if (norm_a == 0.0 || norm_b == 0.0) {
        return 1.0;
    }

    const double cosine = std::clamp(dot / std::sqrt(norm_a * norm_b), -1.0, 1.0);
    return 1.0 - cosine;
}

void fill_f32(float *a, float *b, size_t dim, uint32_t seed) {
    for (size_t i = 0; i < dim; ++i) {
        const int32_t ai = static_cast<int32_t>((i * 17 + seed * 13) % 401) - 200;
        const int32_t bi = static_cast<int32_t>((i * 29 + seed * 7) % 401) - 200;
        a[i] = static_cast<float>(ai) * 0.125f + static_cast<float>((i + seed) % 5) * 0.03125f;
        b[i] = static_cast<float>(bi) * 0.125f - static_cast<float>((i + seed) % 3) * 0.0625f;
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
void fill_f16(float16 *a, float16 *b, size_t dim, uint32_t seed) {
    for (size_t i = 0; i < dim; ++i) {
        const int32_t ai = static_cast<int32_t>((i * 17 + seed * 13) % 401) - 200;
        const int32_t bi = static_cast<int32_t>((i * 29 + seed * 7) % 401) - 200;
        a[i] = static_cast<float16>(static_cast<float>(ai) * 0.125f + static_cast<float>((i + seed) % 5) * 0.03125f);
        b[i] = static_cast<float16>(static_cast<float>(bi) * 0.125f - static_cast<float>((i + seed) % 3) * 0.0625f);
    }
}
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

#if !defined(__AVX2__) && !defined(__aarch64__)
TEST(ComputeCosScalar, DistI16UsesScalarFallback) {
    const std::vector<int16_t> a = {1, 2, -1, 0};
    const std::vector<int16_t> b = {2, 1, -2, 0};
    ComputeCos cos;
    const double got = cos.dist(reinterpret_cast<const uint8_t*>(a.data()),
                                reinterpret_cast<const uint8_t*>(b.data()),
                                DataType::i16, a.size());
    EXPECT_NEAR(reference_cosine_distance(a.data(), b.data(), a.size()), got, 1e-12);
}
#else
TEST(ComputeCosScalar, NotBuiltForThisTarget) {
    GTEST_SKIP() << "scalar fallback is not the active implementation on this target";
}
#endif

#if defined(__aarch64__)
TEST(ComputeCosNeon, DistF32MatchesReference) {
    const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, -5.0f};
    const std::vector<float> b = {0.0f, 2.5f, 1.0f, 0.0f, -1.0f};
    const double got = ComputeCos_Neon::dist_f32(reinterpret_cast<const uint8_t*>(a.data()),
                                                 reinterpret_cast<const uint8_t*>(b.data()),
                                                 a.size());
    EXPECT_NEAR(reference_cosine_distance(a.data(), b.data(), a.size()), got, 1e-6);
}

TEST(ComputeCosNeon, DistI16MatchesReference) {
    const std::vector<int16_t> a = {10, -2, 7, -8, 20};
    const std::vector<int16_t> b = {4, -5, 10, -8, 18};
    const double got = ComputeCos_Neon::dist_i16(reinterpret_cast<const uint8_t*>(a.data()),
                                                 reinterpret_cast<const uint8_t*>(b.data()),
                                                 a.size());
    EXPECT_NEAR(reference_cosine_distance(a.data(), b.data(), a.size()), got, 1e-6);
}
#else
TEST(ComputeCosNeon, NotBuiltForThisTarget) {
    GTEST_SKIP() << "NEON is not enabled for this target";
}
#endif

#if defined(__AVX2__)
TEST(ComputeCosAVX2, DistF32ZeroDimIsZero) {
    auto a = make_buffer<float>(1, 0);
    auto b = make_buffer<float>(1, 0);
    const double got = ComputeCos_AVX2::dist_f32(reinterpret_cast<uint8_t *>(a.ptr),
                                                 reinterpret_cast<uint8_t *>(b.ptr), 0);
    EXPECT_DOUBLE_EQ(0.0, got);
}

TEST(ComputeCosAVX2, DistF32MatchesReferenceAlignedAndUnaligned) {
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
TEST(ComputeCosAVX2, DistF16ZeroDimIsZero) {
    auto a = make_buffer<float16>(1, 0);
    auto b = make_buffer<float16>(1, 0);
    const double got = ComputeCos_AVX2::dist_f16(reinterpret_cast<uint8_t *>(a.ptr),
                                                 reinterpret_cast<uint8_t *>(b.ptr), 0);
    EXPECT_DOUBLE_EQ(0.0, got);
}

TEST(ComputeCosAVX2, DistF16MatchesReferenceAlignedAndUnaligned) {
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

TEST(ComputeCosAVX2, ResolveDistUsesAVX2F16Path) {
    EXPECT_EQ(&ComputeCos_AVX2::dist_f16, ComputeCos::resolve_dist(DataType::f16));
}
#endif

TEST(ComputeCosAVX2, DistI16ZeroDimIsZero) {
    auto a = make_buffer<int16_t>(1, 0);
    auto b = make_buffer<int16_t>(1, 0);
    const double got = ComputeCos_AVX2::dist_i16(reinterpret_cast<uint8_t *>(a.ptr),
                                                 reinterpret_cast<uint8_t *>(b.ptr), 0);
    EXPECT_DOUBLE_EQ(0.0, got);
}

TEST(ComputeCosAVX2, DistI16MatchesReferenceAlignedAndUnaligned) {
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
#endif

} // namespace
