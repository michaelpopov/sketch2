#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/compute/compute_l1_avx2.h"

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
