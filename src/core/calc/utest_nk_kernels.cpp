// Unit tests for NumKong distance kernels and resolver fallback behavior.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "core/calc/calc_engine.h"
#include "core/calc/hwy_kernels.h"
#include "core/calc/nk_kernels.h"
#include "core/compute/utest_compute_helpers.h"
#include "numkong/capabilities.h"

using namespace sketch2;
using namespace sketch2::test;

namespace {

template <typename T, typename FillFn, typename RefFn>
void expect_supported_numeric_matches_reference(DataType type, DistFunc func,
        FillFn fill_fn, RefFn expected_fn, double tolerance) {
    for (size_t dim : {1, 3, 7, 15, 17, 33, 100, 128, 257}) {
        auto ba = make_buffer<T>(dim, 0);
        auto bb = make_buffer<T>(dim, 0);
        fill_fn(ba.ptr, bb.ptr, dim, 42);
        const double expected = expected_fn(ba.ptr, bb.ptr, dim);
        const CalcKernels k = resolve_nk_kernels(func, type);
        ASSERT_NE(k.dist, nullptr);
        const double got = k.dist(reinterpret_cast<const uint8_t*>(ba.ptr),
                                  reinterpret_cast<const uint8_t*>(bb.ptr), dim);
        EXPECT_NEAR(expected, got, tolerance) << "dim=" << dim;
    }
}

} // namespace

TEST(NumKongKernelsTest, L2F32MatchesReference) {
    expect_supported_numeric_matches_reference<float>(
        DataType::f32, DistFunc::L2, fill_f32, reference_l2<float>, 1e-4);
}

TEST(NumKongKernelsTest, L2F16MatchesReference) {
    expect_supported_numeric_matches_reference<float16>(
        DataType::f16, DistFunc::L2, fill_f16, reference_l2<float16>, 1e-1);
}

TEST(NumKongKernelsTest, CosF32MatchesReference) {
    expect_supported_numeric_matches_reference<float>(
        DataType::f32, DistFunc::COS, fill_f32, reference_cosine_distance<float>, 1e-6);
}

TEST(NumKongKernelsTest, CosF16MatchesReference) {
    expect_supported_numeric_matches_reference<float16>(
        DataType::f16, DistFunc::COS, fill_f16, reference_cosine_distance<float16>, 1e-2);
}

TEST(NumKongKernelsTest, CosHelpersF32MatchReference) {
    for (size_t dim : {1, 3, 7, 15, 17, 33, 100, 128, 257}) {
        auto ba = make_buffer<float>(dim, 0);
        auto bb = make_buffer<float>(dim, 0);
        fill_f32(ba.ptr, bb.ptr, dim, 42);
        const CalcKernels k = resolve_nk_kernels(DistFunc::COS, DataType::f32);
        ASSERT_NE(k.dot, nullptr);
        ASSERT_NE(k.squared_norm, nullptr);
        ASSERT_NE(k.dist_with_query_norm, nullptr);

        const double expected_dot = reference_dot(ba.ptr, bb.ptr, dim);
        const double expected_norm = reference_squared_norm(ba.ptr, dim);
        const double expected_dist = reference_cosine_distance(ba.ptr, bb.ptr, dim);
        const double query_norm_sq = k.squared_norm(reinterpret_cast<const uint8_t*>(bb.ptr), dim);

        EXPECT_NEAR(expected_dot,
                    k.dot(reinterpret_cast<const uint8_t*>(ba.ptr),
                          reinterpret_cast<const uint8_t*>(bb.ptr), dim),
                    1e-4)
            << "dim=" << dim;
        EXPECT_NEAR(expected_norm,
                    k.squared_norm(reinterpret_cast<const uint8_t*>(ba.ptr), dim),
                    std::max(1e-4, std::abs(expected_norm) * 1e-6))
            << "dim=" << dim;
        EXPECT_NEAR(expected_dist,
                    k.dist_with_query_norm(reinterpret_cast<const uint8_t*>(ba.ptr),
                                           reinterpret_cast<const uint8_t*>(bb.ptr), dim, query_norm_sq),
                    1e-6)
            << "dim=" << dim;
    }
}

TEST(NumKongKernelsTest, L1FallsBackToHighway) {
    for (DataType type : {DataType::f32, DataType::f16, DataType::i16}) {
        const CalcKernels nk = resolve_nk_kernels(DistFunc::L1, type);
        const CalcKernels hwy = resolve_hwy_kernels(DistFunc::L1, type);
        EXPECT_EQ(hwy.dist, nk.dist) << "type=" << static_cast<int>(type);
        EXPECT_EQ(nullptr, nk.dot);
        EXPECT_EQ(nullptr, nk.squared_norm);
        EXPECT_EQ(nullptr, nk.dist_with_query_norm);
    }
}

TEST(NumKongKernelsTest, I16FallsBackToHighwayForAllMetrics) {
    for (DistFunc func : {DistFunc::L1, DistFunc::L2, DistFunc::COS}) {
        const CalcKernels nk = resolve_nk_kernels(func, DataType::i16);
        const CalcKernels hwy = resolve_hwy_kernels(func, DataType::i16);
        EXPECT_EQ(hwy.dist, nk.dist) << "func=" << static_cast<int>(func);
        EXPECT_EQ(hwy.dot, nk.dot) << "func=" << static_cast<int>(func);
        EXPECT_EQ(hwy.squared_norm, nk.squared_norm) << "func=" << static_cast<int>(func);
        EXPECT_EQ(hwy.dist_with_query_norm, nk.dist_with_query_norm) << "func=" << static_cast<int>(func);
    }
}

TEST(NumKongKernelsTest, ResolveCalcKernelsUsesNumKongBackendWhenRequested) {
    const CalcKernels resolved = resolve_calc_kernels(CalcEngine::numkong, DistFunc::L2, DataType::f32);
    const CalcKernels direct = resolve_nk_kernels(DistFunc::L2, DataType::f32);
    EXPECT_EQ(direct.dist, resolved.dist);
}

TEST(NumKongKernelsTest, DynamicDispatchMetadataIsConsistent) {
    EXPECT_TRUE(nk_calc_uses_dynamic_dispatch());

    const uint64_t compiled = nk_calc_compiled_capabilities();
    const uint64_t available = nk_calc_available_capabilities();

    EXPECT_NE(0u, compiled & static_cast<uint64_t>(nk_cap_serial_k));
    EXPECT_EQ(0u, available & ~compiled);
    EXPECT_STRNE("highway", nk_calc_backend_name(DistFunc::L2, DataType::f32));
}

#if defined(__x86_64__) || defined(_M_X64)
TEST(NumKongKernelsTest, BackendSelectionPrefersSkylakeOverHaswellWhenAvailable) {
    const uint64_t both = static_cast<uint64_t>(nk_cap_serial_k | nk_cap_haswell_k | nk_cap_skylake_k);
    const uint64_t haswell_only = static_cast<uint64_t>(nk_cap_serial_k | nk_cap_haswell_k);
    const uint64_t serial_only = static_cast<uint64_t>(nk_cap_serial_k);

    EXPECT_STREQ("skylake", nk_calc_backend_name_for_capabilities(DistFunc::L2, DataType::f32, both));
    EXPECT_STREQ("skylake", nk_calc_backend_name_for_capabilities(DistFunc::COS, DataType::f16, both));
    EXPECT_STREQ("haswell", nk_calc_backend_name_for_capabilities(DistFunc::L2, DataType::f32, haswell_only));
    EXPECT_STREQ("serial", nk_calc_backend_name_for_capabilities(DistFunc::COS, DataType::f16, serial_only));
}
#elif defined(__aarch64__) || defined(_M_ARM64)
TEST(NumKongKernelsTest, BackendSelectionPrefersSveOverNeonWhenAvailable) {
    const uint64_t sve = static_cast<uint64_t>(nk_cap_serial_k | nk_cap_neon_k | nk_cap_sve_k);
    const uint64_t sve_half = static_cast<uint64_t>(nk_cap_serial_k | nk_cap_neon_k | nk_cap_svehalf_k);
    const uint64_t neon = static_cast<uint64_t>(nk_cap_serial_k | nk_cap_neon_k);

    EXPECT_STREQ("sve", nk_calc_backend_name_for_capabilities(DistFunc::L2, DataType::f32, sve));
    EXPECT_STREQ("svehalf", nk_calc_backend_name_for_capabilities(DistFunc::COS, DataType::f16, sve_half));
    EXPECT_STREQ("neon", nk_calc_backend_name_for_capabilities(DistFunc::L2, DataType::f32, neon));
}
#endif
