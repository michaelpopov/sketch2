// Unit tests for runtime compute-unit selection.

#include <gtest/gtest.h>

#include "core/compute/compute_cos.h"
#include "core/compute/compute_l1.h"
#include "core/compute/compute_l2.h"
#include "core/utils/singleton.h"

using namespace sketch2;

namespace {

class ComputeUnitOverrideGuard {
public:
    explicit ComputeUnitOverrideGuard(ComputeBackendKind kind)
        : original_(get_singleton().compute_unit().kind()) {
        const bool changed = Singleton::force_compute_unit_for_testing(kind);
        EXPECT_TRUE(changed);
    }

    ~ComputeUnitOverrideGuard() {
        const bool restored = Singleton::force_compute_unit_for_testing(original_);
        EXPECT_TRUE(restored);
    }

private:
    ComputeBackendKind original_;
};

template <typename Backend>
void expect_l1_resolvers() {
    EXPECT_EQ(&Backend::dist_f32, ComputeL1::resolve_dist(DataType::f32));
    EXPECT_EQ(&Backend::dist_f16, ComputeL1::resolve_dist(DataType::f16));
    EXPECT_EQ(&Backend::dist_i16, ComputeL1::resolve_dist(DataType::i16));
}

template <typename Backend>
void expect_l2_resolvers() {
    EXPECT_EQ(&Backend::dist_f32, ComputeL2::resolve_dist(DataType::f32));
    EXPECT_EQ(&Backend::dist_f16, ComputeL2::resolve_dist(DataType::f16));
    EXPECT_EQ(&Backend::dist_i16, ComputeL2::resolve_dist(DataType::i16));
}

template <typename Backend>
void expect_cos_resolvers() {
    EXPECT_EQ(&Backend::dist_f32, ComputeCos::resolve_dist(DataType::f32));
    EXPECT_EQ(&Backend::dist_f16, ComputeCos::resolve_dist(DataType::f16));
    EXPECT_EQ(&Backend::dist_i16, ComputeCos::resolve_dist(DataType::i16));

    EXPECT_EQ(&Backend::dist_f32_with_query_norm, ComputeCos::resolve_dist_with_query_norm(DataType::f32));
    EXPECT_EQ(&Backend::dist_f16_with_query_norm, ComputeCos::resolve_dist_with_query_norm(DataType::f16));
    EXPECT_EQ(&Backend::dist_i16_with_query_norm, ComputeCos::resolve_dist_with_query_norm(DataType::i16));

    EXPECT_EQ(&Backend::squared_norm_f32, ComputeCos::resolve_squared_norm(DataType::f32));
    EXPECT_EQ(&Backend::squared_norm_f16, ComputeCos::resolve_squared_norm(DataType::f16));
    EXPECT_EQ(&Backend::squared_norm_i16, ComputeCos::resolve_squared_norm(DataType::i16));

    EXPECT_EQ(&Backend::dot_f32, ComputeCos::resolve_dot(DataType::f32));
    EXPECT_EQ(&Backend::dot_f16, ComputeCos::resolve_dot(DataType::f16));
    EXPECT_EQ(&Backend::dot_i16, ComputeCos::resolve_dot(DataType::i16));
}

TEST(ComputeRuntimeTest, ForcedScalarBackendUsesScalarResolvers) {
    ComputeUnitOverrideGuard guard(ComputeBackendKind::scalar);

    expect_l1_resolvers<ComputeL1>();
    expect_l2_resolvers<ComputeL2>();
    expect_cos_resolvers<ComputeCos>();
}

#if defined(SKETCH_ENABLE_AVX2) && SKETCH_ENABLE_AVX2 && (defined(__x86_64__) || defined(__i386__))
TEST(ComputeRuntimeTest, ForcedAvx2BackendUsesAvx2Resolvers) {
    if (!ComputeUnit::is_supported(ComputeBackendKind::avx2)) {
        GTEST_SKIP() << "AVX2 is not supported on this CPU";
    }

    ComputeUnitOverrideGuard guard(ComputeBackendKind::avx2);

    expect_l1_resolvers<ComputeL1_AVX2>();
    expect_l2_resolvers<ComputeL2_AVX2>();
    expect_cos_resolvers<ComputeCos_AVX2>();
}
#endif

#if defined(SKETCH_ENABLE_AVX512F) && SKETCH_ENABLE_AVX512F && (defined(__x86_64__) || defined(__i386__))
TEST(ComputeRuntimeTest, ForcedAvx512BackendUsesAvx512Resolvers) {
    if (!ComputeUnit::is_supported(ComputeBackendKind::avx512f)) {
        GTEST_SKIP() << "AVX-512F is not supported on this CPU";
    }

    ComputeUnitOverrideGuard guard(ComputeBackendKind::avx512f);

    expect_l1_resolvers<ComputeL1_AVX512>();
    expect_l2_resolvers<ComputeL2_AVX512>();
    expect_cos_resolvers<ComputeCos_AVX512>();
}
#endif

#if defined(SKETCH_ENABLE_AVX512VNNI) && SKETCH_ENABLE_AVX512VNNI && (defined(__x86_64__) || defined(__i386__))
TEST(ComputeRuntimeTest, ForcedAvx512VnniBackendUsesAvx512VnniResolvers) {
    if (!ComputeUnit::is_supported(ComputeBackendKind::avx512_vnni)) {
        GTEST_SKIP() << "AVX-512 VNNI is not supported on this CPU";
    }

    ComputeUnitOverrideGuard guard(ComputeBackendKind::avx512_vnni);

    expect_l1_resolvers<ComputeL1_AVX512_VNNI>();
    expect_l2_resolvers<ComputeL2_AVX512_VNNI>();
    expect_cos_resolvers<ComputeCos_AVX512_VNNI>();
}
#endif

} // namespace
