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

TEST(ComputeRuntimeTest, ForcedScalarBackendUsesScalarResolvers) {
    ComputeUnitOverrideGuard guard(ComputeBackendKind::scalar);

    EXPECT_EQ(&ComputeL1::dist_f32, ComputeL1::resolve_dist(DataType::f32));
    EXPECT_EQ(&ComputeL2::dist_i16, ComputeL2::resolve_dist(DataType::i16));
    EXPECT_EQ(&ComputeCos::dot_f32, ComputeCos::resolve_dot(DataType::f32));
    EXPECT_EQ(&ComputeCos::dist_i16_with_query_norm, ComputeCos::resolve_dist_with_query_norm(DataType::i16));
}

#if defined(SKETCH_ENABLE_AVX2) && SKETCH_ENABLE_AVX2 && (defined(__x86_64__) || defined(__i386__))
TEST(ComputeRuntimeTest, ForcedAvx2BackendUsesAvx2Resolvers) {
    if (!ComputeUnit::is_supported(ComputeBackendKind::avx2)) {
        GTEST_SKIP() << "AVX2 is not supported on this CPU";
    }

    ComputeUnitOverrideGuard guard(ComputeBackendKind::avx2);

    EXPECT_EQ(&ComputeL1_AVX2::dist_f32, ComputeL1::resolve_dist(DataType::f32));
    EXPECT_EQ(&ComputeL2_AVX2::dist_i16, ComputeL2::resolve_dist(DataType::i16));
    EXPECT_EQ(&ComputeCos_AVX2::dot_f32, ComputeCos::resolve_dot(DataType::f32));
    EXPECT_EQ(&ComputeCos_AVX2::dist_i16_with_query_norm, ComputeCos::resolve_dist_with_query_norm(DataType::i16));
}
#endif

} // namespace
