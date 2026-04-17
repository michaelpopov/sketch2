// Provides NEON helper utilities shared by the vectorized compute implementations.

#pragma once
#include "core/utils/arch_detection.h"

#if SKETCH_HAS_NEON
#include <arm_neon.h>

namespace sketch2 {

inline int64_t hsum_s64x2(int64x2_t v) {
    return vgetq_lane_s64(v, 0) + vgetq_lane_s64(v, 1);
}

inline void accumulate_mul_i32_as_i64(
        int32x4_t a, int32x4_t b, int64x2_t* acc0, int64x2_t* acc1) {
    *acc0 = vaddq_s64(*acc0, vmull_s32(vget_low_s32(a), vget_low_s32(b)));
    *acc1 = vaddq_s64(*acc1, vmull_s32(vget_high_s32(a), vget_high_s32(b)));
}

} // namespace sketch2

#endif
