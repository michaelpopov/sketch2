// Provides NEON helper utilities shared by the vectorized compute implementations.

#pragma once
#include "core/utils/arch_detection.h"

#if SKETCH_HAS_NEON
#include <arm_neon.h>

namespace sketch2 {

inline int64_t hsum_s64x2(int64x2_t v) {
    return vgetq_lane_s64(v, 0) + vgetq_lane_s64(v, 1);
}

} // namespace sketch2

#endif
