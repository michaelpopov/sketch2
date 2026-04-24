// Shared helpers for storage tests that build Roaring id trailers.

#pragma once

#include "core/utils/roaring_ids.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

namespace sketch2 {

inline void init_roaring_ids_for_test(
        uint64_t base,
        const std::vector<uint64_t>& values,
        RoaringIds* ids) {
    ASSERT_EQ(0, ids->init_writable(base).code());
    for (uint64_t id : values) {
        ASSERT_EQ(0, ids->add(id).code());
    }
    ids->compact();
}

} // namespace sketch2
