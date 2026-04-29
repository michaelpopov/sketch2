// Shared test helpers for building RoaringIds instances

#pragma once

#include "roaring_ids.h"

#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace sketch2 {

inline void init_roaring_ids_for_test(
        uint64_t base,
        const std::vector<uint64_t>& values,
        RoaringIds* ids) {
    RoaringIdsBuilder builder;
    ASSERT_EQ(0, builder.init(base).code());
    for (uint64_t id : values) {
        ASSERT_EQ(0, builder.add(id).code());
    }
    *ids = std::move(builder).build();
}

} // namespace sketch2
