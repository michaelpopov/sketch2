// Unit tests for shared utility types.

#include "utils/shared_types.h"

#include <gtest/gtest.h>

namespace sketch2 {

TEST(shared_types, ret_copy_preserves_content_flag) {
    const Ret original(0, "payload", true);

    const Ret copied(original);

    EXPECT_EQ(0, copied.code());
    EXPECT_EQ("payload", copied.message());
    EXPECT_TRUE(copied.is_content());
}

} // namespace sketch2
