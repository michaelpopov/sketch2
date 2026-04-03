// Test runner entry point for the utils module.

#include "utils/log.h"
#include <gtest/gtest.h>

using namespace sketch2::log;

int main(int argc, char** argv) {
    set_current_log_level(LogLevel::Critical);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

