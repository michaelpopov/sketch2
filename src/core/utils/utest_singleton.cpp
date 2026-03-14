// Unit tests for startup singleton configuration precedence and sealing.

#include "singleton.h"
#include "log.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <unistd.h>

namespace sketch2 {

class SingletonTest : public ::testing::Test {
protected:
    void SetUp() override {
        char path1[] = "/tmp/sketch2-singleton-a-XXXXXX";
        const int fd1 = mkstemp(path1);
        ASSERT_NE(fd1, -1);
        close(fd1);
        path1_ = path1;

        char path2[] = "/tmp/sketch2-singleton-b-XXXXXX";
        const int fd2 = mkstemp(path2);
        ASSERT_NE(fd2, -1);
        close(fd2);
        path2_ = path2;
    }

    void TearDown() override {
        unsetenv("SKETCH2_CONFIG");
        unsetenv("SKETCH2_LOG_LEVEL");
        unsetenv("SKETCH2_THREAD_POOL_SIZE");
        unsetenv("SKETCH2_LOG_FILE");
        std::remove(path1_.c_str());
        std::remove(path2_.c_str());
    }

    std::string path1_;
    std::string path2_;
};

TEST_F(SingletonTest, EnvOverridesApplyWithOrWithoutConfigBeforeSingletonSeals) {
    const log::LogLevel original = log::get_log_level();

    unsetenv("SKETCH2_CONFIG");
    unsetenv("SKETCH2_LOG_LEVEL");
    unsetenv("SKETCH2_THREAD_POOL_SIZE");
    EXPECT_FALSE(Singleton::apply_config_from_env());

    ASSERT_EQ(0, setenv("SKETCH2_CONFIG", "/tmp/sketch2-missing-config.ini", 1));
    ASSERT_EQ(0, setenv("SKETCH2_LOG_LEVEL", "debug", 1));
    ASSERT_EQ(0, setenv("SKETCH2_THREAD_POOL_SIZE", "3", 1));
    ASSERT_EQ(0, setenv("SKETCH2_LOG_FILE", path2_.c_str(), 1));
    ASSERT_TRUE(Singleton::apply_config_from_env());
    EXPECT_EQ(log::LogLevel::Debug, log::get_log_level());
    ASSERT_NE(nullptr, get_singleton().thread_pool());
    LOG_INFO << "singleton log file test";

    std::ifstream log_input(path2_);
    ASSERT_TRUE(log_input.is_open());
    const std::string log_output((std::istreambuf_iterator<char>(log_input)), std::istreambuf_iterator<char>());
    EXPECT_NE(log_output.find("singleton log file test"), std::string::npos);

    {
        std::ofstream out(path1_);
        ASSERT_TRUE(out.is_open());
        out << "[log]\n";
        out << "level=warn\n";
        out << "[thread_pool]\n";
        out << "size=3\n";
        out.close();
    }

    ASSERT_EQ(0, setenv("SKETCH2_CONFIG", path1_.c_str(), 1));
    EXPECT_FALSE(Singleton::apply_config_from_env());
    EXPECT_EQ(log::LogLevel::Debug, log::get_log_level());
    EXPECT_NE(nullptr, get_singleton().thread_pool());

    log::set_log_level(original);
}

} // namespace sketch2
