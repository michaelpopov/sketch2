// Unit tests for startup singleton configuration helpers.

#include "singleton.h"
#include "log.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>

namespace sketch2 {

class SingletonTest : public ::testing::Test {
protected:
    void SetUp() override {
        char path[] = "/tmp/sketch2-singleton-XXXXXX";
        const int fd = mkstemp(path);
        ASSERT_NE(fd, -1);
        close(fd);
        path_ = path;
    }

    void TearDown() override {
        unsetenv("SKETCH2_CONFIG");
        std::remove(path_.c_str());
    }

    std::string path_;
};

TEST_F(SingletonTest, ApplyConfigFileSetsLogLevelFromIni) {
    const log::LogLevel original = log::get_log_level();

    std::ofstream out(path_);
    ASSERT_TRUE(out.is_open());
    out << "[log]\n";
    out << "level=warn\n";
    out.close();

    ASSERT_TRUE(Singleton::apply_config_file(path_));
    EXPECT_EQ(log::LogLevel::Warn, log::get_log_level());

    log::set_log_level(original);
}

TEST_F(SingletonTest, ApplyConfigFromEnvUsesSketch2ConfigPath) {
    const log::LogLevel original = log::get_log_level();

    std::ofstream out(path_);
    ASSERT_TRUE(out.is_open());
    out << "[log]\n";
    out << "level=debug\n";
    out.close();

    ASSERT_EQ(0, setenv("SKETCH2_CONFIG", path_.c_str(), 1));
    ASSERT_TRUE(Singleton::apply_config_from_env());
    EXPECT_EQ(log::LogLevel::Debug, log::get_log_level());

    log::set_log_level(original);
}

TEST_F(SingletonTest, ApplyConfigFromEnvReturnsFalseWhenConfigIsMissing) {
    unsetenv("SKETCH2_CONFIG");
    EXPECT_FALSE(Singleton::apply_config_from_env());
}

TEST_F(SingletonTest, ApplyConfigFileCreatesThreadPoolWhenSizeIsAboveOne) {
    std::ofstream out(path_);
    ASSERT_TRUE(out.is_open());
    out << "[thread_pool]\n";
    out << "size=3\n";
    out.close();

    ASSERT_TRUE(Singleton::apply_config_file(path_));
    ASSERT_NE(nullptr, get_singleton().thread_pool());
}

TEST_F(SingletonTest, ApplyConfigFileClearsThreadPoolWhenSizeIsMissingOrSmall) {
    {
        std::ofstream out(path_);
        ASSERT_TRUE(out.is_open());
        out << "[thread_pool]\n";
        out << "size=4\n";
        out.close();
    }
    ASSERT_TRUE(Singleton::apply_config_file(path_));
    ASSERT_NE(nullptr, get_singleton().thread_pool());

    {
        std::ofstream out(path_);
        ASSERT_TRUE(out.is_open());
        out << "[thread_pool]\n";
        out << "size=1\n";
        out.close();
    }
    EXPECT_FALSE(Singleton::apply_config_file(path_));
    EXPECT_EQ(nullptr, get_singleton().thread_pool());
}

} // namespace sketch2
