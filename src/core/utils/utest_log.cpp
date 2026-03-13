// Unit tests for the logging system.

#include "log.h"

#include <gtest/gtest.h>

#include <functional>
#include <fstream>
#include <string>
#include <unistd.h>

namespace sketch2::log {

namespace {

std::string capture_stderr_output(const std::function<void()>& fn) {
    int pipe_fds[2];
    if (pipe(pipe_fds) != 0) {
        return "";
    }

    const int saved_stderr = dup(STDERR_FILENO);
    if (saved_stderr == -1) {
        close(pipe_fds[0]);
        close(pipe_fds[1]);
        return "";
    }
    if (dup2(pipe_fds[1], STDERR_FILENO) == -1) {
        close(saved_stderr);
        close(pipe_fds[0]);
        close(pipe_fds[1]);
        return "";
    }
    close(pipe_fds[1]);

    fn();

    if (dup2(saved_stderr, STDERR_FILENO) == -1) {
        close(saved_stderr);
        close(pipe_fds[0]);
        return "";
    }
    close(saved_stderr);

    std::string output;
    char buffer[512];
    ssize_t bytes_read = 0;
    while ((bytes_read = read(pipe_fds[0], buffer, sizeof(buffer))) > 0) {
        output.append(buffer, static_cast<size_t>(bytes_read));
    }
    close(pipe_fds[0]);
    return output;
}

}  // namespace

TEST(LogTest, LogLevelEnum) {
    EXPECT_EQ(static_cast<int>(LogLevel::Critical), 0);
    EXPECT_EQ(static_cast<int>(LogLevel::Debug), 5);
}

TEST(LogTest, LogToString) {
    EXPECT_STREQ(FILELog::to_string(LogLevel::Critical), "CRITICAL ");
    EXPECT_STREQ(FILELog::to_string(LogLevel::Info), "INFO     ");
    EXPECT_STREQ(FILELog::to_string(LogLevel::Debug), "DEBUG    ");
}

TEST(LogTest, LogFromString) {
    EXPECT_EQ(FILELog::from_string("DEBUG"), LogLevel::Debug);
    EXPECT_EQ(FILELog::from_string("info"), LogLevel::Info);
    EXPECT_EQ(FILELog::from_string("CRITICAL"), LogLevel::Critical);
    EXPECT_EQ(FILELog::from_string("UNKNOWN"), LogLevel::Info);
}

TEST(LogTest, LogLevelStorage) {
    LogLevel oldLevel = get_log_level();
    set_log_level(LogLevel::Debug);
    EXPECT_EQ(get_log_level(), LogLevel::Debug);
    set_log_level(oldLevel);
}

TEST(LogTest, TempLogLevel) {
    LogLevel original = get_log_level();
    {
        TempLogLevel temp(LogLevel::Trace);
        EXPECT_EQ(get_log_level(), LogLevel::Trace);
    }
    EXPECT_EQ(get_log_level(), original);
}

TEST(LogTest, Macros) {
    // This just verifies that the macros compile and can be called.
    // We don't easily capture stderr here, but we can check if it doesn't crash.
    LOG_INFO << "Test info message";
    LOG_DEBUG << "Test debug message";
    LOG_ERROR << "Test error message";
}

TEST(LogTest, MacrosAreStatementSafe) {
    bool elseTaken = false;

    if (false)
        LOG_INFO << "This branch should not run";
    else
        elseTaken = true;

    EXPECT_TRUE(elseTaken);
}

TEST(LogTest, EnabledMacrosAreStatementSafe) {
    bool elseTaken = false;

    if (true)
        LOG_INFO << "This branch should run";
    else
        elseTaken = true;

    EXPECT_FALSE(elseTaken);
}

TEST(LogTest, DisabledLogPathCompilesAndSkipsWork) {
    LogLevel original = get_log_level();
    set_log_level(LogLevel::Error);

    bool evaluated = false;
    auto mark_evaluated = [&]() -> const char* {
        evaluated = true;
        return "This disabled branch should not run";
    };

    LOG_DEBUG << mark_evaluated();

    EXPECT_FALSE(evaluated);
    EXPECT_EQ(get_log_level(), LogLevel::Error);
    set_log_level(original);
}

TEST(LogTest, LongMessagesAreTruncatedSafely) {
    LogLevel original = get_log_level();
    set_log_level(LogLevel::Info);

    const std::string payload(FixedBufferStreamBuf::kCapacity * 2, 'x');
    const std::string output = capture_stderr_output([&payload]() {
        LOG_INFO << payload;
    });

    EXPECT_FALSE(output.empty());
    EXPECT_EQ(output.back(), '\n');
    EXPECT_LE(output.size(), FixedBufferStreamBuf::kCapacity);
    EXPECT_NE(output.find("[truncated]"), std::string::npos);

    set_log_level(original);
}

TEST(LogTest, CanWriteLogsToDedicatedFile) {
    LogLevel original_level = get_log_level();
    set_log_level(LogLevel::Info);

    char path[] = "/tmp/sketch2-log-XXXXXX";
    const int temp_fd = mkstemp(path);
    ASSERT_NE(temp_fd, -1);
    close(temp_fd);

    ASSERT_TRUE(configure_log_file(path));
    LOG_INFO << "file sink message";
    reset_log_output();

    std::ifstream input(path);
    ASSERT_TRUE(input.is_open());
    const std::string output((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());

    EXPECT_NE(output.find("file sink message"), std::string::npos);

    unlink(path);
    set_log_level(original_level);
}

} // namespace sketch2::log
