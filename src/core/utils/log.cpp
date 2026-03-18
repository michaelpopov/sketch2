// Defines shared logging state and sink operations for the process-wide utils library.

#include "log.h"

#include <atomic>
#include <cerrno>
#include <fcntl.h>
#include <mutex>
#include <strings.h>
#include <unistd.h>

namespace sketch2 {
namespace log {

namespace {

std::atomic<LogLevel>& level_storage() {
    static std::atomic<LogLevel> level{LogLevel::Error};
    return level;
}

std::atomic<int>& output_fd_storage() {
    static std::atomic<int> fd{STDERR_FILENO};
    return fd;
}

bool& log_file_initialized() {
    static bool initialized = false;
    return initialized;
}

std::mutex& log_file_init_mutex() {
    static std::mutex mutex;
    return mutex;
}

} // namespace

LogLevel current_log_level() {
    return level_storage().load(std::memory_order_relaxed);
}

void set_current_log_level(LogLevel level) {
    level_storage().store(level, std::memory_order_relaxed);
}

const char* log_level_to_string(LogLevel level) {
    static const char* const buffer[] = {
        "CRITICAL ",
        "ERROR    ",
        "WARN     ",
        "INFO     ",
        "TRACE    ",
        "DEBUG    ",
    };
    const size_t index = static_cast<size_t>(level);
    return (index < sizeof(buffer) / sizeof(*buffer)) ? buffer[index] : "UNKNOWN";
}

LogLevel parse_log_level(const char* level) {
    if (level == nullptr) return LogLevel::Info;
    if (strcasecmp(level, "DEBUG") == 0) return LogLevel::Debug;
    if (strcasecmp(level, "TRACE") == 0) return LogLevel::Trace;
    if (strcasecmp(level, "INFO") == 0) return LogLevel::Info;
    if (strcasecmp(level, "WARN") == 0) return LogLevel::Warn;
    if (strcasecmp(level, "ERROR") == 0) return LogLevel::Error;
    if (strcasecmp(level, "CRITICAL") == 0) return LogLevel::Critical;
    return LogLevel::Info;
}

void OutputWriter::Output(const char* data, size_t size) {
    const int fd = output_fd_storage().load(std::memory_order_relaxed);
    size_t remaining = size;
    while (remaining > 0) {
        const ssize_t ret = write(fd, data, remaining);
        if (ret < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (ret == 0) break;
        data += static_cast<size_t>(ret);
        remaining -= static_cast<size_t>(ret);
    }
}

bool OutputWriter::configure_file(const std::string& path) {
    return initialize_log_file(path);
}

bool initialize_log_file(const std::string& path) {
    if (path.empty()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(log_file_init_mutex());
    if (log_file_initialized()) {
        return output_fd_storage().load(std::memory_order_relaxed) != STDERR_FILENO;
    }

    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0666);
    if (fd < 0) {
        return false;
    }

    output_fd_storage().store(fd, std::memory_order_relaxed);
    log_file_initialized() = true;
    return true;
}

int current_log_fd() {
    return output_fd_storage().load(std::memory_order_relaxed);
}

} // namespace log
} // namespace sketch2
