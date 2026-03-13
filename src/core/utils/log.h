// Defines lightweight logging helpers used across the project.
//
// This header provides the project's lightweight stream-style logging macros:
//   LOG_INFO << "loaded " << count << " rows";
//   LOG_ERROR << "failed to open " << path;
//
// Typical setup looks like:
//   sketch2::log::set_log_level(sketch2::log::LogLevel::Debug);
//   LOG_DEBUG << "scanner started";
//   sketch2::log::configure_log_file("/tmp/sketch2.log");
//
// The logging macros are intended to be used as full statements and are safe in
// common control-flow forms:
//   if (ready)
//       LOG_INFO << "ready";
//   else
//       LOG_WARN << "not ready";
//
// Important caveats:
// - TempLogLevel changes a process-global log level, not a thread-local one. A
//   temporary override in one thread affects log filtering in every thread.
// - Output is written to stderr with write(2). This is simple and robust, but
//   it is still synchronous I/O and can become noticeable in very chatty paths.
// - Long messages are truncated to the fixed buffer size. The logger appends a
//   " [truncated]" marker when space allows so clipped output is visible.
// - The formatting helpers use localtime_r/strftime on each enabled log call,
//   so disabled logs are cheap but enabled logs are not free.
//
// For fatal paths, use:
//   CRITICAL_EXIT("invalid configuration");
//
// FixedBufferStreamBuf keeps log message assembly entirely on the stack. The
// previous implementation relied on std::ostringstream plus os.str(), which
// forced enabled log calls through dynamic stream state and an extra string copy
// before the bytes could be written. This buffer stores the message directly in
// a fixed array owned by the Log object, so the hot path stays predictable.
//
// The fixed capacity is an intentional tradeoff. Logging should remain usable
// even in low-memory or failure scenarios, so we prefer truncating oversized
// messages instead of allocating. The dropped suffix is usually less harmful
// than making the logging path depend on the heap.

#pragma once

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <mutex>
#include <ostream>
#include <string>
#include <streambuf>
#include <unistd.h>

namespace sketch2 {
namespace log {

enum class LogLevel { Critical, Error, Warn, Info, Trace, Debug };

class FixedBufferStreamBuf : public std::streambuf {
public:
    static constexpr size_t kCapacity = 4096;

    FixedBufferStreamBuf() : truncated_(false) { reset(); }

    void reset() {
        truncated_ = false;
        setp(buffer_, buffer_ + sizeof(buffer_));
    }

    const char* data() const { return pbase(); }

    size_t size() const { return static_cast<size_t>(pptr() - pbase()); }

    bool truncated() const { return truncated_; }

    void append_char(char ch) { overflow(static_cast<int>(ch)); }

    void append_suffix(const char* suffix) {
        const size_t suffix_len = std::strlen(suffix);
        if (suffix_len == 0) return;

        // When the main message already filled the fixed buffer we still want
        // the emitted line to end in a recognizable way. Rather than silently
        // dropping the suffix, overwrite the tail of the buffer so the sink can
        // still report that truncation happened and terminate the line.
        if (size() + suffix_len > kCapacity) {
            const size_t kept_len = (suffix_len < kCapacity) ? suffix_len : kCapacity;
            std::memcpy(buffer_ + (kCapacity - kept_len), suffix + (suffix_len - kept_len), kept_len);
            setp(buffer_, buffer_ + sizeof(buffer_));
            pbump(static_cast<int>(kCapacity));
            return;
        }

        xsputn(suffix, static_cast<std::streamsize>(suffix_len));
    }

protected:
    int overflow(int ch) override {
        if (ch == traits_type::eof()) return traits_type::not_eof(ch);

        if (pptr() == epptr()) {
            truncated_ = true;
            return traits_type::not_eof(ch);
        }

        *pptr() = static_cast<char>(ch);
        pbump(1);
        return traits_type::not_eof(ch);
    }

    std::streamsize xsputn(const char* s, std::streamsize count) override {
        const std::streamsize available = epptr() - pptr();
        const std::streamsize to_copy = (count < available) ? count : available;
        if (to_copy > 0) {
            std::memcpy(pptr(), s, static_cast<size_t>(to_copy));
            pbump(static_cast<int>(to_copy));
        }
        if (to_copy < count) truncated_ = true;

        // Report the full count to std::ostream even when truncating locally.
        // Returning a short write would mark the stream as failed and suppress
        // later << operations within the same log statement, which is not the
        // behavior we want. Truncation is tracked by this buffer, not via the
        // stream error state.
        return count;
    }

private:
    char buffer_[kCapacity];
    bool truncated_;
};

// Log exists to build log messages with stream syntax and flush them through a
// pluggable sink when the temporary object goes out of scope. It centralizes
// severity handling, timestamp formatting, and sink-specific output dispatch.
template <typename T>
class Log {
public:
    Log();
    ~Log();

    std::ostream& get(LogLevel level, const char* file, int line);

    static LogLevel level();
    static void set_level(LogLevel level);
    static const char* to_string(LogLevel level);
    static LogLevel from_string(const char* level);

protected:
    FixedBufferStreamBuf buffer_;
    std::ostream stream_;

private:
    static std::atomic<LogLevel>& level_storage();

    Log(const Log&) = delete;
    Log& operator=(const Log&) = delete;
};

template <typename T>
Log<T>::Log() : stream_(&buffer_) {}

template <typename T>
std::ostream& Log<T>::get(LogLevel level, const char* file, int line) {
    char header[256] = {'\0'};
    char str_time[64] = {'\0'};
    char str_date[64] = {'\0'};

    // Log objects are created as temporaries, but explicitly resetting the
    // stack buffer keeps the type reusable and clears any stream error flags
    // left behind by a previous truncation.
    buffer_.reset();
    stream_.clear();

    time_t t;
    time(&t);
    tm r;
    localtime_r(&t, &r);
    strftime(str_time, sizeof(str_time), "%X", &r);
    strftime(str_date, sizeof(str_date), "%F", &r);
    snprintf(header, sizeof(header), "%d %s %s ", getpid(), str_date, str_time);

    stream_ << header << to_string(level);
    if (file != nullptr) stream_ << file << ":" << line << "\t";

    return stream_;
}

template <typename T>
Log<T>::~Log() {
    // If the fixed buffer overflowed, overwrite the tail with a truncation
    // marker and newline if necessary. That guarantees the sink never emits a
    // clipped line that looks complete and ensures the final output still ends
    // cleanly even when the payload consumed the whole buffer.
    if (buffer_.truncated()) {
        buffer_.append_suffix(" [truncated]\n");
    } else {
        buffer_.append_char('\n');
    }

    // The sink receives the bytes directly from the stack buffer. This avoids
    // the extra std::string allocation that os.str() would force on every
    // enabled log statement.
    T::Output(buffer_.data(), buffer_.size());
}

template <typename T>
std::atomic<LogLevel>& Log<T>::level_storage() {
    static std::atomic<LogLevel> level{LogLevel::Info};
    return level;
}

template <typename T>
LogLevel Log<T>::level() {
    return level_storage().load(std::memory_order_relaxed);
}

template <typename T>
void Log<T>::set_level(LogLevel level) {
    level_storage().store(level, std::memory_order_relaxed);
}

template <typename T>
const char* Log<T>::to_string(LogLevel level) {
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

template <typename T>
LogLevel Log<T>::from_string(const char* level) {
    if (level == nullptr) return LogLevel::Info;
    if (strcasecmp(level, "DEBUG") == 0) return LogLevel::Debug;
    if (strcasecmp(level, "TRACE") == 0) return LogLevel::Trace;
    if (strcasecmp(level, "INFO") == 0) return LogLevel::Info;
    if (strcasecmp(level, "WARN") == 0) return LogLevel::Warn;
    if (strcasecmp(level, "ERROR") == 0) return LogLevel::Error;
    if (strcasecmp(level, "CRITICAL") == 0) return LogLevel::Critical;

    return LogLevel::Info;
}

// OutputWriter is the default sink for Log. It writes fully formatted log
// messages directly to stderr and retries interrupted or partial writes.
class OutputWriter {
public:
    // Output accepts a raw byte span rather than std::string so Log can flush
    // its stack buffer directly. That keeps the no-heap design intact all the
    // way to write(2) and removes the final copy from the enabled path.
    static void Output(const char* data, size_t size) {
        const int fd = output_fd().load(std::memory_order_relaxed);
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

    // set_fd redirects logging to an already-open file descriptor. Ownership
    // stays with the caller, which keeps this path convenient for tests and for
    // applications that already manage descriptor lifetime elsewhere.
    static void set_fd(int fd) {
        std::lock_guard<std::mutex> lock(config_mutex());
        close_owned_fd_locked();
        output_fd().store(fd, std::memory_order_relaxed);
        owns_fd() = false;
    }

    static int fd() {
        return output_fd().load(std::memory_order_relaxed);
    }

    // configure_file is the convenient "start logging to this file" helper for
    // normal application setup. The logger opens the file in append mode and
    // retains ownership so reset_fd() can safely restore stderr later.
    static bool configure_file(const std::string& path) {
        const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0666);
        if (fd < 0) return false;

        std::lock_guard<std::mutex> lock(config_mutex());
        close_owned_fd_locked();
        output_fd().store(fd, std::memory_order_relaxed);
        owns_fd() = true;
        return true;
    }

    static void reset_fd() {
        std::lock_guard<std::mutex> lock(config_mutex());
        close_owned_fd_locked();
        output_fd().store(STDERR_FILENO, std::memory_order_relaxed);
        owns_fd() = false;
    }

private:
    static std::atomic<int>& output_fd() {
        static std::atomic<int> fd{STDERR_FILENO};
        return fd;
    }

    static bool& owns_fd() {
        static bool owns = false;
        return owns;
    }

    static std::mutex& config_mutex() {
        static std::mutex mutex;
        return mutex;
    }

    static void close_owned_fd_locked() {
        if (owns_fd()) {
            const int fd = output_fd().load(std::memory_order_relaxed);
            if (fd >= 0 && fd != STDERR_FILENO) {
                (void)close(fd);
            }
        }
    }
};

class FILELog : public Log<OutputWriter> {};

inline bool should_log(LogLevel level) { return level <= FILELog::level(); }

// FILE_LOG is intentionally implemented as a single-iteration for loop rather
// than an if/else macro. The level expression is evaluated once, and the
// pointer variable is nulled in the increment step so the loop body can run at
// most once. This shape keeps the macro usable as a single statement in
// `if (cond) LOG_INFO << ...; else ...` without reintroducing the classic
// dangling-else bug.
#define FILE_LOG(level, file, line)                                                                      \
    for (const auto file_log_level__ = (level), *file_log_level_ptr__ = &file_log_level__;              \
         file_log_level_ptr__ != nullptr && ::sketch2::log::should_log(*file_log_level_ptr__);          \
         file_log_level_ptr__ = nullptr)                                                                 \
    ::sketch2::log::FILELog().get(*file_log_level_ptr__, (file), (line))

#define LOG_CRITICAL FILE_LOG(::sketch2::log::LogLevel::Critical, __FILE__, __LINE__)
#define LOG_ERROR FILE_LOG(::sketch2::log::LogLevel::Error, __FILE__, __LINE__)
#define LOG_WARN FILE_LOG(::sketch2::log::LogLevel::Warn, __FILE__, __LINE__)
#define LOG_INFO FILE_LOG(::sketch2::log::LogLevel::Info, __FILE__, __LINE__)
#define LOG_TRACE FILE_LOG(::sketch2::log::LogLevel::Trace, __FILE__, __LINE__)
#define LOG_DEBUG FILE_LOG(::sketch2::log::LogLevel::Debug, __FILE__, __LINE__)

#define CRITICAL_EXIT(x)                                                                                 \
    do {                                                                                                 \
        LOG_CRITICAL << x;                                                                               \
        exit(EXIT_FAILURE);                                                                              \
    } while (0)

// TempLogLevel exists to temporarily override the process-wide log level inside
// a scope and automatically restore the previous level when the scope exits.
class TempLogLevel {
public:
    TempLogLevel(LogLevel new_level) : m_old(FILELog::level()) { FILELog::set_level(new_level); }

    TempLogLevel(const char* level) : m_old(FILELog::level()) { FILELog::set_level(FILELog::from_string(level)); }

    TempLogLevel(const std::string& level) : m_old(FILELog::level()) {
        FILELog::set_level(FILELog::from_string(level.c_str()));
    }

    ~TempLogLevel() { FILELog::set_level(m_old); }

private:
    LogLevel m_old;
};

inline void set_log_level(LogLevel log_level) { FILELog::set_level(log_level); }

inline LogLevel get_log_level() { return FILELog::level(); }

inline void set_log_fd(int fd) { OutputWriter::set_fd(fd); }

inline int get_log_fd() { return OutputWriter::fd(); }

inline bool configure_log_file(const std::string& path) { return OutputWriter::configure_file(path); }

inline void reset_log_output() { OutputWriter::reset_fd(); }

} // namespace log
} // namespace sketch2
