// Declares a simple stack-scoped timer helper for tests and demos.

#pragma once

#include "utils/log.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>

namespace sketch2 {

// Timer exists to let tests and utilities measure elapsed wall-clock time
// without carrying explicit start timestamps around. The intended usage is:
//   Timer timer("merge");
//   run_merge();
//   std::cout << timer.str() << '\n';
//
// steady_clock is used because it is monotonic and therefore suitable for
// elapsed-time measurement even if the system clock changes.
class Timer {
public:
    explicit Timer(std::string_view name, bool log_on_destroy = false)
        : name_(name), start_(std::chrono::steady_clock::now()), log_on_destroy_(log_on_destroy) {}

    ~Timer() { if (log_on_destroy_) { LOG_DEBUG << str(); } }

    void reset() { start_ = std::chrono::steady_clock::now(); }

    const std::string& name() const { return name_; }

    std::chrono::steady_clock::duration elapsed() const { return std::chrono::steady_clock::now() - start_; }

    std::int64_t elapsed_ms() const { return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed()).count(); }

    std::int64_t elapsed_us() const { return std::chrono::duration_cast<std::chrono::microseconds>(elapsed()).count(); }

    std::int64_t operator()() const { return elapsed_ms(); }

    std::string str() const { return name_ + ": " + std::to_string(elapsed_ms()) + " ms"; }

private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
    bool log_on_destroy_ = false;
};

} // namespace sketch2
