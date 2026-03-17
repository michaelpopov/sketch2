// Declares the process-wide startup singleton used for automatic utility init.
//
// Singleton exists to hold process-wide runtime configuration shared by the
// utilities library. The key behavior is that initialization is explicit now:
// there is no longer a startup static that configures logging or thread pools
// at shared-library load time.
//
// Expected initialization flow:
//   1. the host process sets env vars such as SKETCH2_CONFIG,
//      SKETCH2_LOG_LEVEL, SKETCH2_THREAD_POOL_SIZE, or SKETCH2_LOG_FILE
//   2. the host calls sketch2_runtime_init() once
//   3. the singleton merges config and keeps the resulting process-wide state
//
// In this codebase, explicit runtime init is triggered from:
// - sk_runtime_init() in the Parasol C API
// - the Python Parasol wrapper before sk_connect()
// - sqlite3_sketch2_init() when the SQLite extension is loaded directly
//
// Compute backend selection is also process-wide. The singleton chooses the
// best supported ComputeUnit when it is first created, optionally honoring the
// SKETCH2_COMPUTE_BACKEND environment override. Queries then reuse that fixed
// selection. Tests may override the selected backend explicitly.
//
// Configuration precedence is:
// - start from built-in defaults
// - if SKETCH2_CONFIG points to a readable ini file, read values from it
// - if SKETCH2_LOG_LEVEL is set, it overrides log.level
// - if SKETCH2_THREAD_POOL_SIZE is set, it overrides thread_pool.size
// - if SKETCH2_LOG_FILE is set, it selects the log sink
//
// SKETCH2_CONFIG is optional. If it is missing, initialization can still
// succeed using defaults and env overrides. If it is set but unreadable, the
// singleton logs a warning and still applies direct env overrides.
//
// The singleton is intentionally one-shot. The first successful initialization
// or config application seals the object, and later attempts return false
// without mutating process-wide state. That prevents log destination, log
// level, or thread-pool size from changing halfway through process execution.
//
// apply_config_from_env() and apply_config_file() remain available mainly for
// tests and focused initialization paths. Production callers should prefer
// sketch2_runtime_init().

#pragma once

#include "compute_unit.h"

#include <memory>
#include <mutex>
#include <string>

namespace sketch2 {

class ThreadPool;

class Singleton {
public:
    static Singleton& instance();
    static bool runtime_init();
    static bool apply_config_from_env();
    static bool apply_config_file(const std::string& path);
    static bool force_compute_unit_for_testing(ComputeBackendKind kind);
    // Install (threads > 1) or remove (threads <= 1) the process-wide thread pool.
    // Bypasses the initialized flag so tests can install and tear down a pool
    // around individual test cases regardless of singleton initialization order.
    static void force_thread_pool_for_testing(size_t threads);
    // Directly install (or clear, if null) a previously saved pool. Used to
    // restore whatever was in place before a test overrode it.
    static void force_thread_pool_for_testing(std::shared_ptr<ThreadPool> pool);

    const ComputeUnit& compute_unit() const;
    const std::shared_ptr<ThreadPool>& thread_pool() const;

private:
    Singleton();
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
    Singleton(Singleton&&) = delete;
    Singleton& operator=(Singleton&&) = delete;

    struct ConfigValues {
        std::string level;
        std::string thread_pool_size;
        std::string log_file;
    };

    bool runtime_init_();
    bool apply_config_from_env_();
    bool apply_config_file_(const std::string& path);
    bool force_compute_unit_for_testing_(ComputeBackendKind kind);
    bool collect_config_values_(const std::string* path, ConfigValues* values);
    bool apply_config_values_(const ConfigValues& values);
    bool apply_log_level_(const std::string& level);
    bool apply_thread_pool_size_(const std::string& size);
    bool apply_log_file_(const std::string& path);

    std::mutex mutex_;
    ComputeUnit compute_unit_;
    std::shared_ptr<ThreadPool> thread_pool_;
    bool initialized_ = false;
};

Singleton& get_singleton();
bool sketch2_runtime_init();

} // namespace sketch2
