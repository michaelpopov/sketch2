// Declares the process-wide startup singleton used for automatic utility init.
//
// Singleton exists to hold process-wide runtime configuration shared by the
// utilities library. The key behavior is that initialization is not performed
// at shared-library load time; it begins only when an integration surface such
// as sk_new_handle() triggers the higher-level runtime init helper, which then
// calls Singleton::runtime_init().
//
// Expected initialization flow:
//   1. the host process sets env vars such as SKETCH2_CONFIG,
//      SKETCH2_LOG_LEVEL, SKETCH2_THREAD_POOL_SIZE, or SKETCH2_LOG_FILE
//   2. a library entry point such as sk_new_handle() triggers runtime init
//   3. the singleton merges config and keeps the resulting process-wide state
//
// In this codebase, runtime init is triggered from:
// - sk_new_handle() in the Parasol C API
// - tests and focused init paths that call Singleton::runtime_init() directly
//
// Configuration precedence is:
// - start from built-in defaults
// - if SKETCH2_CONFIG points to a readable ini file, read values from it
// - if SKETCH2_LOG_LEVEL is set, it overrides log.level
// - if SKETCH2_THREAD_POOL_SIZE is set, it overrides thread_pool.size
// - if SKETCH2_LOG_FILE is set, it selects the log sink
// - if SKETCH2_BITSET_FILTER_SPILL_THRESHOLD_BYTES is set, it overrides
//   bitset_filter.spill_threshold_bytes
// - if SKETCH2_BITSET_FILTER_SPILL_DIR is set, it overrides bitset_filter.spill_dir
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
// tests and focused initialization paths. Production callers should prefer the
// higher-level runtime init helper in the API layer.

#pragma once

#include "utils/file_path_lock.h"

#include <cstddef>
#include <filesystem>
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
    // Install (threads > 1) or remove (threads <= 1) the process-wide thread pool.
    // Bypasses the initialized flag so tests can install and tear down a pool
    // around individual test cases regardless of singleton initialization order.
    static void force_thread_pool_for_testing(size_t threads);
    // Directly install (or clear, if null) a previously saved pool. Used to
    // restore whatever was in place before a test overrode it.
    static void force_thread_pool_for_testing(std::shared_ptr<ThreadPool> pool);

    const std::shared_ptr<ThreadPool>& thread_pool() const;
    size_t bitset_filter_spill_threshold_bytes() const;
    const std::filesystem::path& bitset_filter_spill_dir() const;
    bool check_file_path(const std::string& file_path);
    bool release_file_path(const std::string& file_path);

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
        std::string bitset_filter_spill_threshold_bytes;
        std::string bitset_filter_spill_dir;
    };

    bool runtime_init_();
    bool apply_config_from_env_();
    bool apply_config_file_(const std::string& path);
    bool collect_config_values_(const std::string* path, ConfigValues* values);
    bool apply_config_values_(const ConfigValues& values, bool allow_defaults);
    bool apply_log_level_(const std::string& level);
    bool apply_default_thread_pool_size_();
    bool apply_thread_pool_size_(const std::string& size);
    bool apply_log_file_(const std::string& path);
    bool apply_bitset_filter_spill_threshold_bytes_(const std::string& size);
    bool apply_bitset_filter_spill_dir_(const std::string& path);

    std::mutex mutex_;
    std::shared_ptr<ThreadPool> thread_pool_;
    size_t bitset_filter_spill_threshold_bytes_ = 1u << 20;
    std::filesystem::path bitset_filter_spill_dir_;
    FilePathLock file_path_lock_;
    bool initialized_ = false;

    void check_swappiness_() const;
    void check_disk_queue_() const;
};

Singleton& get_singleton();

} // namespace sketch2
