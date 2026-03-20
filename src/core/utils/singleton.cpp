// Defines the process-wide singleton instance used for explicit runtime initialization.

#include "singleton.h"

#include "ini_reader.h"
#include "log.h"
#include "thread_pool.h"

#include <algorithm>
#include <cstdlib>
#include <thread>

namespace sketch2 {

namespace {

constexpr unsigned int kFallbackThreadPoolCap = 64;
constexpr unsigned int kHardThreadPoolCap = 256;
constexpr unsigned int kThreadPoolCapMultiplier = 4;

// Clamp configured worker counts relative to hardware so an aggressive config
// cannot spawn an unbounded number of threads on large machines.
unsigned int max_thread_pool_size() {
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    const unsigned int scaled_threads = hardware_threads == 0
        ? kFallbackThreadPoolCap
        : hardware_threads * kThreadPoolCapMultiplier;
    return std::max(2u, std::min(kHardThreadPoolCap, scaled_threads));
}

} // namespace

Singleton::Singleton()
    : compute_unit_(ComputeUnit::detect_best()) {}

Singleton& Singleton::instance() {
    static Singleton singleton;
    return singleton;
}

Singleton& get_singleton() {
    return Singleton::instance();
}

bool sketch2_runtime_init() {
    return Singleton::runtime_init();
}

bool Singleton::runtime_init() {
    return instance().runtime_init_();
}

bool Singleton::apply_config_from_env() {
    return instance().apply_config_from_env_();
}

bool Singleton::apply_config_file(const std::string& path) {
    return instance().apply_config_file_(path);
}

bool Singleton::force_compute_unit_for_testing(ComputeBackendKind kind) {
    return instance().force_compute_unit_for_testing_(kind);
}

void Singleton::force_thread_pool_for_testing(size_t threads) {
    std::lock_guard<std::mutex> lock(instance().mutex_);
    if (threads > 1) {
        instance().thread_pool_ = std::make_shared<ThreadPool>(threads);
    } else {
        instance().thread_pool_.reset();
    }
}

void Singleton::force_thread_pool_for_testing(std::shared_ptr<ThreadPool> pool) {
    std::lock_guard<std::mutex> lock(instance().mutex_);
    instance().thread_pool_ = std::move(pool);
}

const ComputeUnit& Singleton::compute_unit() const {
    return compute_unit_;
}

const std::shared_ptr<ThreadPool>& Singleton::thread_pool() const {
    return thread_pool_;
}

bool Singleton::check_file_path(const std::string& file_path) {
    return file_path_lock_.check_file_path(file_path);
}

bool Singleton::release_file_path(const std::string& file_path) {
    return file_path_lock_.release_file_path(file_path);
}

// runtime_init_ always seals the singleton, even when it ends up using only
// defaults. Once the process commits to a runtime configuration, later init
// attempts are rejected so logging, threading, and compute dispatch stay fixed.
bool Singleton::runtime_init_() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) {
        return false;
    }

    ConfigValues values;
    const bool applied = collect_config_values_(nullptr, &values) && apply_config_values_(values);
    initialized_ = true;
    return applied;
}

// This narrower helper is mainly for tests and focused init paths. Unlike
// runtime_init_, it only seals on success so callers can recover from bad env
// input and try another initialization path.
bool Singleton::apply_config_from_env_() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) {
        return false;
    }

    ConfigValues values;
    const bool applied = collect_config_values_(nullptr, &values) && apply_config_values_(values);
    if (applied) {
        initialized_ = true;
    }
    return applied;
}

// File-based init follows the same "seal only on success" rule as the env-only
// helper so callers can report or recover from a bad config file path.
bool Singleton::apply_config_file_(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) {
        return false;
    }

    ConfigValues values;
    const bool applied = collect_config_values_(&path, &values) && apply_config_values_(values);
    if (applied) {
        initialized_ = true;
    }
    return applied;
}

// Tests are allowed to force the active compute backend after initialization so
// resolver selection can be verified without rebuilding for each ISA variant.
bool Singleton::force_compute_unit_for_testing_(ComputeBackendKind kind) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ComputeUnit::is_supported(kind)) {
        return false;
    }
    compute_unit_ = ComputeUnit(kind);
    LOG_INFO << "Compute backend set to '" << compute_unit_.name()
             << "' because tests explicitly forced this runtime override.";
    return true;
}

// Merge configuration in precedence order: optional file first, then direct
// environment overrides. The merged struct is returned instead of mutating
// state incrementally so callers can decide when to seal the singleton.
bool Singleton::collect_config_values_(const std::string* path, ConfigValues* values) {
    if (values == nullptr) {
        return false;
    }

    ConfigValues merged;
    std::string config_path;
    if (path != nullptr) {
        config_path = *path;
    } else {
        const char* env_path = std::getenv("SKETCH2_CONFIG");
        if (env_path != nullptr && env_path[0] != '\0') {
            config_path = env_path;
        }
    }

    if (!config_path.empty()) {
        IniReader reader;
        const Ret ret = reader.init(config_path);
        if (ret.code() == 0) {
            merged.level = reader.get_str("log.level", "");
            merged.thread_pool_size = reader.get_str("thread_pool.size", "");
        } else {
            LOG_WARN << "Failed to read SKETCH2_CONFIG from " << config_path
                     << ": " << ret.message();
        }
    }

    const char* env_level = std::getenv("SKETCH2_LOG_LEVEL");
    if (env_level != nullptr && env_level[0] != '\0') {
        merged.level = env_level;
    }

    const char* env_thread_pool_size = std::getenv("SKETCH2_THREAD_POOL_SIZE");
    if (env_thread_pool_size != nullptr && env_thread_pool_size[0] != '\0') {
        merged.thread_pool_size = env_thread_pool_size;
    }

    const char* env_log_file = std::getenv("SKETCH2_LOG_FILE");
    if (env_log_file != nullptr && env_log_file[0] != '\0') {
        merged.log_file = env_log_file;
    }

    *values = std::move(merged);
    return true;
}

// Apply sinks before log level so any warnings or info messages emitted by
// later steps already flow to the final destination.
bool Singleton::apply_config_values_(const ConfigValues& values) {
    bool applied = false;

    if (!values.log_file.empty()) {
        applied = apply_log_file_(values.log_file) || applied;
    }

    if (!values.level.empty()) {
        applied = apply_log_level_(values.level) || applied;
    }

    if (!values.thread_pool_size.empty()) {
        applied = apply_thread_pool_size_(values.thread_pool_size) || applied;
    }

    return applied;
}

bool Singleton::apply_log_level_(const std::string& level) {
    if (level.empty()) {
        return false;
    }

    log::FILELog::set_level(log::FILELog::from_string(level.c_str()));
    return true;
}

// Parse, clamp, and create the pool in one place so all initialization paths
// share the same enable/disable semantics and maximum-size protection.
bool Singleton::apply_thread_pool_size_(const std::string& size) {
    if (size.empty()) {
        return false;
    }

    try {
        const int thread_pool_size = std::stoi(size);
        if (thread_pool_size > 1) {
            const unsigned int capped_thread_pool_size = max_thread_pool_size();
            const unsigned int requested_thread_pool_size = static_cast<unsigned int>(thread_pool_size);
            const unsigned int effective_thread_pool_size =
                std::min(requested_thread_pool_size, capped_thread_pool_size);
            if (effective_thread_pool_size != requested_thread_pool_size) {
                LOG_WARN << "Configured thread pool size " << requested_thread_pool_size
                         << " exceeds maximum " << capped_thread_pool_size
                         << "; clamping to " << effective_thread_pool_size << ".";
            }
            thread_pool_ = std::make_shared<ThreadPool>(effective_thread_pool_size);
            LOG_INFO << "Started thread pool with " << effective_thread_pool_size << " threads.";
        } else {
            thread_pool_.reset();
            LOG_INFO << "Thread pool disabled by configured size " << thread_pool_size << ".";
        }
    } catch (const std::exception&) {
        return false;
    }

    return true;
}

bool Singleton::apply_log_file_(const std::string& path) {
    if (path.empty()) {
        return false;
    }

    return log::initialize_log_file(path);
}

} // namespace sketch2
