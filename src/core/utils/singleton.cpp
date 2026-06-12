// Defines the process-wide singleton instance used for explicit runtime initialization.

#include "singleton.h"

#include "ini_reader.h"
#include "log.h"
#include "thread_pool.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <system_error>
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

std::filesystem::path default_bitset_filter_spill_dir() {
    std::error_code ec;
    std::filesystem::path dir = std::filesystem::temp_directory_path(ec);
    if (ec || dir.empty()) {
        return "/tmp";
    }
    return dir;
}

} // namespace

Singleton::Singleton()
        : bitset_filter_spill_dir_(default_bitset_filter_spill_dir()) {}

Singleton& Singleton::instance() {
    static Singleton singleton;
    return singleton;
}

Singleton& get_singleton() {
    return Singleton::instance();
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

const std::shared_ptr<ThreadPool>& Singleton::thread_pool() const {
    return thread_pool_;
}

size_t Singleton::bitset_filter_spill_threshold_bytes() const {
    return bitset_filter_spill_threshold_bytes_;
}

const std::filesystem::path& Singleton::bitset_filter_spill_dir() const {
    return bitset_filter_spill_dir_;
}

bool Singleton::check_file_path(const std::string& file_path) {
    return file_path_lock_.check_file_path(file_path);
}

bool Singleton::release_file_path(const std::string& file_path) {
    return file_path_lock_.release_file_path(file_path);
}

// runtime_init_ always seals the singleton, even when it ends up using only
// defaults. Once the process commits to a runtime configuration, later init
// attempts are rejected so logging and threading stay fixed.
bool Singleton::runtime_init_() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) {
        return false;
    }

    ConfigValues values;
    const bool applied = collect_config_values_(nullptr, &values) && apply_config_values_(values, /*allow_defaults=*/true);
    initialized_ = true;

    check_swappiness_();
    check_disk_queue_();

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
    const bool applied = collect_config_values_(nullptr, &values) && apply_config_values_(values, /*allow_defaults=*/false);
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
    const bool applied = collect_config_values_(&path, &values) && apply_config_values_(values, /*allow_defaults=*/false);
    if (applied) {
        initialized_ = true;
    }
    return applied;
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
        LOG_INFO << "Initialize from file " << config_path;
        IniReader reader;
        const Ret ret = reader.init(config_path);
        if (ret.code() == 0) {
            merged.level = reader.get_str("log.level", "");
            merged.log_file = reader.get_str("log.path", "");
            merged.thread_pool_size = reader.get_str("thread_pool.size", "");
            merged.bitset_filter_spill_threshold_bytes =
                reader.get_str("bitset_filter.spill_threshold_bytes", "");
            merged.bitset_filter_spill_dir = reader.get_str("bitset_filter.spill_dir", "");
        } else {
            LOG_WARN << "Failed to read SKETCH2_CONFIG from " << config_path
                     << ": " << ret.message();
        }
    }

    const char* env_level = std::getenv("SKETCH2_LOG_LEVEL");
    if (env_level != nullptr && env_level[0] != '\0') {
        LOG_INFO << "Log level is set in env var: " << env_level;
        merged.level = env_level;
    }

    const char* env_log_file = std::getenv("SKETCH2_LOG_FILE");
    if (env_log_file != nullptr && env_log_file[0] != '\0') {
        LOG_INFO << "Log file is set in env var: " << env_log_file;
        merged.log_file = env_log_file;
    }

    const char* env_thread_pool_size = std::getenv("SKETCH2_THREAD_POOL_SIZE");
    if (env_thread_pool_size != nullptr && env_thread_pool_size[0] != '\0') {
        LOG_INFO << "Thread pool size is set in env var: " << env_thread_pool_size;
        merged.thread_pool_size = env_thread_pool_size;
    }

    const char* env_bitset_filter_spill_threshold_bytes =
        std::getenv("SKETCH2_BITSET_FILTER_SPILL_THRESHOLD_BYTES");
    if (env_bitset_filter_spill_threshold_bytes != nullptr
            && env_bitset_filter_spill_threshold_bytes[0] != '\0') {
        LOG_INFO << "Bitset filter spill threshold is set in env var: "
                 << env_bitset_filter_spill_threshold_bytes;
        merged.bitset_filter_spill_threshold_bytes = env_bitset_filter_spill_threshold_bytes;
    }

    const char* env_bitset_filter_spill_dir = std::getenv("SKETCH2_BITSET_FILTER_SPILL_DIR");
    if (env_bitset_filter_spill_dir != nullptr && env_bitset_filter_spill_dir[0] != '\0') {
        LOG_INFO << "Bitset filter spill directory is set in env var: " << env_bitset_filter_spill_dir;
        merged.bitset_filter_spill_dir = env_bitset_filter_spill_dir;
    }

    *values = std::move(merged);
    return true;
}

// Apply sinks before log level so any warnings or info messages emitted by
// later steps already flow to the final destination.
bool Singleton::apply_config_values_(const ConfigValues& values, bool allow_defaults) {
    bool applied = false;

    if (!values.log_file.empty()) {
        applied = apply_log_file_(values.log_file) || applied;
    }

    if (!values.level.empty()) {
        applied = apply_log_level_(values.level) || applied;
    }

    if (!values.thread_pool_size.empty()) {
        applied = apply_thread_pool_size_(values.thread_pool_size) || applied;
    } else if (allow_defaults) {
        applied = apply_default_thread_pool_size_() || applied;
    }

    if (!values.bitset_filter_spill_threshold_bytes.empty()) {
        applied = apply_bitset_filter_spill_threshold_bytes_(
            values.bitset_filter_spill_threshold_bytes) || applied;
    }

    if (!values.bitset_filter_spill_dir.empty()) {
        applied = apply_bitset_filter_spill_dir_(values.bitset_filter_spill_dir) || applied;
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

// Start a pool sized to the machine when no explicit configuration is given.
// Uses hardware_concurrency() and respects the same disable semantics as the
// explicit thread_pool.size path (<= 1 means no pool).
bool Singleton::apply_default_thread_pool_size_() {
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    if (hardware_threads <= 1) {
        LOG_INFO << "Thread pool left disabled because hardware_concurrency() returned "
                 << hardware_threads << ".";
        return true;  // configuration applied: stay disabled
    }

    const unsigned int effective_thread_pool_size =
        std::min(hardware_threads, max_thread_pool_size());

    thread_pool_ = std::make_shared<ThreadPool>(effective_thread_pool_size);
    LOG_INFO << "Started thread pool with " << effective_thread_pool_size
             << " threads (default based on hardware cores).";
    return true;
}

// Parse, clamp, and create the pool in one place so all initialization paths
// share the same enable/disable semantics and maximum-size protection.
bool Singleton::apply_thread_pool_size_(const std::string& size) {
    if (size.empty()) {
        return false;
    }

    try {
        size_t parsed_chars = 0;
        const int thread_pool_size = std::stoi(size, &parsed_chars);
        if (parsed_chars != size.size()) {
            return false;
        }
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

bool Singleton::apply_bitset_filter_spill_threshold_bytes_(const std::string& size) {
    if (size.empty()) {
        return false;
    }

    try {
        size_t parsed_chars = 0;
        const unsigned long long parsed = std::stoull(size, &parsed_chars);
        if (parsed_chars != size.size()
                || parsed > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
            return false;
        }
        bitset_filter_spill_threshold_bytes_ = static_cast<size_t>(parsed);
    } catch (const std::exception&) {
        return false;
    }

    return true;
}

bool Singleton::apply_bitset_filter_spill_dir_(const std::string& path) {
    if (path.empty()) {
        return false;
    }

    bitset_filter_spill_dir_ = path;
    return true;
}

void Singleton::check_swappiness_() const {
    std::ifstream swappiness_input("/proc/sys/vm/swappiness");
    if (!swappiness_input.is_open()) {
        return;
    }

    std::string swappiness_str;
    swappiness_input >> swappiness_str;
    if (swappiness_str.empty()) {
        return;
    }

    try {
        size_t parsed_chars = 0;
        const int swappiness = std::stoi(swappiness_str, &parsed_chars);
        if (parsed_chars != swappiness_str.size()) {
            return;
        }
        if (swappiness != 0) {
            LOG_WARN << "vm.swappiness is set to " << swappiness
                     << ". For best I/O performance, run `sudo sysctl vm.swappiness=0` "
                     << "and `echo 'vm.swappiness=0' | sudo tee /etc/sysctl.d/99-swappiness.conf`.";
        }
    } catch (const std::exception&) {
        return;
    }
}

void Singleton::check_disk_queue_() const {
    LOG_INFO << "Check disk queue length for the disk storing data. "
             << "Example of command that can improve performance: `echo 4096 | sudo tee /sys/block/sde/queue/read_ahead_kb`.";
}

} // namespace sketch2
