// Defines the startup singleton instance for automatic utility initialization.

#include "singleton.h"

#include "ini_reader.h"
#include "log.h"
#include "thread_pool.h"

#include <cstdlib>

namespace sketch2 {

Singleton::Singleton() {
    (void)apply_config_from_env_();
}

Singleton& Singleton::instance() {
    static Singleton singleton;
    return singleton;
}

Singleton& get_singleton() {
    return Singleton::instance();
}

bool Singleton::apply_config_from_env() {
    return instance().apply_config_from_env_();
}

bool Singleton::apply_config_file(const std::string& path) {
    return instance().apply_config_file_(path);
}

std::shared_ptr<ThreadPool> Singleton::thread_pool() const {
    return thread_pool_;
}

bool Singleton::apply_config_from_env_() {
    const char* path = std::getenv("SKETCH2_CONFIG");
    if (path == nullptr || path[0] == '\0') {
        return false;
    }

    return apply_config_file_(path);
}

bool Singleton::apply_config_file_(const std::string& path) {
    IniReader reader;
    const Ret ret = reader.init(path);
    if (ret.code() != 0) {
        return false;
    }

    bool applied = false;

    const std::string level = reader.get_str("log.level", "");
    if (!level.empty()) {
        log::set_log_level(log::FILELog::from_string(level.c_str()));
        applied = true;
    }

    try {
        const int thread_pool_size = reader.get_int("thread_pool.size", 0);
        if (thread_pool_size > 1) {
            thread_pool_ = std::make_shared<ThreadPool>(static_cast<size_t>(thread_pool_size));
            applied = true;
        } else {
            thread_pool_.reset();
        }
    } catch (const std::exception&) {
        return false;
    }

    return applied;
}

namespace {

Singleton& g_singleton = Singleton::instance();

} // namespace

} // namespace sketch2
