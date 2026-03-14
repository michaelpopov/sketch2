// Declares the process-wide startup singleton used for automatic utility init.

#pragma once

#include <memory>
#include <string>

namespace sketch2 {

class ThreadPool;

// Singleton exists to run small one-time process startup hooks from a normal
// translation unit instead of from heavy public headers. Its constructor is
// invoked by the static instance in singleton.cpp when the process starts or a
// shared library containing Sketch2 utilities is loaded.
class Singleton {
public:
    Singleton();

    static Singleton& instance();
    static bool apply_config_from_env();
    static bool apply_config_file(const std::string& path);

    std::shared_ptr<ThreadPool> thread_pool() const;

private:
    bool apply_config_from_env_();
    bool apply_config_file_(const std::string& path);

    std::shared_ptr<ThreadPool> thread_pool_;
};

Singleton& get_singleton();

} // namespace sketch2
