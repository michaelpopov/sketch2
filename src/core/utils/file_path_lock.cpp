#include "utils/file_path_lock.h"

#include <limits.h>
#include <unistd.h>

#include <array>

namespace sketch2 {

namespace {

bool resolve_real_path(const std::string& file_path, std::string* out) {
    if (out == nullptr) {
        return false;
    }

    std::array<char, PATH_MAX> resolved;
    if (realpath(file_path.c_str(), resolved.data()) == nullptr) {
        return false;
    }

    *out = resolved.data();
    return true;
}

} // namespace

bool FilePathLock::check_file_path(const std::string& file_path) {
    std::string real_path;
    if (!resolve_real_path(file_path, &real_path)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const auto [it, inserted] = paths_.emplace(real_path);
    if (inserted) {
        aliases_[file_path] = real_path;
    }
    return inserted;
}

bool FilePathLock::release_file_path(const std::string& file_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string real_path;
    auto alias_it = aliases_.find(file_path);
    if (alias_it != aliases_.end()) {
        real_path = alias_it->second;
    } else if (!resolve_real_path(file_path, &real_path)) {
        return false;
    }

    const auto it = paths_.find(real_path);
    if (it == paths_.end()) {
        return false;
    }
    paths_.erase(it);
    if (alias_it != aliases_.end()) {
        aliases_.erase(alias_it);
    }
    return true;
}

} // namespace sketch2
