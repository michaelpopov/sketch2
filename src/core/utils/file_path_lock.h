#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace sketch2 {

class FilePathLock {
public:
    // Returns true if the path exists and was not already locked.
    bool check_file_path(const std::string& file_path);
    bool release_file_path(const std::string& file_path);

private:
    std::unordered_set<std::string> paths_;
    std::unordered_map<std::string, std::string> aliases_;
    std::mutex mutex_;
};

} // namespace sketch2
