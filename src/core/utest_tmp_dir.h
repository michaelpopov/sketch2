// Shared test utility: resolve the system temporary directory.
// Honours $TMPDIR when it is set and points to an existing directory,
// otherwise falls back to /tmp.

#pragma once

#include <cstdlib>
#include <filesystem>
#include <string>

namespace sketch2 {

inline std::string tmp_dir() {
    const char* env = std::getenv("TMPDIR");
    if (env && std::filesystem::is_directory(env)) {
        return std::string(env);
    }
    return "/tmp";
}

} // namespace sketch2
