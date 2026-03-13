// Declares the minimal INI reader interface.

#pragma once

#include "utils/shared_types.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace sketch2 {

// IniReader exists to load small configuration files like dataset metadata
// without introducing an external parser. It flattens section/key pairs into a
// simple lookup table and exposes typed accessors for the values the project uses.
class IniReader {
public:
    Ret init(const std::string& path);

    int get_int(const std::string& name, int def) const;
    std::string get_str(const std::string& name, const std::string& def) const;
    std::vector<std::string> get_str_list(const std::string& name) const;

private:
    std::unordered_map<std::string, std::string> values_;
};

} // namespace sketch2
