#include "ini_reader.h"
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <limits>

namespace sketch2 {

static std::string trim(const std::string& s) {
    size_t begin = 0;
    size_t end = s.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(s[begin]))) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(begin, end - begin);
}

static std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
        const size_t pos = s.find(',', start);
        const size_t end = (pos == std::string::npos) ? s.size() : pos;
        const std::string item = trim(s.substr(start, end - start));
        if (!item.empty()) {
            out.push_back(item);
        }
        if (pos == std::string::npos) {
            break;
        }
        start = pos + 1;
    }
    return out;
}

Ret IniReader::init(const std::string& path) {
    values_.clear();

    std::ifstream in(path);
    if (!in) {
        return Ret("IniReader: failed to open file: " + path);
    }

    std::string section;
    std::string line;
    while (std::getline(in, line)) {
        std::string cur = trim(line);
        if (cur.empty() || cur[0] == ';' || cur[0] == '#') {
            continue;
        }

        if (cur.front() == '[' && cur.back() == ']') {
            section = trim(cur.substr(1, cur.size() - 2));
            continue;
        }

        const size_t eq = cur.find('=');
        if (eq == std::string::npos) {
            continue;
        }

        const std::string key = trim(cur.substr(0, eq));
        const std::string value = trim(cur.substr(eq + 1));
        if (key.empty()) {
            continue;
        }

        const std::string full_key = section.empty() ? key : (section + "." + key);
        values_[full_key] = value;
    }

    return Ret(0);
}

int IniReader::get_int(const std::string& name, int def) const {
    const auto it = values_.find(name);
    if (it == values_.end()) {
        return def;
    }

    const char* str = it->second.c_str();
    char* end = nullptr;
    errno = 0;
    long value = std::strtol(str, &end, 10);
    if (errno != 0 || end == str) {
        throw std::runtime_error("IniReader::get_int: invalid int value");
    }
    while (*end != '\0') {
        if (!std::isspace(static_cast<unsigned char>(*end))) {
            throw std::runtime_error("IniReader::get_int: invalid int value");
        }
        ++end;
    }
    if (value < static_cast<long>(std::numeric_limits<int>::min()) ||
        value > static_cast<long>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("IniReader::get_int: invalid int value");
    }
    return static_cast<int>(value);
}

std::string IniReader::get_str(const std::string& name, const std::string& def) const {
    const auto it = values_.find(name);
    if (it == values_.end()) {
        return def;
    }
    return it->second;
}

std::vector<std::string> IniReader::get_str_list(const std::string& name) const {
    const auto it = values_.find(name);
    if (it == values_.end()) {
        return {};
    }
    return split_csv(it->second);
}

} // namespace sketch2
