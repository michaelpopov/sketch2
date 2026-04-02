#include "utils.h"

#include "core/utils/ini_reader.h"
#include "core/utils/shared_consts.h"
#include "core/utils/string_utils.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

namespace sketch2api::detail {

using namespace sketch2;

void set_error(sk_handle_t* handle, const std::string& message) {
    if (handle == nullptr) {
        return;
    }
    handle->error = -1;
    std::strncpy(handle->message, message.c_str(), sizeof(handle->message) - 1);
    handle->message[sizeof(handle->message) - 1] = '\0';
}

bool is_valid_dataset_name(const char* name) {
    if (name == nullptr || name[0] == '\0') {
        return false;
    }

    for (const unsigned char* p = reinterpret_cast<const unsigned char*>(name); *p != '\0'; ++p) {
        const unsigned char c = *p;
        const bool is_ok =
            (c >= 'a' && c <= 'z') ||
            (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') ||
            c == '_' || c == '-' || c == '.';
        if (!is_ok) {
            return false;
        }
    }

    return true;
}

std::filesystem::path dataset_dir_path(const sk_handle_t* handle, const char* name) {
    return std::filesystem::path(handle->db_root) / name;
}

std::filesystem::path dataset_ini_path(const sk_handle_t* handle, const char* name) {
    const auto dataset_path = dataset_dir_path(handle, name);
    return dataset_path / (std::string(name) + ".ini");
}

std::filesystem::path dataset_lock_path(const sk_handle_t* handle, const char* name) {
    const auto dataset_path = dataset_dir_path(handle, name);
    return dataset_path / (std::string(name) + ".lock");
}

void clear_cached_results(sk_handle_t* handle) {
    (void)handle;
}

void close_dataset(sk_handle_t* handle) {
    handle->ds.reset();
    handle->dataset_name.clear();
    handle->dataset_dir.clear();
    handle->dataset_ini.clear();
    clear_cached_results(handle);
}

Ret validate_dataset_type(const char* type) {
    if (type == nullptr || type[0] == '\0') {
        return Ret("Invalid type parameter");
    }

    try {
        (void)data_type_from_string(type);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }

    return Ret(0);
}

Ret validate_dataset_dist_func(const char* dist_func) {
    if (dist_func == nullptr || dist_func[0] == '\0') {
        return Ret("Invalid distance function parameter");
    }

    try {
        validate_dist_func(dist_func_from_string(dist_func));
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }

    return Ret(0);
}

Ret lock_dataset_owner(const std::filesystem::path& ini_path, std::unique_ptr<FileLockGuard>* owner_lock) {
    if (owner_lock == nullptr) {
        return Ret("Invalid owner lock output parameter");
    }

    IniReader cfg;
    CHECK(cfg.init(ini_path.string()));
    const std::vector<std::string> dirs = cfg.get_str_list("dataset.dirs");
    if (dirs.empty()) {
        return Ret("Dataset dirs are not set");
    }

    const std::string dataset_name = std::filesystem::path(ini_path).stem().string();
    owner_lock->reset(new FileLockGuard());
    return (*owner_lock)->lock(dirs.front() + "/" + dataset_name + ".lock");
}

std::string vector_to_string(const uint8_t* data, DataType type, uint16_t dim) {
    size_t buf_size = std::max<size_t>(64, static_cast<size_t>(dim) * 32);
    for (;;) {
        std::vector<char> buf(buf_size);
        Ret ret = print_vector(const_cast<uint8_t*>(data), type, dim, buf.data(), buf.size());
        if (ret.code() == 0) {
            return std::string(buf.data());
        }
        if (ret.message().find("buffer is too small") == std::string::npos) {
            throw std::runtime_error(ret.message());
        }
        buf_size *= 2;
    }
}

int print_reader_vectors(const DataReader& reader) {
    const uint16_t dim = static_cast<uint16_t>(reader.dim());
    for (auto it = reader.begin(); !it.eof(); it.next()) {
        const std::string vec = vector_to_string(it.data(), reader.type(), dim);
        if (std::fprintf(stdout, "%llu : %s\n",
                static_cast<unsigned long long>(it.id()), vec.c_str()) < 0) {
            return -1;
        }
    }
    return 0;
}

std::vector<std::filesystem::path> collect_paths_with_extension(
        const std::filesystem::path& dir_path, const char* ext) {
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ext) {
            paths.push_back(entry.path());
        }
    }

    std::sort(paths.begin(), paths.end());
    return paths;
}

int print_stats_block(const std::string& label, size_t vectors_count, size_t deleted_count) {
    if (std::fprintf(stdout, "%s:\n", label.c_str()) < 0) {
        return -1;
    }
    if (std::fprintf(stdout, "    Vectors count: %zu\n", vectors_count) < 0) {
        return -1;
    }
    if (std::fprintf(stdout, "    Deleted count: %zu\n\n", deleted_count) < 0) {
        return -1;
    }
    return 0;
}

} // namespace sketch2api::detail
