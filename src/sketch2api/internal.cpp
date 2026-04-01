#include "internal.h"

#include "core/compute/scanner.h"
#include "core/storage/input_generator.h"
#include "core/utils/log.h"
#include "core/utils/string_utils.h"
#include "core/utils/timer.h"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>
#include <system_error>

using namespace sketch2;

namespace sketch2api::detail {

namespace {

std::string trim_whitespace(const std::string& value) {
    size_t begin = 0;
    size_t end = value.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(begin, end - begin);
}

std::vector<std::string> split_dirs_list(const char* dirs) {
    std::vector<std::string> out;
    if (dirs == nullptr) {
        return out;
    }

    std::string value(dirs);
    size_t start = 0;
    while (start < value.size()) {
        const size_t comma = value.find(',', start);
        const std::string entry = value.substr(start,
            comma == std::string::npos ? std::string::npos : comma - start);
        const std::string trimmed = trim_whitespace(entry);
        if (!trimmed.empty()) {
            out.push_back(trimmed);
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return out;
}

std::vector<std::filesystem::path> resolve_data_dirs(
        const char* dirs, const std::filesystem::path& base_path) {
    std::vector<std::filesystem::path> out;
    const std::vector<std::string> tokens = split_dirs_list(dirs);
    if (tokens.empty()) {
        out.push_back(base_path);
        return out;
    }
    for (const std::string& token : tokens) {
        std::filesystem::path path(token);
        if (path.is_relative()) {
            path = base_path / path;
        }
        out.push_back(path);
    }
    return out;
}

std::string join_dirs(const std::vector<std::filesystem::path>& dirs) {
    std::string out;
    for (size_t i = 0; i < dirs.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += dirs[i].string();
    }
    return out;
}

} // namespace

#define ERR(x) { \
    set_error(handle, x); \
    return -1; \
}

#define DECL \
    if (handle == nullptr) { \
        return -1; \
    } \
    handle->error = 0; \
    handle->message[0] = '\0';

int sk_create_(sk_handle_t* handle, const char* name, const char* dirs, unsigned int dim,
        const char* type, unsigned int range_size, const char* dist_func) {
    DECL
    const std::string log_prefix = std::string("Create dataset ") + name + ": ";

    if (handle->db_root.empty()) {
        ERR(log_prefix + "Invalid db root")
    }
    if (!is_valid_dataset_name(name)) {
        ERR(log_prefix + "Invalid dataset name")
    }
    if (dim < 4 || dim > std::numeric_limits<uint16_t>::max()) {
        ERR(log_prefix + "Invalid dim parameter")
    }
    if (range_size <= 10) {
        ERR(log_prefix + "Invalid range parameter")
    }

    const Ret type_ret = validate_dataset_type(type);
    if (type_ret.code() != 0) {
        ERR(log_prefix + type_ret.message().c_str())
    }
    const Ret dist_ret = validate_dataset_dist_func(dist_func);
    if (dist_ret.code() != 0) {
        ERR(log_prefix + dist_ret.message().c_str())
    }

    std::filesystem::create_directories(handle->db_root);

    const std::filesystem::path dir_path = dataset_dir_path(handle, name);
    const std::filesystem::path ini_path = dataset_ini_path(handle, name);
    const std::filesystem::path lock_path = dataset_lock_path(handle, name);

    if (std::filesystem::exists(dir_path) || std::filesystem::exists(ini_path) ||
        std::filesystem::exists(lock_path)) {
        ERR(log_prefix + "Dataset already exists")
    }

    LOG_TRACE << log_prefix << "Create directory " << dir_path;
    std::filesystem::create_directories(dir_path);

    const std::vector<std::filesystem::path> data_dirs = resolve_data_dirs(dirs, dir_path);
    for (const auto& data_dir : data_dirs) {
        std::error_code ec;
        LOG_TRACE << log_prefix << "Create directory " << data_dir;
        std::filesystem::create_directories(data_dir, ec);
        if (ec) {
            ERR(log_prefix + "Failed to create dataset directories " + data_dir.string())
        }
    }
    const std::string dirs_value = join_dirs(data_dirs);

    LOG_TRACE << log_prefix << "Write config file " << ini_path;
    FILE* ini = std::fopen(ini_path.c_str(), "w");
    if (ini == nullptr) {
        ERR(log_prefix + "Failed to open dataset ini file " + ini_path.string())
    }

    const int written = std::fprintf(ini,
        "[dataset]\n"
        "dirs=%s\n"
        "range_size=%u\n"
        "dim=%u\n"
        "type=%s\n"
        "dist_func=%s\n",
        dirs_value.c_str(),
        range_size,
        dim,
        type,
        dist_func);
    const int close_rc = std::fclose(ini);
    if (written < 0 || close_rc != 0) {
        ERR(log_prefix + "Failed to write dataset ini file")
    }

    LOG_TRACE << log_prefix << "Create lock file " << lock_path;
    FILE* lock = std::fopen(lock_path.c_str(), "w");
    if (lock == nullptr) {
        ERR(log_prefix + "Failed to create dataset lock file " + lock_path.string())
    }

    const uint64_t update_notifier_counter = 0;
    const int lock_written = fwrite(&update_notifier_counter, sizeof(update_notifier_counter), 1, lock);
    const int lock_close_rc = std::fclose(lock);
    if (lock_written != 1 || lock_close_rc != 0) {
        ERR(log_prefix + "Failed to write dataset lock file " + lock_path.string())
    }

    LOG_TRACE << log_prefix << "Completed successfully.";
    return sk_open(handle, name);
}

int sk_drop_(sk_handle_t* handle, const char* name) {
    DECL
    const std::string log_prefix = std::string("Drop dataset ") + name + ": ";

    if (!is_valid_dataset_name(name)) {
        ERR("Invalid dataset name")
    }

    const std::filesystem::path dir_path = dataset_dir_path(handle, name);
    const std::filesystem::path ini_path = dataset_ini_path(handle, name);
    const std::filesystem::path lock_path = dataset_lock_path(handle, name);

    if (!std::filesystem::exists(ini_path)) {
        ERR(log_prefix + "Dataset ini file is not present")
    }
    if (!std::filesystem::exists(lock_path)) {
        ERR(log_prefix + "Dataset lock file is not present")
    }
    if (!std::filesystem::exists(dir_path)) {
        ERR(log_prefix + "Dataset directory is not present")
    }

    std::unique_ptr<FileLockGuard> owner_lock;
    const Ret owner_lock_ret = lock_dataset_owner(ini_path, &owner_lock);
    if (owner_lock_ret.code() != 0) {
        ERR(log_prefix + owner_lock_ret.message().c_str())
    }

    if (handle->ds != nullptr && handle->dataset_name == name) {
        close_dataset(handle);
    }

    std::error_code ec;

    LOG_TRACE << log_prefix << "Remove config file " << ini_path.string();
    std::filesystem::remove(ini_path, ec);
    if (ec) {
        ERR(log_prefix + "Failed to remove dataset ini file " + ini_path.string())
    }

    LOG_TRACE << log_prefix << "Remove lock file " << lock_path.string();
    std::filesystem::remove(lock_path, ec);
    if (ec) {
        ERR(log_prefix + "Failed to remove dataset lock file " + lock_path.string())
    }
    
    LOG_TRACE << log_prefix << "Remove directory " << dir_path.string();
    std::filesystem::remove_all(dir_path, ec);
    if (ec) {
        ERR(log_prefix + "Failed to remove dataset directory " + dir_path.string())
    }

    LOG_TRACE << log_prefix << "Completed successfully.";
    return 0;
}

int sk_open_(sk_handle_t* handle, const char* name) {
    DECL
    const std::string log_prefix = std::string("Open dataset ") + name + ": ";

    if (handle->ds != nullptr) {
        ERR(log_prefix + "Dataset is already open")
    }
    if (!is_valid_dataset_name(name)) {
        ERR(log_prefix + "Invalid dataset name")
    }

    const std::filesystem::path ini_path = dataset_ini_path(handle, name);
    const std::filesystem::path lock_path = dataset_lock_path(handle, name);
    if (!std::filesystem::exists(ini_path)) {
        ERR(log_prefix + "Dataset ini file is not present")
    }
    if (!std::filesystem::exists(lock_path)) {
        ERR(log_prefix + "Dataset lock file is not present")
    }

    LOG_TRACE << log_prefix << "Init dateset node with config file " << ini_path.string();
    auto ds = std::make_unique<DatasetNode>();
    Ret ret = ds->init(ini_path.string());
    if (ret.code() != 0) {
        ERR(log_prefix + ret.message().c_str())
    }

    handle->ds = std::move(ds);
    handle->dataset_name = name;
    handle->dataset_dir = dataset_dir_path(handle, name).string();
    handle->dataset_ini = ini_path.string();
    clear_cached_results(handle);

    LOG_TRACE << log_prefix << "Completed successfully.";
    return 0;
}

int sk_close_(sk_handle_t* handle, const char* name) {
    DECL
    const std::string log_prefix = std::string("Close dataset ") + name + ": ";

    if (handle->ds == nullptr) {
        ERR(log_prefix + "No dataset is open")
    }
    if (!is_valid_dataset_name(name)) {
        ERR(log_prefix + "Invalid dataset name")
    }
    if (handle->dataset_name != name) {
        ERR(log_prefix + "Dataset name does not match the open dataset")
    }

    close_dataset(handle);

    LOG_TRACE << log_prefix << "Completed successfully.";
    return 0;
}

int sk_knn_(sk_handle_t* handle, const char* vec, unsigned int k,
        uint64_t** ids_out, size_t* count_out) {
    DECL

    if (handle->ds == nullptr) {
        ERR("No dataset is open")
    }
    if (vec == nullptr || k == 0 || ids_out == nullptr || count_out == nullptr) {
        ERR("Invalid arguments")
    }

    *ids_out = nullptr;
    *count_out = 0;

    std::vector<uint8_t> buf(data_type_size(handle->ds->type()) * handle->ds->dim());
    Ret ret = parse_vector(
        buf.data(), buf.size(), handle->ds->type(), static_cast<uint16_t>(handle->ds->dim()), vec);
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }

    std::vector<DistItem> items;
    Scanner scanner;
    ret = scanner.find_items(handle->ds->reader_dataset(), k, buf.data(), items);
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }

    if (!items.empty()) {
        auto* ids = static_cast<uint64_t*>(std::malloc(items.size() * sizeof(uint64_t)));
        if (ids == nullptr) {
            ERR("Out of memory")
        }
        for (size_t i = 0; i < items.size(); ++i) {
            ids[i] = items[i].id;
        }
        *ids_out = ids;
    }
    *count_out = items.size();
    return 0;
}

int sk_mdelta_(sk_handle_t* handle) {
    DECL

    if (handle->ds == nullptr) {
        ERR("No dataset is open")
    }

    Ret ret = handle->ds->merge();
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }

    return 0;
}

int sk_get_(sk_handle_t* handle, uint64_t id, char** value_out) {
    DECL

    if (handle->ds == nullptr) {
        ERR("No dataset is open")
    }
    if (value_out == nullptr) {
        ERR("Invalid output parameter")
    }
    *value_out = nullptr;

    auto [vec_data, ret] = handle->ds->get_vector(id);
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }
    if (vec_data == nullptr) {
        ERR("Vector not found")
    }

    const std::string value = vector_to_string(
        vec_data, handle->ds->type(), static_cast<uint16_t>(handle->ds->dim()));
    auto* out = static_cast<char*>(std::malloc(value.size() + 1));
    if (out == nullptr) {
        ERR("Out of memory")
    }
    std::memcpy(out, value.c_str(), value.size() + 1);
    *value_out = out;
    return 0;
}

int sk_print_(sk_handle_t* handle) {
    DECL

    if (handle->ds == nullptr) {
        ERR("No dataset is open")
    }

    auto reader = handle->ds->reader();
    for (;;) {
        auto [part, ret] = reader->next();
        if (ret.code() != 0) {
            ERR(ret.message().c_str())
        }
        if (!part) {
            break;
        }
        if (print_reader_vectors(*part) != 0) {
            ERR("Failed to print dataset")
        }
    }

    return 0;
}

int sk_start_writing_(sk_handle_t* handle) {
    DECL

    if (handle->ds == nullptr) {
        ERR("No dataset is open")
    }

    Ret ret = handle->ds->start_writing();
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }

    return 0;
}

int sk_write_vector_(sk_handle_t* handle, uint64_t id, const char* data) {
    DECL

    if (handle->ds == nullptr) {
        ERR("No dataset is open")
    }
    if (data == nullptr || data[0] == '\0') {
        ERR("Invalid vector parameter")
    }

    Ret ret = handle->ds->write_vector(id, data);
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }

    return 0;
}

int sk_write_deleted_(sk_handle_t* handle, uint64_t id) {
    DECL

    if (handle->ds == nullptr) {
        ERR("No dataset is open")
    }

    Ret ret = handle->ds->write_deleted(id);
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }

    return 0;
}

int sk_abort_writing_(sk_handle_t* handle) {
    DECL

    if (handle->ds == nullptr) {
        ERR("No dataset is open")
    }

    Ret ret = handle->ds->abort_writing();
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }

    return 0;
}

int sk_complete_writing_(sk_handle_t* handle) {
    DECL

    if (handle->ds == nullptr) {
        ERR("No dataset is open")
    }

    Ret ret = handle->ds->complete_writing();
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }

    return 0;
}

int sk_generate_(sk_handle_t* handle, uint64_t count, uint64_t start_id, int pattern) {
    return sk_generate_impl_(handle, count, start_id, pattern, false);
}

int sk_generate_bin_(sk_handle_t* handle, uint64_t count, uint64_t start_id, int pattern) {
    return sk_generate_impl_(handle, count, start_id, pattern, true);
}

int sk_generate_impl_(sk_handle_t* handle, uint64_t count, uint64_t start_id, int pattern, bool binary) {
    DECL

    if (handle->ds == nullptr) {
        ERR("No dataset is open")
    }
    if (count == 0) {
        ERR("Invalid count parameter")
    }
    if (count > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        start_id > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        handle->ds->dim() > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        ERR("Arguments are too large")
    }

    PatternType pattern_type;
    if (pattern == 0) {
        pattern_type = PatternType::Sequential;
    } else if (pattern == 1) {
        pattern_type = PatternType::Detailed;
    } else {
        ERR("Invalid pattern parameter")
    }

    GeneratorConfig cfg;
    cfg.pattern_type = pattern_type;
    cfg.count = static_cast<size_t>(count);
    cfg.min_id = static_cast<size_t>(start_id);
    cfg.type = handle->ds->type();
    cfg.dim = static_cast<size_t>(handle->ds->dim());
    cfg.max_val = 1000;
    cfg.binary = binary;

    const std::filesystem::path input_path = std::filesystem::path(handle->dataset_dir) / kInputFileName;
    Timer generate_timer(binary ? "sk_generate_bin: generate input" : "sk_generate: generate input");
    Ret ret = generate_input_file(input_path.string(), cfg);
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }
    LOG_DEBUG << generate_timer.str();

    Timer store_timer(binary ? "sk_generate_bin: store input" : "sk_generate: store input");
    ret = handle->ds->store(input_path.string());
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }
    LOG_DEBUG << store_timer.str();

    return 0;
}

int sk_load_file_(sk_handle_t* handle, const char* path) {
    DECL

    if (handle->ds == nullptr) {
        ERR("No dataset is open")
    }
    if (path == nullptr || path[0] == '\0') {
        ERR("Invalid path parameter")
    }

    Ret ret = handle->ds->store(path);
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }

    return 0;
}

int sk_stats_(sk_handle_t* handle) {
    DECL

    if (handle->ds == nullptr) {
        ERR("No dataset is open")
    }

    if (std::fprintf(stdout,
            "dataset:\n"
            "    Name: %s\n"
            "    Type: %s\n"
            "    Dist: %s\n"
            "    Dim: %llu\n"
            "    Range: %llu\n"
            "    Ini path: %s\n"
            "    Data path: %s\n\n",
            handle->dataset_name.c_str(),
            data_type_to_string(handle->ds->type()),
            dist_func_to_string(handle->ds->dist_func()),
            static_cast<unsigned long long>(handle->ds->dim()),
            static_cast<unsigned long long>(handle->ds->range_size()),
            handle->dataset_ini.c_str(),
            handle->dataset_dir.c_str()) < 0) {
        ERR("Failed to print dataset stats")
    }

    const std::filesystem::path dir_path = handle->dataset_dir;
    for (const auto& path : collect_paths_with_extension(dir_path, ".data")) {
        DataReader reader;
        Ret ret = reader.init(path.string());
        if (ret.code() != 0) {
            ERR(ret.message().c_str())
        }
        if (print_stats_block(path.filename().string(), reader.count(), reader.deleted_count()) != 0) {
            ERR("Failed to print data file stats")
        }
    }

    for (const auto& path : collect_paths_with_extension(dir_path, ".delta")) {
        DataReader reader;
        Ret ret = reader.init(path.string());
        if (ret.code() != 0) {
            ERR(ret.message().c_str())
        }
        if (print_stats_block(path.filename().string(), reader.count(), reader.deleted_count()) != 0) {
            ERR("Failed to print delta file stats")
        }
    }

    return 0;
}

} // namespace sketch2api::detail
