#include "internal.h"

#include "core/compute/scanner.h"
#include "core/storage/input_generator.h"
#include "core/utils/log.h"
#include "core/utils/string_utils.h"
#include "core/utils/timer.h"

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

using namespace sketch2;

namespace sketch2api::detail {

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

int sk_create_(sk_handle_t* handle, const char* name, unsigned int dim, const char* type,
        unsigned int range_size, const char* dist_func) {
    DECL

    if (handle->db_root.empty()) {
        ERR("Invalid db root")
    }
    if (!is_valid_dataset_name(name)) {
        ERR("Invalid dataset name")
    }
    if (dim < 4 || dim > std::numeric_limits<uint16_t>::max()) {
        ERR("Invalid dim parameter")
    }
    if (range_size <= 10) {
        ERR("Invalid range parameter")
    }

    const Ret type_ret = validate_dataset_type(type);
    if (type_ret.code() != 0) {
        ERR(type_ret.message().c_str())
    }
    const Ret dist_ret = validate_dataset_dist_func(dist_func);
    if (dist_ret.code() != 0) {
        ERR(dist_ret.message().c_str())
    }

    std::filesystem::create_directories(handle->db_root);

    const std::filesystem::path dir_path = dataset_dir_path(handle, name);
    const std::filesystem::path ini_path = dataset_ini_path(handle, name);
    const std::filesystem::path lock_path = dataset_lock_path(handle, name);

    if (std::filesystem::exists(dir_path) || std::filesystem::exists(ini_path) ||
        std::filesystem::exists(lock_path)) {
        ERR("Dataset already exists")
    }

    std::filesystem::create_directories(dir_path);

    FILE* ini = std::fopen(ini_path.c_str(), "w");
    if (ini == nullptr) {
        std::filesystem::remove_all(dir_path);
        ERR("Failed to open dataset ini file")
    }

    const int written = std::fprintf(ini,
        "[dataset]\n"
        "dirs=%s\n"
        "range_size=%u\n"
        "dim=%u\n"
        "type=%s\n"
        "dist_func=%s\n",
        dir_path.string().c_str(),
        range_size,
        dim,
        type,
        dist_func);
    const int close_rc = std::fclose(ini);
    if (written < 0 || close_rc != 0) {
        std::error_code ec;
        std::filesystem::remove(ini_path, ec);
        std::filesystem::remove_all(dir_path, ec);
        ERR("Failed to write dataset ini file")
    }

    FILE* lock = std::fopen(lock_path.c_str(), "w");
    if (lock == nullptr) {
        std::error_code ec;
        std::filesystem::remove(ini_path, ec);
        std::filesystem::remove_all(dir_path, ec);
        ERR("Failed to create dataset lock file")
    }

    const uint64_t update_notifier_counter = 0;
    const int lock_written = fwrite(&update_notifier_counter, sizeof(update_notifier_counter), 1, lock);
    const int lock_close_rc = std::fclose(lock);
    if (lock_written < 0 || lock_close_rc != 0) {
        std::error_code ec;
        std::filesystem::remove(lock_path, ec);
        std::filesystem::remove(ini_path, ec);
        std::filesystem::remove_all(dir_path, ec);
        ERR("Failed to write dataset lock file")
    }

    return sk_open(handle, name);
}

int sk_drop_(sk_handle_t* handle, const char* name) {
    DECL

    if (!is_valid_dataset_name(name)) {
        ERR("Invalid dataset name")
    }

    const std::filesystem::path dir_path = dataset_dir_path(handle, name);
    const std::filesystem::path ini_path = dataset_ini_path(handle, name);
    const std::filesystem::path lock_path = dataset_lock_path(handle, name);

    if (!std::filesystem::exists(ini_path)) {
        ERR("Dataset ini file is not present")
    }
    if (!std::filesystem::exists(lock_path)) {
        ERR("Dataset lock file is not present")
    }
    if (!std::filesystem::exists(dir_path)) {
        ERR("Dataset directory is not present")
    }

    std::unique_ptr<FileLockGuard> owner_lock;
    const Ret owner_lock_ret = lock_dataset_owner(ini_path, &owner_lock);
    if (owner_lock_ret.code() != 0) {
        ERR(owner_lock_ret.message().c_str())
    }

    if (handle->ds != nullptr && handle->dataset_name == name) {
        close_dataset(handle);
    }

    std::error_code ec;
    std::filesystem::remove(ini_path, ec);
    if (ec) {
        ERR("Failed to remove dataset ini file")
    }
    std::filesystem::remove(lock_path, ec);
    if (ec) {
        ERR("Failed to remove dataset lock file")
    }
    std::filesystem::remove_all(dir_path, ec);
    if (ec) {
        ERR("Failed to remove dataset directory")
    }

    return 0;
}

int sk_open_(sk_handle_t* handle, const char* name) {
    DECL

    if (handle->ds != nullptr) {
        ERR("Dataset is already open")
    }
    if (!is_valid_dataset_name(name)) {
        ERR("Invalid dataset name")
    }

    const std::filesystem::path ini_path = dataset_ini_path(handle, name);
    const std::filesystem::path lock_path = dataset_lock_path(handle, name);
    if (!std::filesystem::exists(ini_path)) {
        ERR("Dataset ini file is not present")
    }
    if (!std::filesystem::exists(lock_path)) {
        ERR("Dataset lock file is not present")
    }

    auto ds = std::make_unique<DatasetNode>();
    Ret ret = ds->init(ini_path.string());
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }

    handle->ds = std::move(ds);
    handle->dataset_name = name;
    handle->dataset_dir = dataset_dir_path(handle, name).string();
    handle->dataset_ini = ini_path.string();
    clear_cached_results(handle);

    return 0;
}

int sk_close_(sk_handle_t* handle, const char* name) {
    DECL

    if (handle->ds == nullptr) {
        ERR("No dataset is open")
    }
    if (!is_valid_dataset_name(name)) {
        ERR("Invalid dataset name")
    }
    if (handle->dataset_name != name) {
        ERR("Dataset name does not match the open dataset")
    }

    close_dataset(handle);
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
