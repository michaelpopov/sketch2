#include "internal.h"

#include "core/calc/scanner_ex.h"
#include "core/compute/compute_cos.h"
#include "core/storage/input_generator.h"
#include "core/utils/compute_unit.h"
#include "core/utils/log.h"
#include "core/utils/singleton.h"
#include "core/utils/string_utils.h"
#include "core/utils/timer.h"

#include <cassert>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <experimental/scope>
#include <filesystem>
#include <limits>
#include <new>
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

Ret bitset_file_path(const sk_handle_t* handle, const char* name, std::filesystem::path* out) {
    if (handle == nullptr || handle->ds == nullptr) {
        return Ret("No dataset is open");
    }
    if (!is_valid_dataset_name(name)) {
        return Ret("Invalid bitset name");
    }
    if (out == nullptr) {
        return Ret("Invalid output path argument");
    }

    const std::vector<std::string>& dirs = handle->ds->dirs();
    if (dirs.empty() || dirs.front().empty()) {
        return Ret("Dataset dirs are not set");
    }

    *out = std::filesystem::path(dirs.front()) / (std::string(name) + ".bitset");
    return Ret(0);
}

struct BitsetBuilderState {
    uint8_t* data = nullptr;
    size_t size = 0;
    size_t capacity = 0;
    uint64_t first_id = 0;
    uint64_t last_id = 0;
    bool has_first_id = false;
    bool has_error = false;
    bool has_nomem = false;
    const char* error_message = nullptr;
};

constexpr const char* kBitsetBuilderOrderError =
    "bitset builder: ids must be ordered in non-decreasing order";
constexpr uint64_t kBitsetBuilderMaxId = 100000000u;
constexpr const char* kBitsetBuilderMaxIdError =
    "bitset builder: id must be <= 100000000";
constexpr const char* kBitsetBuilderSizeError =
    "bitset builder: required bitset size exceeds supported limit";
constexpr const char* kBitsetBuilderNomemError = "sketch2: out of memory";
constexpr size_t kBitsetHeaderBytes = sizeof(uint64_t);

void set_bitset_builder_error(bool* out_of_memory, const char** error_message_out,
    bool is_nomem, const char* message) {
    if (out_of_memory != nullptr) {
        *out_of_memory = is_nomem;
    }
    if (error_message_out != nullptr) {
        *error_message_out = message;
    }
}

int return_bitset_builder_error(BitsetBuilderState* builder, bool is_nomem, const char* message,
        bool* out_of_memory, const char** error_message_out) {
    if (builder != nullptr) {
        builder->has_error = true;
        builder->has_nomem = is_nomem;
        builder->error_message = message;
    }
    set_bitset_builder_error(out_of_memory, error_message_out, is_nomem, message);
    return -1;
}

Ret build_bitset_filter(const void* allowed_ids_blob, size_t allowed_ids_blob_size,
        uint64_t* allowed_ids_base_id, const uint8_t** allowed_ids_bits,
        uint64_t* allowed_ids_bits_size, bool* has_allowed_ids_out) {
    if (allowed_ids_base_id == nullptr || allowed_ids_bits == nullptr ||
            allowed_ids_bits_size == nullptr || has_allowed_ids_out == nullptr) {
        return Ret("Invalid arguments");
    }

    const bool has_allowed_ids =
        !(allowed_ids_blob == nullptr && allowed_ids_blob_size == 0);
    if (has_allowed_ids && allowed_ids_blob == nullptr) {
        return Ret("Invalid allowed_ids blob argument");
    }

    *allowed_ids_base_id = 0;
    *allowed_ids_bits = static_cast<const uint8_t*>(allowed_ids_blob);
    *allowed_ids_bits_size = static_cast<uint64_t>(allowed_ids_blob_size);
    *has_allowed_ids_out = has_allowed_ids;

    if (has_allowed_ids) {
        if (allowed_ids_blob_size < sizeof(uint64_t)) {
            return Ret("allowed_ids blob is too small");
        }
        std::memcpy(allowed_ids_base_id, *allowed_ids_bits, sizeof(uint64_t));
        *allowed_ids_bits += sizeof(uint64_t);
        *allowed_ids_bits_size -= sizeof(uint64_t);
    }
    return Ret(0);
}

Ret extract_items_outputs(const std::vector<DistItem>& items,
        uint64_t** ids_out, double** scores_out, size_t* count_out) {
    if (ids_out == nullptr || scores_out == nullptr || count_out == nullptr) {
        return Ret("Invalid arguments");
    }

    *ids_out = nullptr;
    *scores_out = nullptr;
    *count_out = 0;

    if (!items.empty()) {
        auto* ids = static_cast<uint64_t*>(std::malloc(items.size() * sizeof(uint64_t)));
        auto* scores = static_cast<double*>(std::malloc(items.size() * sizeof(double)));
        if (ids == nullptr || scores == nullptr) {
            std::free(ids);
            std::free(scores);
            return Ret("Out of memory");
        }
        for (size_t i = 0; i < items.size(); ++i) {
            ids[i] = items[i].id;
            scores[i] = items[i].score;
        }
        *ids_out = ids;
        *scores_out = scores;
    }
    *count_out = items.size();
    return Ret(0);
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

    const std::vector<std::filesystem::path> data_dirs = resolve_data_dirs(dirs, dir_path);
    const std::string dirs_value = join_dirs(data_dirs);

    // Roll back all created filesystem artifacts on any failure below.
    bool success = false;
    std::experimental::scope_exit cleanup([&]() {
        if (success) return;
        std::error_code ec;
        for (const auto& dd : data_dirs) {
            std::filesystem::remove_all(dd, ec);
        }
        std::filesystem::remove_all(dir_path, ec);
    });

    LOG_TRACE << log_prefix << "Create directory " << dir_path;
    std::filesystem::create_directories(dir_path);

    for (const auto& data_dir : data_dirs) {
        std::error_code ec;
        LOG_TRACE << log_prefix << "Create directory " << data_dir;
        std::filesystem::create_directories(data_dir, ec);
        if (ec) {
            ERR(log_prefix + "Failed to create dataset directories " + data_dir.string())
        }
    }

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

    success = true;
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

int sk_close_(sk_handle_t* handle) {
    DECL
    const std::string log_prefix = std::string("Close dataset ") + handle->dataset_name + ": ";

    if (handle->ds == nullptr) {
        ERR(log_prefix + "No dataset is open")
    }

    close_dataset(handle);

    LOG_TRACE << log_prefix << "Completed successfully.";
    return 0;
}

Ret run_knn_items_query(
        const DatasetNode& dataset, const char* vec, size_t k,
        const BitsetFilter* bitset_filter, std::vector<DistItem>* items) {
    if (vec == nullptr || items == nullptr || k == 0) {
        return Ret("Invalid arguments");
    }

    const char* query = vec;
    std::string loaded_query;
    if (vec[0] == '@') {
        Ret load_ret = load_vector(vec + 1, loaded_query);
        if (load_ret.code() != 0) {
            return load_ret;
        }
        query = loaded_query.c_str();
    }

    std::vector<uint8_t> buf(data_type_size(dataset.type()) * dataset.dim());
    Ret ret = parse_vector(
        buf.data(), buf.size(), dataset.type(), static_cast<uint16_t>(dataset.dim()), query);
    if (ret.code() != 0) {
        return ret;
    }

    ScannerEx scanner{selected_calc_engine(get_singleton().compute_unit().kind())};
    return scanner.find_items(dataset.reader_dataset(), k, buf.data(), *items, bitset_filter);
}

Ret run_knn_items_query(
        const DatasetNode& dataset, const float* vec, uint64_t vec_size, size_t k,
        const BitsetFilter* bitset_filter, std::vector<DistItem>* items) {
    if (vec == nullptr || items == nullptr || k == 0) {
        return Ret("Invalid arguments");
    }
    if (vec_size != dataset.dim()) {
        return Ret("Invalid query vector size");
    }

    const uint64_t dataset_dim = dataset.dim();
    std::vector<uint8_t> buf(data_type_size(dataset.type()) * dataset_dim);
    Ret ret = convert_vector(buf.data(), buf.size(), dataset.type(), dataset_dim, vec, vec_size);
    if (ret.code() != 0) {
        return ret;
    }

    ScannerEx scanner{selected_calc_engine(get_singleton().compute_unit().kind())};
    return scanner.find_items(dataset.reader_dataset(), k, buf.data(), *items, bitset_filter);
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

    std::vector<DistItem> items;
    Ret ret = run_knn_items_query(*handle->ds, vec, static_cast<size_t>(k), nullptr, &items);
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

int sk_knn_vector_items_(sk_handle_t* handle, const float* vec, uint64_t vec_size, unsigned int k,
        const void* allowed_ids_blob, size_t allowed_ids_blob_size,
        uint64_t** ids_out, double** scores_out, size_t* count_out) {
    DECL

    if (handle->ds == nullptr) {
        ERR("No dataset is open")
    }
    if (vec == nullptr || vec_size == 0 || k == 0 ||
            ids_out == nullptr || scores_out == nullptr || count_out == nullptr) {
        ERR("Invalid arguments")
    }

    *ids_out = nullptr;
    *scores_out = nullptr;
    *count_out = 0;

    uint64_t allowed_ids_base_id = 0;
    const uint8_t* allowed_ids_bits = nullptr;
    uint64_t allowed_ids_bits_size = 0;
    bool has_allowed_ids = false;
    Ret ret = build_bitset_filter(
        allowed_ids_blob, allowed_ids_blob_size, &allowed_ids_base_id,
        &allowed_ids_bits, &allowed_ids_bits_size, &has_allowed_ids);
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }
    const BitsetFilter bitset_filter {
        .base_id = allowed_ids_base_id,
        .data = allowed_ids_bits,
        .size = allowed_ids_bits_size,
    };
    const BitsetFilter* bitset_filter_ptr = has_allowed_ids ? &bitset_filter : nullptr;

    std::vector<DistItem> items;
    ret = run_knn_items_query(
        *handle->ds, vec, vec_size, static_cast<size_t>(k), bitset_filter_ptr, &items);
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }

    ret = extract_items_outputs(items, ids_out, scores_out, count_out);
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }
    return 0;
}

int sk_knn_items_(sk_handle_t* handle, const char* vec, unsigned int k,
        const void* allowed_ids_blob, size_t allowed_ids_blob_size,
        uint64_t** ids_out, double** scores_out, size_t* count_out) {
    DECL

    if (handle->ds == nullptr) {
        ERR("No dataset is open")
    }
    if (vec == nullptr || k == 0 || ids_out == nullptr || scores_out == nullptr || count_out == nullptr) {
        ERR("Invalid arguments")
    }
    *ids_out = nullptr;
    *scores_out = nullptr;
    *count_out = 0;

    uint64_t allowed_ids_base_id = 0;
    const uint8_t* allowed_ids_bits = nullptr;
    uint64_t allowed_ids_bits_size = 0;
    bool has_allowed_ids = false;
    Ret ret = build_bitset_filter(
        allowed_ids_blob, allowed_ids_blob_size, &allowed_ids_base_id,
        &allowed_ids_bits, &allowed_ids_bits_size, &has_allowed_ids);
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }
    const BitsetFilter bitset_filter {
        .base_id = allowed_ids_base_id,
        .data = allowed_ids_bits,
        .size = allowed_ids_bits_size,
    };
    const BitsetFilter* bitset_filter_ptr = has_allowed_ids ? &bitset_filter : nullptr;

    std::vector<DistItem> items;
    ret = run_knn_items_query(*handle->ds, vec, static_cast<size_t>(k), bitset_filter_ptr, &items);
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }

    ret = extract_items_outputs(items, ids_out, scores_out, count_out);
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }
    return 0;
}

int sk_score_ascending_is_better_(sk_handle_t* handle, bool* out) {
    DECL

    if (handle->ds == nullptr) {
        ERR("No dataset is open")
    }
    if (out == nullptr) {
        ERR("Invalid output argument")
    }

    *out = smaller_score_is_better(handle->ds->dist_func());
    return 0;
}

const char* sk_knn_engine_name_for_testing_() {
    return calc_engine_name(selected_calc_engine(get_singleton().compute_unit().kind()));
}

int sk_merge_delta_(sk_handle_t* handle) {
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

    auto [value_str, ret] = handle->ds->get_vector_string(id, 2);
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }
    if (value_str.empty()) {
        ERR("Vector not found")
    }

    auto* out = static_cast<char*>(std::malloc(value_str.size() + 1));
    if (out == nullptr) {
        ERR("Out of memory")
    }
    std::memcpy(out, value_str.c_str(), value_str.size() + 1);
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

int sk_generate_test_data_(sk_handle_t* handle, const char* path, uint64_t count, uint64_t start_id, bool binary) {
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
    if (!path) {
        ERR("Invalid path")
    }

    PatternType pattern_type = PatternType::Sequential;

    GeneratorConfig cfg;
    cfg.pattern_type = pattern_type;
    cfg.count = static_cast<size_t>(count);
    cfg.min_id = static_cast<size_t>(start_id);
    cfg.type = handle->ds->type();
    cfg.dim = static_cast<size_t>(handle->ds->dim());
    cfg.max_val = 1000;
    cfg.binary = binary;
    if (handle->ds->dist_func() == DistFunc::COS) {
        cfg.pattern_type = PatternType::CosCompatible;
    } else if (handle->ds->dist_func() == DistFunc::DOT) {
        cfg.pattern_type = PatternType::DotCompatible;
    }

    Timer generate_timer("sk_generate: generate input");
    Ret ret = generate_input_file(path, cfg);
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }
    LOG_DEBUG << generate_timer.str();

    Timer store_timer("sk_generate: store input");
    ret = handle->ds->store(path);
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }
    LOG_DEBUG << store_timer.str();

    return 0;
}

SKETCH2API_HIDDEN int sk_generate_test_metadata_(sk_handle_t* handle, 
    const char* path, uint64_t count, uint64_t start_id) {
    (void)handle;
    Ret ret = generate_dummy_metadata(path, count, start_id);
    return ret.code();
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

int sk_bitset_create_(sk_handle_t* handle, const void* blob, size_t blob_size, const char* name) {
    DECL

    std::filesystem::path path;
    const Ret path_ret = bitset_file_path(handle, name, &path);
    if (path_ret.code() != 0) {
        ERR(path_ret.message().c_str())
    }
    if (blob == nullptr && blob_size > 0) {
        ERR("Invalid bitset blob argument")
    }

    std::filesystem::create_directories(path.parent_path());

    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        ERR("Failed to open bitset file for writing: " + path.string())
    }

    std::experimental::scope_exit cleanup([&]() {
        std::fclose(f);
    });

    if (blob_size > 0) {
        const size_t written = std::fwrite(blob, 1, blob_size, f);
        if (written != blob_size) {
            ERR("Failed to write bitset file: " + path.string())
        }
    }
    if (std::fflush(f) != 0) {
        ERR("Failed to flush bitset file: " + path.string())
    }

    return 0;
}

int sk_bitset_drop_(sk_handle_t* handle, const char* name) {
    DECL

    std::filesystem::path path;
    const Ret path_ret = bitset_file_path(handle, name, &path);
    if (path_ret.code() != 0) {
        ERR(path_ret.message().c_str())
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
    if (ec) {
        ERR("Failed to remove bitset file: " + path.string())
    }

    return 0;
}

int sk_bitset_load_(sk_handle_t* handle, const char* name, void** blob_out, size_t* blob_size_out) {
    DECL

    if (blob_out == nullptr || blob_size_out == nullptr) {
        ERR("Invalid output arguments")
    }
    *blob_out = nullptr;
    *blob_size_out = 0;

    std::filesystem::path path;
    const Ret path_ret = bitset_file_path(handle, name, &path);
    if (path_ret.code() != 0) {
        ERR(path_ret.message().c_str())
    }

    std::error_code ec;
    const auto file_size_u64 = std::filesystem::file_size(path, ec);
    if (ec) {
        ERR("Failed to stat bitset file: " + path.string())
    }
    if (file_size_u64 > static_cast<uintmax_t>(std::numeric_limits<size_t>::max())) {
        ERR("Bitset file is too large")
    }
    const size_t file_size = static_cast<size_t>(file_size_u64);

    void* buffer = nullptr;
    if (file_size > 0) {
        buffer = std::malloc(file_size);
        if (buffer == nullptr) {
            ERR("Out of memory")
        }
    }

    std::experimental::scope_exit cleanup([&]() {
        if (*blob_out == nullptr) {
            std::free(buffer);
        }
    });

    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        ERR("Failed to open bitset file for reading: " + path.string())
    }

    std::experimental::scope_exit close_file([&]() {
        std::fclose(f);
    });

    if (file_size > 0) {
        const size_t read = std::fread(buffer, 1, file_size, f);
        if (read != file_size) {
            ERR("Failed to read bitset file: " + path.string())
        }
    }

    *blob_out = buffer;
    *blob_size_out = file_size;
    return 0;
}

int sk_bitset_builder_add_(
        void** state, uint64_t id, bool* out_of_memory, const char** error_message_out) {
    set_bitset_builder_error(out_of_memory, error_message_out, false, nullptr);

    if (state == nullptr) {
        set_bitset_builder_error(out_of_memory, error_message_out, false,
            "bitset builder: invalid builder state");
        return -1;
    }

    auto* builder = static_cast<BitsetBuilderState*>(*state);
    if (builder == nullptr) {
        builder = new (std::nothrow) BitsetBuilderState();
        if (builder == nullptr) {
            set_bitset_builder_error(out_of_memory, error_message_out, true, kBitsetBuilderNomemError);
            return -1;
        }
        *state = builder;
    }
    if (builder->has_error) {
        set_bitset_builder_error(
            out_of_memory, error_message_out, builder->has_nomem, builder->error_message);
        return -1;
    }
    if (id > kBitsetBuilderMaxId) {
        return return_bitset_builder_error(
            builder, false, kBitsetBuilderMaxIdError, out_of_memory, error_message_out);
    }

    if (!builder->has_first_id) {
        builder->first_id = id;
        builder->has_first_id = true;
    } else if (id < builder->last_id) {
        return return_bitset_builder_error(
            builder, false, kBitsetBuilderOrderError, out_of_memory, error_message_out);
    }
    builder->last_id = id;

    const uint64_t relative_id = id - builder->first_id;
    const uint64_t byte_index_u64 = relative_id >> 3u;
    if (byte_index_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max() - kBitsetHeaderBytes - 1u)) {
        return return_bitset_builder_error(
            builder, false, kBitsetBuilderSizeError, out_of_memory, error_message_out);
    }

    const size_t byte_index = static_cast<size_t>(byte_index_u64);
    const size_t needed_size = kBitsetHeaderBytes + byte_index + 1u;
    if (needed_size > builder->capacity) {
        size_t new_capacity = builder->capacity > 0 ? builder->capacity : 1u;
        while (new_capacity < needed_size) {
            if (new_capacity > std::numeric_limits<size_t>::max() / 2u) {
                new_capacity = needed_size;
                break;
            }
            new_capacity *= 2u;
        }
        if (new_capacity < needed_size) {
            return return_bitset_builder_error(
                builder, false, kBitsetBuilderSizeError, out_of_memory, error_message_out);
        }

        uint8_t* new_data = static_cast<uint8_t*>(std::realloc(builder->data, new_capacity));
        if (new_data == nullptr) {
            return return_bitset_builder_error(
                builder, true, kBitsetBuilderNomemError, out_of_memory, error_message_out);
        }

        if (new_capacity > builder->capacity) {
            std::memset(new_data + builder->capacity, 0, new_capacity - builder->capacity);
        }
        if (builder->data == nullptr) {
            std::memcpy(new_data, &builder->first_id, sizeof(builder->first_id));
        }

        builder->data = new_data;
        builder->capacity = new_capacity;
    }
    if (needed_size > builder->size) {
        builder->size = needed_size;
    }

    builder->data[kBitsetHeaderBytes + byte_index] |= static_cast<uint8_t>(1u << (relative_id & 7u));
    return 0;
}

int sk_bitset_build_(
        uint64_t* ids, uint64_t count, void** blob_out, size_t* blob_size_out,
        bool* out_of_memory, const char** error_message_out) {
    set_bitset_builder_error(out_of_memory, error_message_out, false, nullptr);

    if (blob_out == nullptr || blob_size_out == nullptr) {
        set_bitset_builder_error(out_of_memory, error_message_out, false,
            "bitset builder: invalid output arguments");
        return -1;
    }
    *blob_out = nullptr;
    *blob_size_out = 0;

    if (count == 0) {
        return 0;
    }
    if (ids == nullptr) {
        set_bitset_builder_error(out_of_memory, error_message_out, false,
            "bitset builder: invalid ids argument");
        return -1;
    }

    const uint64_t first_id = ids[0];
    const uint64_t last_id = ids[count - 1];
    if (first_id > kBitsetBuilderMaxId || last_id > kBitsetBuilderMaxId) {
        set_bitset_builder_error(out_of_memory, error_message_out, false, kBitsetBuilderMaxIdError);
        return -1;
    }
    for (uint64_t i = 1; i < count; ++i) {
        if (ids[i] < ids[i - 1]) {
            set_bitset_builder_error(out_of_memory, error_message_out, false, kBitsetBuilderOrderError);
            return -1;
        }
    }

    const uint64_t relative_last_id = last_id - first_id;
    const uint64_t bitset_bytes_u64 = (relative_last_id >> 3u) + 1u;
    if (bitset_bytes_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max() - kBitsetHeaderBytes)) {
        set_bitset_builder_error(out_of_memory, error_message_out, false, kBitsetBuilderSizeError);
        return -1;
    }

    const size_t bitset_bytes = static_cast<size_t>(bitset_bytes_u64);
    const size_t blob_size = kBitsetHeaderBytes + bitset_bytes;
    uint8_t* blob = static_cast<uint8_t*>(std::malloc(blob_size));
    if (blob == nullptr) {
        set_bitset_builder_error(out_of_memory, error_message_out, true, kBitsetBuilderNomemError);
        return -1;
    }

    std::memset(blob, 0, blob_size);
    std::memcpy(blob, &first_id, sizeof(first_id));
    for (uint64_t i = 0; i < count; ++i) {
        const uint64_t relative_id = ids[i] - first_id;
        const size_t byte_index = static_cast<size_t>(relative_id >> 3u);
        blob[kBitsetHeaderBytes + byte_index] |= static_cast<uint8_t>(1u << (relative_id & 7u));
    }

    *blob_out = blob;
    *blob_size_out = blob_size;
    return 0;
}

int sk_bitset_builder_finish_(void** state, void** blob_out, size_t* blob_size_out,
        bool* out_of_memory, const char** error_message_out) {
    set_bitset_builder_error(out_of_memory, error_message_out, false, nullptr);

    if (blob_out == nullptr || blob_size_out == nullptr) {
        set_bitset_builder_error(out_of_memory, error_message_out, false,
            "bitset builder: invalid output arguments");
        return -1;
    }
    *blob_out = nullptr;
    *blob_size_out = 0;

    if (state == nullptr) {
        set_bitset_builder_error(out_of_memory, error_message_out, false,
            "bitset builder: invalid builder state");
        return -1;
    }

    auto* builder = static_cast<BitsetBuilderState*>(*state);
    *state = nullptr;
    if (builder == nullptr) {
        return 0;
    }

    std::experimental::scope_exit cleanup([&]() {
        std::free(builder->data);
        delete builder;
    });

    if (builder->has_error) {
        set_bitset_builder_error(
            out_of_memory, error_message_out, builder->has_nomem, builder->error_message);
        return -1;
    }
    if (builder->size == 0 || builder->data == nullptr || !builder->has_first_id) {
        return 0;
    }

    *blob_out = builder->data;
    *blob_size_out = builder->size;
    builder->data = nullptr;
    builder->size = 0;
    builder->capacity = 0;
    return 0;
}

int sk_stats_(sk_handle_t* handle, const char* path) {
    DECL

    if (handle->ds == nullptr) {
        ERR("No dataset is open")
    }

    FILE* output = nullptr;
    if (path == nullptr || *path == '\0') {
        output = stdout;
    } else {
        output = fopen(path, "w");
        if (output == nullptr) {
            ERR(std::string("Failed to open output file ") + path)
        }
    }

    std::experimental::scope_exit cleanup([&]() {
        if (output != stdout) {
            fclose(output);
        }
    });

    if (std::fprintf(output,
            "Name: %s\n"
            "Type: %s\n"
            "Dist: %s\n"
            "Dim: %llu\n"
            "Range: %llu\n"
            "Config path: %s\n"
            "Data path: %s\n\n",
            handle->dataset_name.c_str(),
            data_type_to_string(handle->ds->type()),
            dist_func_to_string(handle->ds->dist_func()),
            static_cast<unsigned long long>(handle->ds->dim()),
            static_cast<unsigned long long>(handle->ds->range_size()),
            handle->dataset_ini.c_str(),
            handle->dataset_dir.c_str()) < 0) {
        ERR("Failed to print dataset stats")
    }

    const auto& dirs = handle->ds->dirs();
    for (const auto& dir_str: dirs) {
        (void)fprintf(output, "==== Data path: %s\n", dir_str.c_str());
        const std::filesystem::path dir_path{dir_str};

        for (const auto& file_path : collect_paths_with_extension(dir_path, ".data")) {
            DataReader reader;
            Ret ret = reader.init(file_path);
            if (ret.code() != 0) {
                ERR(ret.message().c_str())
            }
            if (print_stats_block(output, file_path.filename().string(), reader.count(), reader.deleted_count()) != 0) {
                ERR("Failed to print data file stats")
            }
        }

        for (const auto& file_path : collect_paths_with_extension(dir_path, ".delta")) {
            DataReader reader;
            Ret ret = reader.init(file_path.string());
            if (ret.code() != 0) {
                ERR(ret.message().c_str())
            }
            if (print_stats_block(output, file_path.filename().string(), reader.count(), reader.deleted_count()) != 0) {
                ERR("Failed to print delta file stats")
            }
        }
    }

    return 0;
}

} // namespace sketch2api::detail
