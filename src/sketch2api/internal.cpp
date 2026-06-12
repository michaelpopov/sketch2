#include "internal.h"

#include "core/compute/highway.h"
#include "core/compute/scanner.h"
#include "core/storage/input_generator.h"
#include "core/bitset/bitset_filter_control.h"
#include "core/bitset/chunked_bits.h"
#include "core/utils/log.h"
#include "core/utils/shared_consts.h"
#include "core/utils/singleton.h"
#include "core/utils/string_utils.h"
#include "core/utils/timer.h"

#include <cassert>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <experimental/scope>
#include <filesystem>
#include <limits>
#include <new>
#include <string>
#include <vector>
#include <system_error>

namespace sketch2 {

bool sketch2_runtime_init() {
    const bool initialized = Singleton::runtime_init();
    if (initialized) {
        initialize_hwy_runtime();
    }
    return initialized;
}

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

std::string normalize_pattern_name(const char* pattern) {
    if (pattern == nullptr) {
        return "";
    }

    std::string normalized;
    normalized.reserve(std::strlen(pattern));
    for (const unsigned char ch : std::string(pattern)) {
        if (std::isspace(ch)) {
            continue;
        }
        normalized.push_back(ch == '-' ? '_' : static_cast<char>(std::tolower(ch)));
    }
    return normalized;
}

PatternType auto_pattern_type(DistFunc dist_func) {
    if (dist_func == DistFunc::COS) {
        return PatternType::CosCompatible;
    }
    if (dist_func == DistFunc::DOT) {
        return PatternType::DotCompatible;
    }
    return PatternType::Sequential;
}

Ret parse_pattern_type(const char* pattern, DistFunc dist_func, PatternType* out) {
    if (out == nullptr) {
        return Ret("Invalid pattern output");
    }

    const std::string normalized = normalize_pattern_name(pattern);
    if (normalized.empty() || normalized == "auto") {
        *out = auto_pattern_type(dist_func);
        return Ret(0);
    }
    if (normalized == "sequential") {
        *out = PatternType::Sequential;
        return Ret(0);
    }
    if (normalized == "detailed") {
        *out = PatternType::Detailed;
        return Ret(0);
    }
    if (normalized == "cos" || normalized == "cos_compatible") {
        *out = PatternType::CosCompatible;
        return Ret(0);
    }
    if (normalized == "dot" || normalized == "dot_compatible") {
        *out = PatternType::DotCompatible;
        return Ret(0);
    }
    if (normalized == "perf_test") {
        *out = PatternType::PerfTest;
        return Ret(0);
    }

    return Ret("Unknown test data pattern: " + normalized);
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

// Initialize `view` from a serialized allowed-ids blob. On success, sets
// `*present` to true when a real filter was supplied (non-null blob), or
// false when the caller passed nullptr/0 to request no filtering. The caller
// is responsible for wrapping the view as `BitsetFilter{.view = &view}`.
Ret init_allowed_ids_view(const void* allowed_ids_blob, size_t allowed_ids_blob_size,
        ChunkedBitsView* view, bool* present) {
    if (view == nullptr || present == nullptr) {
        return Ret("Invalid arguments");
    }
    *present = false;

    if (allowed_ids_blob == nullptr && allowed_ids_blob_size == 0) {
        return Ret(0);
    }
    if (allowed_ids_blob == nullptr) {
        return Ret("Invalid allowed_ids blob argument");
    }

    CHECK(view->init_blob(allowed_ids_blob, allowed_ids_blob_size));
    *present = true;
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
        "data_merge_ratio=%llu\n"
        "dim=%u\n"
        "type=%s\n"
        "dist_func=%s\n",
        dirs_value.c_str(),
        range_size,
        static_cast<unsigned long long>(kDataMergeRatio),
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

    Scanner scanner;
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

    Scanner scanner;
    return scanner.find_items(dataset.reader_dataset(), k, buf.data(), *items, bitset_filter);
}

Ret run_and_extract_knn_items_query(
        const DatasetNode& dataset, const char* vec, size_t k, const BitsetFilter* bitset_filter,
        uint64_t** ids_out, double** scores_out, size_t* count_out) {
    std::vector<DistItem> items;
    Ret ret = run_knn_items_query(dataset, vec, k, bitset_filter, &items);
    if (ret.code() != 0) {
        return ret;
    }
    return extract_items_outputs(items, ids_out, scores_out, count_out);
}

Ret run_and_extract_knn_items_query(
        const DatasetNode& dataset, const float* vec, uint64_t vec_size, size_t k,
        const BitsetFilter* bitset_filter, uint64_t** ids_out, double** scores_out,
        size_t* count_out) {
    std::vector<DistItem> items;
    Ret ret = run_knn_items_query(dataset, vec, vec_size, k, bitset_filter, &items);
    if (ret.code() != 0) {
        return ret;
    }
    return extract_items_outputs(items, ids_out, scores_out, count_out);
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

    ChunkedBitsView allowed_ids_view;
    bool has_filter = false;
    Ret ret = init_allowed_ids_view(
        allowed_ids_blob, allowed_ids_blob_size, &allowed_ids_view, &has_filter);
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }
    const BitsetFilter bitset_filter{.view = &allowed_ids_view};
    const BitsetFilter* bitset_filter_ptr = has_filter ? &bitset_filter : nullptr;

    ret = run_and_extract_knn_items_query(
        *handle->ds, vec, vec_size, static_cast<size_t>(k), bitset_filter_ptr,
        ids_out, scores_out, count_out);
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

    ChunkedBitsView allowed_ids_view;
    bool has_filter = false;
    Ret ret = init_allowed_ids_view(
        allowed_ids_blob, allowed_ids_blob_size, &allowed_ids_view, &has_filter);
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }
    const BitsetFilter bitset_filter{.view = &allowed_ids_view};
    const BitsetFilter* bitset_filter_ptr = has_filter ? &bitset_filter : nullptr;

    ret = run_and_extract_knn_items_query(
        *handle->ds, vec, static_cast<size_t>(k), bitset_filter_ptr,
        ids_out, scores_out, count_out);
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }
    return 0;
}

int sk_knn_items_bitset_filter_(sk_handle_t* handle, const char* vec, unsigned int k,
        const void* allowed_ids, uint64_t** ids_out, double** scores_out, size_t* count_out) {
    if (allowed_ids == nullptr) {
        return sk_knn_items_(handle, vec, k, nullptr, 0, ids_out, scores_out, count_out);
    }

    const auto* control = static_cast<const BitsetFilterControl*>(allowed_ids);
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

    const BitsetFilter bitset_filter{.view = &control->view};
    Ret ret = run_and_extract_knn_items_query(
        *handle->ds, vec, static_cast<size_t>(k), &bitset_filter,
        ids_out, scores_out, count_out);
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }
    return 0;
}

int sk_knn_vector_items_bitset_filter_(sk_handle_t* handle, const float* vec, uint64_t vec_size,
        unsigned int k, const void* allowed_ids,
        uint64_t** ids_out, double** scores_out, size_t* count_out) {
    if (allowed_ids == nullptr) {
        return sk_knn_vector_items_(handle, vec, vec_size, k, nullptr, 0, ids_out, scores_out, count_out);
    }

    const auto* control = static_cast<const BitsetFilterControl*>(allowed_ids);
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

    const BitsetFilter bitset_filter{.view = &control->view};
    Ret ret = run_and_extract_knn_items_query(
        *handle->ds, vec, vec_size, static_cast<size_t>(k), &bitset_filter,
        ids_out, scores_out, count_out);
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
    return "highway";
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

int sk_generate_test_data_(
        sk_handle_t* handle, const char* path, uint64_t count, uint64_t start_id,
        const char* pattern, bool binary) {
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
    Ret parse_ret = parse_pattern_type(pattern, handle->ds->dist_func(), &pattern_type);
    if (parse_ret.code() != 0) {
        ERR(parse_ret.message().c_str())
    }

    GeneratorConfig cfg;
    cfg.pattern_type = pattern_type;
    cfg.count = static_cast<size_t>(count);
    cfg.min_id = static_cast<size_t>(start_id);
    cfg.type = handle->ds->type();
    cfg.dim = static_cast<size_t>(handle->ds->dim());
    cfg.max_val = 1000;
    cfg.binary = binary;

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

int sk_import_data_(sk_handle_t* handle, const char* path) {
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

} // namespace sketch2
