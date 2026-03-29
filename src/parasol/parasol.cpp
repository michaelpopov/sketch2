// Implements the public C API for dataset lifecycle, mutation, and query operations.

#include "parasol.h"

#include "core/compute/scanner.h"
#include "core/storage/data_reader.h"
#include "core/storage/dataset_node.h"
#include "core/utils/file_lock.h"
#include "core/utils/ini_reader.h"
#include "core/storage/input_generator.h"
#include "core/utils/shared_consts.h"
#include "core/utils/shared_types.h"
#include "core/utils/singleton.h"
#include "core/utils/string_utils.h"
#include "core/utils/log.h"
#include "core/utils/timer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace sketch2;

struct sk_handle {
    sk_handle() {
        std::memset(message, 0, sizeof(message));
    }

    std::unique_ptr<DatasetNode> ds;
    std::string db_root;
    std::string dataset_name;
    std::string dataset_dir;
    std::string dataset_ini;
    std::vector<uint64_t> knn_result;
    std::string get_result;
    bool has_id_result = false;
    uint64_t id_result = 0;
    int error = 0;
    char message[256];
};

namespace {

void set_error(sk_handle_t* handle, const char* message) {
    if (handle == nullptr) {
        return;
    }
    handle->error = -1;
    std::strncpy(handle->message, message, sizeof(handle->message) - 1);
    handle->message[sizeof(handle->message) - 1] = '\0';
}

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

// Restricts dataset names to a filesystem-safe subset used consistently by the
// API when deriving directory, ini, and lock file paths.
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
    return std::filesystem::path(handle->db_root) / (std::string(name) + ".ini");
}

std::filesystem::path dataset_lock_path(const sk_handle_t* handle, const char* name) {
    return std::filesystem::path(handle->db_root) / (std::string(name) + ".lock");
}

void clear_cached_results(sk_handle_t* handle) {
    handle->knn_result.clear();
    handle->get_result.clear();
    handle->has_id_result = false;
    handle->id_result = 0;
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

    owner_lock->reset(new FileLockGuard());
    return (*owner_lock)->lock(dirs.front() + "/" + kOwnerLockFileName);
}

// Formats a stored vector into text, growing the temporary buffer until
// print_vector reports success or a non-size-related failure.
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

// Fills a typed vector buffer with the same scalar value in every component,
// validating finiteness and numeric range before writing type-specific values.
Ret fill_vector_with_scalar(std::vector<uint8_t>* buf, DataType type, uint64_t dim, double value) {
    if (buf == nullptr) {
        return Ret("Invalid buffer");
    }
    if (!std::isfinite(value)) {
        return Ret("Value must be finite");
    }

    switch (type) {
        case DataType::f32: {
            auto* out = reinterpret_cast<float*>(buf->data());
            for (uint64_t i = 0; i < dim; ++i) {
                out[i] = static_cast<float>(value);
            }
            return Ret(0);
        }
        case DataType::i16: {
            if (value < static_cast<double>(std::numeric_limits<int16_t>::min()) ||
                value > static_cast<double>(std::numeric_limits<int16_t>::max())) {
                return Ret("Value is out of range for i16");
            }
            auto* out = reinterpret_cast<int16_t*>(buf->data());
            for (uint64_t i = 0; i < dim; ++i) {
                out[i] = static_cast<int16_t>(value);
            }
            return Ret(0);
        }
        case DataType::f16: {
            auto* out = reinterpret_cast<float16*>(buf->data());
            for (uint64_t i = 0; i < dim; ++i) {
                out[i] = static_cast<float16>(value);
            }
            return Ret(0);
        }
        default:
            return Ret("Unsupported data type");
    }
}

} // namespace

int sk_runtime_init(void) {
    try {
        (void)sketch2_runtime_init();
        return 0;
    } catch (...) {
        return -1;
    }
}

sk_handle_t* sk_connect(const char* db_path) {
    try {
        if (db_path == nullptr || db_path[0] == '\0') {
            return nullptr;
        }

        std::filesystem::path root = db_path;
        std::filesystem::create_directories(root);

        auto* handle = new sk_handle;
        handle->db_root = root.string();
        return handle;
    } catch (...) {
        return nullptr;
    }
}

void sk_disconnect(sk_handle_t* handle) {
    delete handle;
}

// Creates the dataset directory, ini, and lock files as one logical operation.
// On any write failure it removes partially created artifacts before returning an error.
static int sk_create_(sk_handle_t* handle, const char* name, unsigned int dim, const char* type,
        unsigned int range_size, const char* dist_func);
int sk_create(sk_handle_t* handle, const char* name, unsigned int dim, const char* type,
        unsigned int range_size, const char* dist_func) {
    try {
        return sk_create_(handle, name, dim, type, range_size, dist_func);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}
static int sk_create_(sk_handle_t* handle, const char* name, unsigned int dim, const char* type,
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

static int sk_drop_(sk_handle_t* handle, const char* name);
int sk_drop(sk_handle_t* handle, const char* name) {
    try {
        return sk_drop_(handle, name);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}
// Drops a dataset only after taking the owner lock so concurrent writers cannot
// recreate or mutate files while the directory tree is being removed.
static int sk_drop_(sk_handle_t* handle, const char* name) {
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

static int sk_open_(sk_handle_t* handle, const char* name);
int sk_open(sk_handle_t* handle, const char* name) {
    try {
        return sk_open_(handle, name);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}
// Opens an existing dataset, validates the on-disk metadata through Dataset::init,
// and refreshes all cached per-handle query results.
static int sk_open_(sk_handle_t* handle, const char* name) {
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

static int sk_close_(sk_handle_t* handle, const char* name);
int sk_close(sk_handle_t* handle, const char* name) {
    try {
        return sk_close_(handle, name);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}
static int sk_close_(sk_handle_t* handle, const char* name) {
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

static int sk_upsert_(sk_handle_t* handle, uint64_t id, const char* value);
int sk_upsert(sk_handle_t* handle, uint64_t id, const char* value) {
    try {
        return sk_upsert_(handle, id, value);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}
// Parses a textual vector into the dataset's binary type and forwards it to the
// dataset write path.
static int sk_upsert_(sk_handle_t* handle, uint64_t id, const char* value) {
    DECL

    if (handle->ds == nullptr) {
        ERR("No dataset is open")
    }
    if (value == nullptr) {
        ERR("Invalid vector parameter")
    }

    std::vector<uint8_t> buf(data_type_size(handle->ds->type()) * handle->ds->dim());
    Ret ret = parse_vector(
        buf.data(), buf.size(), handle->ds->type(), static_cast<uint16_t>(handle->ds->dim()), value);
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }

    ret = handle->ds->add_vector(id, buf.data());
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }

    return 0;
}

static int sk_ups2_(sk_handle_t* handle, uint64_t id, double value);
int sk_ups2(sk_handle_t* handle, uint64_t id, double value) {
    try {
        return sk_ups2_(handle, id, value);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}
// Builds a uniform vector from one scalar and inserts it into the open dataset.
static int sk_ups2_(sk_handle_t* handle, uint64_t id, double value) {
    DECL

    if (handle->ds == nullptr) {
        ERR("No dataset is open")
    }

    std::vector<uint8_t> buf(data_type_size(handle->ds->type()) * handle->ds->dim());
    Ret ret = fill_vector_with_scalar(&buf, handle->ds->type(), handle->ds->dim(), value);
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }

    ret = handle->ds->add_vector(id, buf.data());
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }

    return 0;
}

static int sk_del_(sk_handle_t* handle, uint64_t id);
int sk_del(sk_handle_t* handle, uint64_t id) {
    try {
        return sk_del_(handle, id);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}
static int sk_del_(sk_handle_t* handle, uint64_t id) {
    DECL

    if (handle->ds == nullptr) {
        ERR("No dataset is open")
    }

    Ret ret = handle->ds->delete_vector(id);
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }

    return 0;
}

static int sk_knn_(sk_handle_t* handle, const char* vec, unsigned int k);
int sk_knn(sk_handle_t* handle, const char* vec, unsigned int k) {
    try {
        return sk_knn_(handle, vec, k);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}
// Runs KNN against the currently open dataset by parsing the query vector,
// dispatching the scanner, and caching the resulting ids on the handle.
static int sk_knn_(sk_handle_t* handle, const char* vec, unsigned int k) {
    DECL

    if (handle->ds == nullptr) {
        ERR("No dataset is open")
    }
    if (vec == nullptr || k == 0) {
        ERR("Invalid arguments")
    }

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

    handle->knn_result.clear();
    handle->knn_result.reserve(items.size());
    for (const auto& item : items) {
        handle->knn_result.push_back(item.id);
    }
    return 0;
}

static uint64_t sk_kres_(sk_handle_t* handle, int64_t index);
uint64_t sk_kres(sk_handle_t* handle, int64_t index) {
    try {
        return sk_kres_(handle, index);
    } catch (const std::exception& ex) {
        set_error(handle, ex.what());
        return 0;
    }
}
static uint64_t sk_kres_(sk_handle_t* handle, int64_t index) {
    if (handle == nullptr) {
        return 0;
    }

    handle->error = 0;
    handle->message[0] = '\0';
    if (handle->knn_result.empty()) {
        return 0;
    }

    if (index == -1) {
        return static_cast<uint64_t>(handle->knn_result.size());
    }
    if (index < 0 || static_cast<size_t>(index) >= handle->knn_result.size()) {
        return 0;
    }

    return handle->knn_result[static_cast<size_t>(index)];
}

static int sk_macc_(sk_handle_t* handle);
int sk_macc(sk_handle_t* handle) {
    try {
        return sk_macc_(handle);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}
static int sk_macc_(sk_handle_t* handle) {
    DECL

    if (handle->ds == nullptr) {
        ERR("No dataset is open")
    }

    Ret ret = handle->ds->store_accumulator();
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }

    return 0;
}

static int sk_mdelta_(sk_handle_t* handle);
int sk_mdelta(sk_handle_t* handle) {
    try {
        return sk_mdelta_(handle);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}
static int sk_mdelta_(sk_handle_t* handle) {
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

static int sk_get_(sk_handle_t* handle, uint64_t id);
int sk_get(sk_handle_t* handle, uint64_t id) {
    try {
        return sk_get_(handle, id);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}
// Resolves a vector by id from the freshest visible dataset state and caches
// its text representation for sk_gres().
static int sk_get_(sk_handle_t* handle, uint64_t id) {
    DECL

    if (handle->ds == nullptr) {
        ERR("No dataset is open")
    }

    handle->get_result.clear();

    auto [vec_data, ret] = handle->ds->get_vector(id);
    if (ret.code() != 0) {
        ERR(ret.message().c_str())
    }
    if (vec_data == nullptr) {
        ERR("Vector not found")
    }

    handle->get_result = vector_to_string(
        vec_data, handle->ds->type(), static_cast<uint16_t>(handle->ds->dim()));
    return 0;
}

static const char* sk_gres_(sk_handle_t* handle);
const char* sk_gres(sk_handle_t* handle) {
    try {
        return sk_gres_(handle);
    } catch (const std::exception& ex) {
        set_error(handle, ex.what());
        return "";
    }
}
static const char* sk_gres_(sk_handle_t* handle) {
    if (handle == nullptr) {
        return "";
    }
    return handle->get_result.c_str();
}

static int sk_ires_(sk_handle_t* handle, uint64_t* value);
int sk_ires(sk_handle_t* handle, uint64_t* value) {
    try {
        return sk_ires_(handle, value);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}
static int sk_ires_(sk_handle_t* handle, uint64_t* value) {
    DECL

    if (value == nullptr) {
        ERR("Invalid output parameter")
    }
    if (!handle->has_id_result) {
        ERR("No id result is cached")
    }

    *value = handle->id_result;
    return 0;
}

static int sk_print_(sk_handle_t* handle);
int sk_print(sk_handle_t* handle) {
    try {
        return sk_print_(handle);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}
// Streams every persisted data reader to stdout in dataset order.
static int sk_print_(sk_handle_t* handle) {
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

static int sk_generate_impl_(sk_handle_t* handle, uint64_t count, uint64_t start_id, int pattern, bool binary);
static int sk_generate_(sk_handle_t* handle, uint64_t count, uint64_t start_id, int pattern);
int sk_generate(sk_handle_t* handle, uint64_t count, uint64_t start_id, int pattern) {
    try {
        return sk_generate_(handle, count, start_id, pattern);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}
// Generates a temporary input file using one of the built-in patterns and then
// imports it through the regular dataset store path.
static int sk_generate_(sk_handle_t* handle, uint64_t count, uint64_t start_id, int pattern) {
    return sk_generate_impl_(handle, count, start_id, pattern, false);
}

static int sk_generate_bin_(sk_handle_t* handle, uint64_t count, uint64_t start_id, int pattern);
int sk_generate_bin(sk_handle_t* handle, uint64_t count, uint64_t start_id, int pattern) {
    try {
        return sk_generate_bin_(handle, count, start_id, pattern);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}
// Generates a temporary binary input file using one of the built-in patterns
// and then imports it through the regular dataset store path.
static int sk_generate_bin_(sk_handle_t* handle, uint64_t count, uint64_t start_id, int pattern) {
    return sk_generate_impl_(handle, count, start_id, pattern, true);
}

static int sk_generate_impl_(sk_handle_t* handle, uint64_t count, uint64_t start_id, int pattern, bool binary) {
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

static int sk_load_file_(sk_handle_t* handle, const char* path);
int sk_load_file(sk_handle_t* handle, const char* path) {
    try {
        return sk_load_file_(handle, path);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}
// Reuses the regular dataset store path so bulk imports from Python can stream
// vectors into a temp text file and hand the whole file to C++ in one call.
static int sk_load_file_(sk_handle_t* handle, const char* path) {
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

static int sk_stats_(sk_handle_t* handle);
int sk_stats(sk_handle_t* handle) {
    try {
        return sk_stats_(handle);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}
// Prints high-level dataset metadata followed by per-file vector/deletion counts
// for the accumulator, base files, and delta files.
static int sk_stats_(sk_handle_t* handle) {
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

    if (print_stats_block(
            "accumulator",
            handle->ds->accumulator_vectors_count(),
            handle->ds->accumulator_deleted_count()) != 0) {
        ERR("Failed to print accumulator stats")
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

int sk_error(sk_handle_t* handle) {
    if (handle == nullptr) {
        return -1;
    }
    return handle->error;
}

const char* sk_error_message(sk_handle_t* handle) {
    if (handle == nullptr) {
        return "";
    }
    return handle->message;
}
