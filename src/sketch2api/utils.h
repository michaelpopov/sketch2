#ifndef SKETCH2API_UTILS_H
#define SKETCH2API_UTILS_H

#include "sketch2api.h"

#include "core/storage/data_reader.h"
#include "core/storage/dataset_node.h"
#include "core/utils/file_lock.h"
#include "core/utils/shared_types.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#define SKETCH2API_HIDDEN __attribute__((visibility("hidden")))
#else
#define SKETCH2API_HIDDEN
#endif

struct sk_handle {
    sk_handle() {
        std::memset(message, 0, sizeof(message));
    }

    std::unique_ptr<sketch2::DatasetNode> ds;
    std::string db_root;
    std::string dataset_name;
    std::string dataset_dir;
    std::string dataset_ini;
    int error = 0;
    char message[256];
};

namespace sketch2api::detail {

SKETCH2API_HIDDEN void set_error(sk_handle_t* handle, const char* message);
SKETCH2API_HIDDEN bool is_valid_dataset_name(const char* name);
SKETCH2API_HIDDEN std::filesystem::path dataset_dir_path(const sk_handle_t* handle, const char* name);
SKETCH2API_HIDDEN std::filesystem::path dataset_ini_path(const sk_handle_t* handle, const char* name);
SKETCH2API_HIDDEN std::filesystem::path dataset_lock_path(const sk_handle_t* handle, const char* name);
SKETCH2API_HIDDEN void clear_cached_results(sk_handle_t* handle);
SKETCH2API_HIDDEN void close_dataset(sk_handle_t* handle);
SKETCH2API_HIDDEN sketch2::Ret validate_dataset_type(const char* type);
SKETCH2API_HIDDEN sketch2::Ret validate_dataset_dist_func(const char* dist_func);
SKETCH2API_HIDDEN sketch2::Ret lock_dataset_owner(
    const std::filesystem::path& ini_path, std::unique_ptr<sketch2::FileLockGuard>* owner_lock);
SKETCH2API_HIDDEN std::string vector_to_string(
    const uint8_t* data, sketch2::DataType type, uint16_t dim);
SKETCH2API_HIDDEN int print_reader_vectors(const sketch2::DataReader& reader);
SKETCH2API_HIDDEN std::vector<std::filesystem::path> collect_paths_with_extension(
    const std::filesystem::path& dir_path, const char* ext);
SKETCH2API_HIDDEN int print_stats_block(
    const std::string& label, size_t vectors_count, size_t deleted_count);

} // namespace sketch2api::detail

#endif // SKETCH2API_UTILS_H
