// Implements the public C API for dataset lifecycle, mutation, and query operations.

#include "sketch2api.h"
#include "internal.h"
#include "utils.h"

#include "core/utils/singleton.h"

#include <cstdlib>
#include <filesystem>

using namespace sketch2;
using namespace sketch2api::detail;

namespace {

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

} // namespace

sk_handle_t* sk_new_handle(const char* db_path) {
    try {
        if (db_path == nullptr || db_path[0] == '\0') {
            return nullptr;
        }

        (void)sketch2_runtime_init();

        std::filesystem::path root = db_path;
        std::filesystem::create_directories(root);

        auto* handle = new sk_handle;
        handle->db_root = root.string();
        return handle;
    } catch (...) {
        return nullptr;
    }
}

void sk_release_handle(sk_handle_t* handle) {
    delete handle;
}

int sk_create(sk_handle_t* handle, const char* name, const char* dirs, unsigned int dim, const char* type,
        unsigned int range_size, const char* dist_func) {
    try {
        return sk_create_(handle, name, dirs, dim, type, range_size, dist_func);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}

int sk_drop(sk_handle_t* handle, const char* name) {
    try {
        return sk_drop_(handle, name);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}

int sk_open(sk_handle_t* handle, const char* name) {
    try {
        return sk_open_(handle, name);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}

int sk_close(sk_handle_t* handle) {
    try {
        return sk_close_(handle);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}

int sk_knn(sk_handle_t* handle, const char* vec, unsigned int k,
        uint64_t** ids_out, size_t* count_out) {
    try {
        return sk_knn_(handle, vec, k, ids_out, count_out);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}

int sk_merge_delta(sk_handle_t* handle) {
    try {
        return sk_merge_delta_(handle);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}

int sk_get(sk_handle_t* handle, uint64_t id, char** value_out) {
    try {
        return sk_get_(handle, id, value_out);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}

int sk_print(sk_handle_t* handle) {
    try {
        return sk_print_(handle);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}

int sk_start_writing(sk_handle_t* handle) {
    try {
        return sk_start_writing_(handle);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}

int sk_write_vector(sk_handle_t* handle, uint64_t id, const char* data) {
    try {
        return sk_write_vector_(handle, id, data);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}

int sk_write_deleted(sk_handle_t* handle, uint64_t id) {
    try {
        return sk_write_deleted_(handle, id);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}

int sk_abort_writing(sk_handle_t* handle) {
    try {
        return sk_abort_writing_(handle);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}

int sk_complete_writing(sk_handle_t* handle) {
    try {
        return sk_complete_writing_(handle);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}

int sk_generate(sk_handle_t* handle, uint64_t count, uint64_t start_id, int pattern) {
    try {
        return sk_generate_(handle, count, start_id, pattern);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}

int sk_generate_bin(sk_handle_t* handle, uint64_t count, uint64_t start_id, int pattern) {
    try {
        return sk_generate_bin_(handle, count, start_id, pattern);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}

int sk_load_file(sk_handle_t* handle, const char* path) {
    try {
        return sk_load_file_(handle, path);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}

int sk_stats(sk_handle_t* handle) {
    try {
        return sk_stats_(handle);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
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

void sk_free(void* ptr) {
    std::free(ptr);
}
