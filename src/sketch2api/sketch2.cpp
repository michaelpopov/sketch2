// Implements the public C API for dataset lifecycle, mutation, and query operations.

#include "sketch2.h"
#include "sketch2api_testing.h"
#include "internal.h"
#include "sketch2api_utils.h"

#include "core/compute/highway.h"
#include "core/utils/chunked_bits.h"
#include "core/utils/log.h"
#include "core/utils/shared_consts.h"
#include "core/utils/singleton.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <new>

using namespace sketch2;
using namespace sketch2::log;
using namespace sketch2api::detail;

namespace sketch2 {

bool sketch2_runtime_init() {
    const bool initialized = Singleton::runtime_init();
    if (initialized) {
        initialize_hwy_runtime();
    }
    return initialized;
}

} // namespace sketch2

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

bool align_up_size(size_t value, size_t alignment, size_t* out) {
    const size_t mask = alignment - 1u;
    if (value > std::numeric_limits<size_t>::max() - mask) {
        return false;
    }
    *out = (value + mask) & ~mask;
    return true;
}

void set_builder_error(bool* out_of_memory, const char** error_message_out,
        bool is_nomem, const char* message) {
    if (out_of_memory != nullptr) {
        *out_of_memory = is_nomem;
    }
    if (error_message_out != nullptr) {
        *error_message_out = message;
    }
}

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

int sk_knn_vector_items(sk_handle_t* handle, const float* vec, uint64_t vec_size, unsigned int k,
        const void* allowed_ids_blob, size_t allowed_ids_blob_size,
        uint64_t** ids_out, double** scores_out, size_t* count_out) {
    try {
        return sk_knn_vector_items_(
            handle, vec, vec_size, k, allowed_ids_blob, allowed_ids_blob_size,
            ids_out, scores_out, count_out);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}

int sk_knn_items(sk_handle_t* handle, const char* vec, unsigned int k,
        const void* allowed_ids_blob, size_t allowed_ids_blob_size,
        uint64_t** ids_out, double** scores_out, size_t* count_out) {
    try {
        return sk_knn_items_(handle, vec, k, allowed_ids_blob, allowed_ids_blob_size,
            ids_out, scores_out, count_out);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}

int sk_knn_items_allowlist(sk_handle_t* handle, const char* vec, unsigned int k,
        const void* allowed_ids, uint64_t** ids_out, double** scores_out, size_t* count_out) {
    try {
        return sk_knn_items_allowlist_(
            handle, vec, k, allowed_ids, ids_out, scores_out, count_out);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}

int sk_score_ascending_is_better(sk_handle_t* handle, bool* out) {
    try {
        return sk_score_ascending_is_better_(handle, out);
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

int sk_generate_test_data(sk_handle_t* handle, 
    const char* path, uint64_t count, uint64_t start_id, const char* pattern, bool binary) {
    try {
        return sk_generate_test_data_(handle, path, count, start_id, pattern, binary);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}

int sk_generate_test_metadata(sk_handle_t* handle, 
    const char* path, uint64_t count, uint64_t start_id) {
    try {
        return sk_generate_test_metadata_(handle, path, count, start_id);
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

int sk_stats(sk_handle_t* handle, const char* path) {
    try {
        return sk_stats_(handle, path);
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

int sk_allowlist_builder_add(
        void** state, uint64_t id, bool* out_of_memory, const char** error_message_out) {
    set_builder_error(out_of_memory, error_message_out, false, nullptr);
    if (state == nullptr) {
        set_builder_error(out_of_memory, error_message_out, false,
            "allowlist builder: invalid builder state");
        return -1;
    }

    try {
        auto* chunked_bits = static_cast<ChunkedBits*>(*state);
        if (chunked_bits == nullptr) {
            chunked_bits = new ChunkedBits();
            *state = chunked_bits;
        }

        const Ret ret = chunked_bits->add(id);
        if (ret.code() != 0) {
            set_builder_error(out_of_memory, error_message_out, false,
                "allowlist builder: add failed");
            return -1;
        }
        return 0;
    } catch (const std::bad_alloc&) {
        set_builder_error(out_of_memory, error_message_out, true, "sketch2: out of memory");
        return -1;
    } catch (const std::exception&) {
        set_builder_error(out_of_memory, error_message_out, false, "sketch2: internal error");
        return -1;
    } catch (...) {
        set_builder_error(out_of_memory, error_message_out, false, "sketch2: unexpected error");
        return -1;
    }
}

int sk_allowlist_builder_finish(
        void** state, void** out, bool* out_of_memory, const char** error_message_out) {
    set_builder_error(out_of_memory, error_message_out, false, nullptr);
    if (state == nullptr || out == nullptr) {
        set_builder_error(out_of_memory, error_message_out, false,
            "allowlist builder: invalid finish arguments");
        return -1;
    }
    *out = nullptr;

    auto* chunked_bits = static_cast<ChunkedBits*>(*state);
    *state = nullptr;

    try {
        if (chunked_bits == nullptr) {
            chunked_bits = new ChunkedBits();
        }
        std::unique_ptr<ChunkedBits> bits(chunked_bits);

        const Ret finish_ret = bits->finish();
        if (finish_ret.code() != 0) {
            set_builder_error(out_of_memory, error_message_out, false,
                "allowlist builder: finish failed");
            return -1;
        }

        const size_t blob_size = bits->serialized_size_bytes();
        size_t allocation_size = 0;
        if (blob_size == 0) {
            set_builder_error(out_of_memory, error_message_out, false,
                "allowlist builder: serialized blob size is unavailable");
            return -1;
        }
        if (!align_up_size(blob_size, kChunkedBitsBlobAlignment, &allocation_size)) {
            set_builder_error(out_of_memory, error_message_out, false,
                "allowlist builder: serialized blob is too large");
            return -1;
        }

        void* blob = std::aligned_alloc(kChunkedBitsBlobAlignment, allocation_size);
        if (blob == nullptr) {
            set_builder_error(out_of_memory, error_message_out, true, "sketch2: out of memory");
            return -1;
        }
        std::unique_ptr<void, decltype(&std::free)> blob_guard(blob, std::free);

        const Ret serialize_ret = bits->serialize(blob, blob_size);
        if (serialize_ret.code() != 0) {
            set_builder_error(out_of_memory, error_message_out, false,
                "allowlist builder: serialize failed");
            return -1;
        }

        auto view = std::make_unique<ChunkedBitsView>();
        const Ret view_ret = view->init_owned_blob(blob, blob_size);
        if (view_ret.code() != 0) {
            set_builder_error(out_of_memory, error_message_out, false,
                "allowlist builder: view init failed");
            return -1;
        }
        blob_guard.release();
        *out = view.release();
        return 0;
    } catch (const std::bad_alloc&) {
        set_builder_error(out_of_memory, error_message_out, true, "sketch2: out of memory");
        return -1;
    } catch (const std::exception&) {
        set_builder_error(out_of_memory, error_message_out, false, "sketch2: internal error");
        return -1;
    } catch (...) {
        set_builder_error(out_of_memory, error_message_out, false, "sketch2: unexpected error");
        return -1;
    }
}

void sk_release_allowlist(void* ptr) {
    delete static_cast<ChunkedBitsView*>(ptr);
}

void sk_set_log_level(const char* log_level) {
    if (!log_level) {
        return;
    }
    LogLevel level = parse_log_level(log_level);
    set_current_log_level(level);
}

void sk_version(char* buf, size_t buf_size) {
    if (buf == nullptr || buf_size == 0) {
        return;
    }
    const size_t version_len = std::strlen(kSketch2Version);
    const size_t copy_len = (version_len < (buf_size - 1)) ? version_len : (buf_size - 1);
    std::memcpy(buf, kSketch2Version, copy_len);
    buf[copy_len] = '\0';
}

const char* sk_knn_engine_name_for_testing(void) {
    try {
        return sk_knn_engine_name_for_testing_();
    } catch (...) {
        return "";
    }
}
