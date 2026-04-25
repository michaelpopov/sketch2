#pragma once
#include "sketch2api_utils.h"

namespace sketch2api::detail {

SKETCH2API_HIDDEN int sk_create_(
    sk_handle_t* handle, const char* name, const char* dirs, unsigned int dim, const char* type,
    unsigned int range_size, const char* dist_func);
SKETCH2API_HIDDEN int sk_drop_(sk_handle_t* handle, const char* name);
SKETCH2API_HIDDEN int sk_open_(sk_handle_t* handle, const char* name);
SKETCH2API_HIDDEN int sk_close_(sk_handle_t* handle);
SKETCH2API_HIDDEN int sk_knn_(
    sk_handle_t* handle, const char* vec, unsigned int k, uint64_t** ids_out, size_t* count_out);
SKETCH2API_HIDDEN int sk_knn_vector_items_(
    sk_handle_t* handle, const float* vec, uint64_t vec_size,
    unsigned int k, const void* allowed_ids_blob, size_t allowed_ids_blob_size,
    uint64_t** ids_out, double** scores_out, size_t* count_out);
SKETCH2API_HIDDEN int sk_knn_items_(
    sk_handle_t* handle, const char* vec, unsigned int k,
    const void* allowed_ids_blob, size_t allowed_ids_blob_size,
    uint64_t** ids_out, double** scores_out, size_t* count_out);
SKETCH2API_HIDDEN int sk_knn_items_allowlist_(
    sk_handle_t* handle, const char* vec, unsigned int k,
    const void* allowed_ids, uint64_t** ids_out, double** scores_out, size_t* count_out);
SKETCH2API_HIDDEN int sk_score_ascending_is_better_(sk_handle_t* handle, bool* out);
SKETCH2API_HIDDEN const char* sk_knn_engine_name_for_testing_();
SKETCH2API_HIDDEN int sk_merge_delta_(sk_handle_t* handle);
SKETCH2API_HIDDEN int sk_get_(sk_handle_t* handle, uint64_t id, char** value_out);
SKETCH2API_HIDDEN int sk_print_(sk_handle_t* handle);
SKETCH2API_HIDDEN int sk_start_writing_(sk_handle_t* handle);
SKETCH2API_HIDDEN int sk_write_vector_(sk_handle_t* handle, uint64_t id, const char* data);
SKETCH2API_HIDDEN int sk_write_deleted_(sk_handle_t* handle, uint64_t id);
SKETCH2API_HIDDEN int sk_abort_writing_(sk_handle_t* handle);
SKETCH2API_HIDDEN int sk_complete_writing_(sk_handle_t* handle);
SKETCH2API_HIDDEN int sk_generate_test_data_(
    sk_handle_t* handle, const char* path, uint64_t count, uint64_t start_id,
    const char* pattern = nullptr, bool binary = false);
SKETCH2API_HIDDEN int sk_generate_test_metadata_(sk_handle_t* handle, 
    const char* path, uint64_t count, uint64_t start_id);
SKETCH2API_HIDDEN int sk_load_file_(sk_handle_t* handle, const char* path);
SKETCH2API_HIDDEN int sk_stats_(sk_handle_t* handle, const char* path = nullptr);

} // namespace sketch2api::detail
