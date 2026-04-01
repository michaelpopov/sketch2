#ifndef SKETCH2API_INTERNAL_H
#define SKETCH2API_INTERNAL_H

#include "utils.h"

namespace sketch2api::detail {

SKETCH2API_HIDDEN int sk_create_(
    sk_handle_t* handle, const char* name, unsigned int dim, const char* type,
    unsigned int range_size, const char* dist_func);
SKETCH2API_HIDDEN int sk_drop_(sk_handle_t* handle, const char* name);
SKETCH2API_HIDDEN int sk_open_(sk_handle_t* handle, const char* name);
SKETCH2API_HIDDEN int sk_close_(sk_handle_t* handle, const char* name);
SKETCH2API_HIDDEN int sk_knn_(
    sk_handle_t* handle, const char* vec, unsigned int k, uint64_t** ids_out, size_t* count_out);
SKETCH2API_HIDDEN int sk_mdelta_(sk_handle_t* handle);
SKETCH2API_HIDDEN int sk_get_(sk_handle_t* handle, uint64_t id, char** value_out);
SKETCH2API_HIDDEN int sk_print_(sk_handle_t* handle);
SKETCH2API_HIDDEN int sk_start_writing_(sk_handle_t* handle);
SKETCH2API_HIDDEN int sk_write_vector_(sk_handle_t* handle, uint64_t id, const char* data);
SKETCH2API_HIDDEN int sk_write_deleted_(sk_handle_t* handle, uint64_t id);
SKETCH2API_HIDDEN int sk_abort_writing_(sk_handle_t* handle);
SKETCH2API_HIDDEN int sk_complete_writing_(sk_handle_t* handle);
SKETCH2API_HIDDEN int sk_generate_(sk_handle_t* handle, uint64_t count, uint64_t start_id, int pattern);
SKETCH2API_HIDDEN int sk_generate_bin_(
    sk_handle_t* handle, uint64_t count, uint64_t start_id, int pattern);
SKETCH2API_HIDDEN int sk_generate_impl_(
    sk_handle_t* handle, uint64_t count, uint64_t start_id, int pattern, bool binary);
SKETCH2API_HIDDEN int sk_load_file_(sk_handle_t* handle, const char* path);
SKETCH2API_HIDDEN int sk_stats_(sk_handle_t* handle);

} // namespace sketch2api::detail

#endif // SKETCH2API_INTERNAL_H
