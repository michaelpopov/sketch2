// Declares the public C API exposed by the sketch2api layer.

#ifndef SKETCH2API_H
#define SKETCH2API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sk_handle sk_handle_t;

/*
 * Initialize a handler for a database root directory.
 */
sk_handle_t* sk_connect(const char* db_path);

/*
 * Release resources associated with a handler.
 */
void sk_disconnect(sk_handle_t* handle);

/*
 * Create dataset metadata, lock file, and data directory under the handler root.
 */
int sk_create(sk_handle_t* handle, const char* name, unsigned int dim, const char* type,
    unsigned int range_size, const char* dist_func);

/*
 * Drop a dataset by name.
 */
int sk_drop(sk_handle_t* handle, const char* name);

/*
 * Open a dataset by name.
 */
int sk_open(sk_handle_t* handle, const char* name);

/*
 * Close the currently open dataset. The name must match the open dataset.
 */
int sk_close(sk_handle_t* handle, const char* name);

/*
 * Run KNN and return an allocated result array. The caller owns *ids_out and
 * must release it with sk_free(). count_out receives the number of ids.
 */
int sk_knn(sk_handle_t* handle, const char* vec, unsigned int k,
    uint64_t** ids_out, size_t* count_out);

/*
 * Merge delta files into data files.
 */
int sk_mdelta(sk_handle_t* handle);

/*
 * Fetch a vector by id and return an allocated text representation. The caller
 * owns *value_out and must release it with sk_free().
 */
int sk_get(sk_handle_t* handle, uint64_t id, char** value_out);

/*
 * Print the current dataset contents to stdout.
 */
int sk_print(sk_handle_t* handle);

/*
 * Start a staged write session for the currently open dataset. Subsequent
 * sk_write_vector() and sk_write_deleted() calls accumulate into a temporary
 * input file until sk_complete_writing() is called.
 */
int sk_start_writing(sk_handle_t* handle);

/*
 * Append one vector to the active staged write session. The vector payload is
 * parsed using the current dataset type and dimension.
 */
int sk_write_vector(sk_handle_t* handle, uint64_t id, const char* data);

/*
 * Append one deleted-id marker to the active staged write session.
 */
int sk_write_deleted(sk_handle_t* handle, uint64_t id);

/*
 * Abort the active staged write session, remove the temporary input file, and
 * discard any accumulated staged rows.
 */
int sk_abort_writing(sk_handle_t* handle);

/*
 * Finalize the active staged write session, load the accumulated input into
 * the current dataset, and remove the temporary input file.
 */
int sk_complete_writing(sk_handle_t* handle);

/*
 * Generate test vectors and load them into the current dataset.
 */
int sk_generate(sk_handle_t* handle, uint64_t count, uint64_t start_id, int pattern);

/*
 * Generate test vectors in binary input format and load them into the current dataset.
 */
int sk_generate_bin(sk_handle_t* handle, uint64_t count, uint64_t start_id, int pattern);

/*
 * Load vectors from a text or binary input file into the current dataset.
 */
int sk_load_file(sk_handle_t* handle, const char* path);

/*
 * Print dataset file statistics to stdout.
 */
int sk_stats(sk_handle_t* handle);

/*
 * Return the last error code.
 */
int sk_error(sk_handle_t* handle);

/*
 * Return the last error message.
 */
const char* sk_error_message(sk_handle_t* handle);

/*
 * Release memory returned by sketch2api allocation-returning functions.
 */
void sk_free(void* ptr);

#ifdef __cplusplus
}
#endif

#endif // SKETCH2API_H
