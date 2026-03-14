// Declares the public C API exposed by the parasol layer.

#ifndef PARASOL_H
#define PARASOL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sk_handle sk_handle_t;

/*
 * Initialize process-wide Sketch2 runtime configuration once.
 */
int sk_runtime_init(void);

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
 * Upsert a vector encoded as text.
 */
int sk_upsert(sk_handle_t* handle, uint64_t id, const char* value);

/*
 * Upsert a vector filled with the same scalar in every component.
 */
int sk_ups2(sk_handle_t* handle, uint64_t id, double value);

/*
 * Delete a vector by id.
 */
int sk_del(sk_handle_t* handle, uint64_t id);

/*
 * Run KNN and cache ids on the handler.
 */
int sk_knn(sk_handle_t* handle, const char* vec, unsigned int k);

/*
 * Read cached KNN results directly. index=-1 returns count.
 * Returns 0 if no KNN result is cached or the index is invalid.
 */
uint64_t sk_kres(sk_handle_t* handle, int64_t index);

/*
 * Flush the accumulator to dataset files.
 */
int sk_macc(sk_handle_t* handle);

/*
 * Merge delta files into data files.
 */
int sk_mdelta(sk_handle_t* handle);

/*
 * Fetch a vector by id and cache its text form on the handler.
 */
int sk_get(sk_handle_t* handle, uint64_t id);

/*
 * Return the cached vector text. Returns an empty string if no value is cached.
 */
const char* sk_gres(sk_handle_t* handle);

/*
 * Find an id by exact vector value and cache it on the handler.
 */
int sk_gid(sk_handle_t* handle, const char* vec);

/*
 * Copy the cached id into value.
 */
int sk_ires(sk_handle_t* handle, uint64_t* value);

/*
 * Print the current dataset contents to stdout.
 */
int sk_print(sk_handle_t* handle);

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

#ifdef __cplusplus
}
#endif

#endif
