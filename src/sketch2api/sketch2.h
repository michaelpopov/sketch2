// Declares the public C API exposed by the sketch2api layer.

#ifndef SKETCH2_H
#define SKETCH2_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sk_handle sk_handle_t;

/*
 * Initialize a handle for a database root directory.
 */
sk_handle_t* sk_new_handle(const char* db_path);

/*
 * Release resources associated with a handle.
 */
void sk_release_handle(sk_handle_t* handle);

/*
 * Create dataset metadata, lock file, and data directory under the handle root.
 */
int sk_create(sk_handle_t* handle, const char* name, const char* dirs, unsigned int dim,
    const char* type, unsigned int range_size, const char* dist_func);

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
int sk_close(sk_handle_t* handle);

/*
 * Run KNN and return an allocated result array. The caller owns *ids_out and
 * must release it with sk_free(). count_out receives the number of ids.
 */
int sk_knn(sk_handle_t* handle, const char* vec, unsigned int k,
    uint64_t** ids_out, size_t* count_out);

/*
 * Run KNN for an in-memory float query vector with optional bitset filtering
 * and return allocated id/score arrays. The caller owns *ids_out and
 * *scores_out and must release both with sk_free(). If allowed_ids_blob is
 * nullptr and allowed_ids_blob_size is 0, no bitset filtering is applied.
 */
int sk_knn_vector_items(sk_handle_t* handle, const float* vec, uint64_t vec_size, unsigned int k,
    const void* allowed_ids_blob, size_t allowed_ids_blob_size,
    uint64_t** ids_out, double** scores_out, size_t* count_out);
   
/*
 * Run KNN with optional bitset filtering and return allocated id/score arrays.
 * The caller owns *ids_out and *scores_out and must release both with sk_free().
 * If allowed_ids_blob is nullptr and allowed_ids_blob_size is 0, no bitset
 * filtering is applied.
 */
int sk_knn_items(sk_handle_t* handle, const char* vec, unsigned int k,
    const void* allowed_ids_blob, size_t allowed_ids_blob_size,
    uint64_t** ids_out, double** scores_out, size_t* count_out);

/*
 * Return true when smaller score means better match for the currently open dataset.
 */
int sk_score_ascending_is_better(sk_handle_t* handle, bool* out);

/*
 * Merge delta files into data files.
 */
int sk_merge_delta(sk_handle_t* handle);

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
 * Generate test file with vectors and load them into the current dataset.
 */
int sk_generate_test_data(sk_handle_t* handle, 
    const char* path, uint64_t count, uint64_t start_id, bool binary);

/*
* Generate file with test metadata in CSV format.
*/
int sk_generate_test_metadata(sk_handle_t* handle, 
    const char* path, uint64_t count, uint64_t start_id);

/*
 * Load vectors from a text or binary input file into the current dataset.
 */
int sk_load_file(sk_handle_t* handle, const char* path);

/*
 * Persist an allowlist bitset blob for the currently open dataset.
 * The blob is stored in the first dataset directory as <name>.bitset.
 */
int sk_bitset_create(sk_handle_t* handle, const void* blob, size_t blob_size, const char* name);

/*
 * Remove a persisted allowlist bitset blob (<name>.bitset) for the currently
 * open dataset.
 */
int sk_bitset_drop(sk_handle_t* handle, const char* name);

/*
 * Load a persisted allowlist bitset blob (<name>.bitset) for the currently
 * open dataset. The caller owns *blob_out and must release it with sk_free().
 */
int sk_bitset_load(sk_handle_t* handle, const char* name, void** blob_out, size_t* blob_size_out);

/*
 * Build an in-memory allowlist bitset blob from an ordered array of ids.
 * Expectations:
 * - ids must point to count ids sorted in non-decreasing order
 * - duplicate ids are allowed
 * - ids may be nullptr only when count is 0
 * - blob_out and blob_size_out must be non-null
 * - count == 0 is valid and returns an empty blob
 * - out_of_memory and error_message_out are optional outputs
 * On success, the caller owns *blob_out and must release it with sk_free().
 */
int sk_bitset_build(
    uint64_t* ids, uint64_t count, void** blob_out, size_t* blob_size_out,
    bool* out_of_memory, const char** error_message_out);

/*
 * Internal-facing utility used by adapters (for example SQLite) to build an
 * in-memory allowlist bitset blob in the Sketch2 API binary format.
 */
int sk_bitset_builder_add(
    void** state, uint64_t id, bool* out_of_memory, const char** error_message_out);

/*
 * Finalize and release the bitset builder state. Must be called even if a
 * prior sk_bitset_builder_add() returned an error. On success, the caller
 * owns *blob_out and must release it with sk_free().
 */
int sk_bitset_builder_finish(
    void** state, void** blob_out, size_t* blob_size_out,
    bool* out_of_memory, const char** error_message_out);

/*
 * Print dataset file statistics to stdout or a text file.
 */
int sk_stats(sk_handle_t* handle, const char* path);

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

/*
 * Set global log level in Sketch2
 */
void sk_set_log_level(const char* log_level);

/*
 * Write Sketch2 version string into caller-provided buffer.
 */
void sk_version(char* buf, size_t buf_size);

/*
 * Write the compiled compute engine name into caller-provided buffer.
 */
void compute_engine(char* buf, int buf_size);

#ifdef __cplusplus
}
#endif

#endif // SKETCH2_H
