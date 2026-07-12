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
 * nullptr and allowed_ids_blob_size is 0, no filtering is applied. Otherwise
 * allowed_ids_blob must point to a 32-byte-aligned serialized chunked-Roaring
 * bitset filter BLOB.
 */
int sk_knn_vector_items(sk_handle_t* handle, const float* vec, uint64_t vec_size, unsigned int k,
    const void* allowed_ids_blob, size_t allowed_ids_blob_size,
    uint64_t** ids_out, double** scores_out, size_t* count_out);
   
/*
 * Run KNN with optional bitset filtering and return allocated id/score arrays.
 * The caller owns *ids_out and *scores_out and must release both with sk_free().
 * If allowed_ids_blob is nullptr and allowed_ids_blob_size is 0, no filtering
 * is applied. Otherwise allowed_ids_blob must point to a 32-byte-aligned
 * serialized chunked-Roaring bitset filter BLOB.
 */
int sk_knn_items(sk_handle_t* handle, const char* vec, unsigned int k,
    const void* allowed_ids_blob, size_t allowed_ids_blob_size,
    uint64_t** ids_out, double** scores_out, size_t* count_out);

/*
 * Run KNN with an opaque API-owned bitset filter object produced by
 * sk_bitset_finish(). This is intended for in-process adapters such
 * as SQLite that need to pass a bitset filter without copying it through their own
 * BLOB storage.
 */
int sk_knn_items_bitset_filter(sk_handle_t* handle, const char* vec, unsigned int k,
    const void* allowed_ids, uint64_t** ids_out, double** scores_out, size_t* count_out);
/*
 * Run KNN for an in-memory float query vector with an opaque API-owned bitset
 * filter object produced by sk_bitset_finish(). This mirrors
 * sk_knn_items_bitset_filter(), but accepts vec/vec_size instead of a text
 * query string. If allowed_ids is nullptr, no filtering is applied.
 */
int sk_knn_vector_items_bitset_filter(sk_handle_t* handle, const float* vec, uint64_t vec_size,
    unsigned int k, const void* allowed_ids,
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
 * Generate test file with vectors using a named pattern and load them into the
 * current dataset. Pass NULL or "auto" to select the metric-default pattern.
 */
int sk_generate_test_data(sk_handle_t* handle, 
    const char* path, uint64_t count, uint64_t start_id, const char* pattern, bool binary);

/*
* Generate file with test metadata in CSV format.
*/
int sk_generate_test_metadata(sk_handle_t* handle, 
    const char* path, uint64_t count, uint64_t start_id);

/*
 * Load vectors from a text or binary input file into the current dataset.
 */
int sk_import_data(sk_handle_t* handle, const char* path);

/*
 * Print dataset file statistics to stdout or a text file.
 */
int sk_stats(sk_handle_t* handle, const char* path);

/*
 * Verify payload checksums for all persisted data and delta files in the
 * currently open dataset. This may read every byte of every data/delta file.
 */
int sk_verify_integrity(sk_handle_t* handle);

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
 * Build an API-owned serialized chunked-Roaring bitset filter object. The finished
 * object is opaque; pass it to sk_knn_items_bitset_filter() or
 * sk_knn_vector_items_bitset_filter() and delete it with sk_bitset_delete().
 */
int sk_bitset_add_id(
    void** state, uint64_t id, bool* out_of_memory, const char** error_message_out,
    const char* name);
int sk_bitset_add_id_name(
    void** state, uint64_t id, bool* out_of_memory, const char** error_message_out);
/*
 * Append `size` ids to an API-owned bitset filter builder. *state must point
 * to a builder previously created by sk_bitset_create_builder() or another
 * sk_bitset_* call that initializes it; this function does not auto-create a
 * builder. size must be greater than zero and ids must be non-NULL.
 *
 * `ids` must be sorted in non-decreasing order. The builder validates this
 * and rejects unsorted input. The sorted contract lets the implementation
 * group ids by chunk in a single forward scan and feed CRoaring's range-
 * coalescing bulk-add path directly, which is significantly faster than
 * repeated single-id sk_bitset_add_id_name() calls -- especially for dense
 * runs of consecutive ids. Duplicates are tolerated (set semantics).
 *
 * On failure -1 is returned; any ids inserted before a failing one remain in
 * the builder, so the caller should treat the builder as poisoned and either
 * delete *state or call sk_bitset_finish() without trusting the contents.
 */
int sk_bitset_add_multiple_ids_name(
    void** state, const uint64_t* ids, uint64_t size,
    bool* out_of_memory, const char** error_message_out);
int sk_bitset_create_builder(
    void** state, bool* out_of_memory, const char** error_message_out, const char* name);
int sk_bitset_finish(
    void** state, void** out, bool* out_of_memory, const char** error_message_out);
int sk_bitset_load(
    const char* name, void** out, bool* out_of_memory, const char** error_message_out);
/*
 * Delete an opaque bitset filter object returned by sk_bitset_finish() or
 * sk_bitset_load().
 */
void sk_bitset_delete(void* ptr);
int sk_bitset_drop(
    const char* name, int* removed_out, bool* out_of_memory, const char** error_message_out);

/*
 * Cache management. The cache keeps named bitset filter files open and mapped
 * after creation or load. These entry points evict cached entries; the
 * underlying on-disk files are untouched. Use sk_bitset_drop() to remove
 * a file from disk.
 */
int sk_bitset_cache_remove(
    const char* name, int* removed_out, bool* out_of_memory, const char** error_message_out);
int sk_bitset_cache_clear(
    bool* out_of_memory, const char** error_message_out);

/*
 * Text embedding support (backed by llama.cpp, CPU only).
 *
 * An embedder wraps one loaded GGUF embedding model; pooling behavior comes
 * from model metadata unless overridden. Embedders are thread-safe: calls on
 * one handle are serialized internally and error state is per-thread. Share
 * one embedder per process (a single call already saturates the compute pool)
 * and batch with sk_embed_texts() for throughput; each additional open loads
 * the model again and creates another worker pool.
 */
typedef struct sk_embedder sk_embedder_t;

/*
 * Load a GGUF embedding model from model_path. options is NULL/"" for
 * defaults, or a comma-separated key=value list:
 *   pooling=mean|cls|last       override pooling (default: model metadata)
 *   normalize=0|1               L2-normalize output vectors (default: 1)
 *   truncate=0|1                truncate over-long inputs (default: 0 = error)
 *   context=N                   max tokens per input (default: min of model
 *                               training context and 4096)
 *   threads=N                   compute threads (default: CPUs available to
 *                               the process; clamped to that budget and to
 *                               what the OS can grant, e.g. under container
 *                               pid limits). The embedder owns a resident
 *                               pool; nothing is spawned per call
 * Returns NULL on failure; if error_message_out is non-NULL it receives a
 * pointer to a thread-local message valid until the calling thread's next
 * sk_embedder_* or sk_embed_* call.
 *
 * CPU requirement (x86-64): default builds compile the inference kernels for
 * AVX2/FMA (Intel Haswell 2013+, AMD Zen 2017+). On CPUs or hypervisor CPU
 * models lacking a required extension, open fails with a clear error instead
 * of SIGILL. Other libsketch2 APIs keep the library's baseline requirements.
 */
sk_embedder_t* sk_embedder_open(const char* model_path, const char* options,
    const char** error_message_out);

/*
 * Release the embedder. Must not run concurrently with other calls on the
 * same handle.
 */
void sk_embedder_close(sk_embedder_t* embedder);

/*
 * Return the dimension of vectors produced by this embedder.
 */
int sk_embedder_dim(sk_embedder_t* embedder, unsigned int* dim_out);

/*
 * Return the maximum number of tokens accepted per input text.
 */
int sk_embedder_max_tokens(sk_embedder_t* embedder, unsigned int* max_tokens_out);

/*
 * Generate the embedding for one text piece. On success *vec_out receives an
 * allocated array of *dim_out floats; the caller owns it and must release it
 * with sk_free().
 */
int sk_embed_text(sk_embedder_t* embedder, const char* text,
    float** vec_out, size_t* dim_out);

/*
 * Generate embeddings for count text pieces, packing inputs into shared
 * forward passes — much faster than calling sk_embed_text() in a loop. On
 * success *vecs_out receives a row-major array of count * *dim_out floats
 * (row i = texts[i]); the caller frees it with sk_free(). On failure nothing
 * is allocated and the error message reports the failing input index.
 */
int sk_embed_texts(sk_embedder_t* embedder, const char** texts, size_t count,
    float** vecs_out, size_t* dim_out);

/*
 * Error code of the calling thread's most recent embedder call (0 success,
 * -1 failure). Error state is thread-local, errno-style.
 */
int sk_embedder_error(sk_embedder_t* embedder);

/*
 * Error message of the calling thread's most recent failed embedder call
 * ("" if it succeeded); valid until the thread's next embedder call.
 */
const char* sk_embedder_error_message(sk_embedder_t* embedder);

/*
 * Set global log level in Sketch2
 */
void sk_set_log_level(const char* log_level);

/*
 * Write Sketch2 version string into caller-provided buffer.
 */
void sk_version(char* buf, size_t buf_size);


#ifdef __cplusplus
}
#endif

#endif // SKETCH2_H
