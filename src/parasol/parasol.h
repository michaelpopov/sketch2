#ifndef PARASOL_H
#define PARASOL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sk_dataset_metadata {
    char dir[256];
    char type[16];
    unsigned int dim;
    unsigned int range_size;
    unsigned int data_merge_ratio;
} sk_dataset_metadata_t;

typedef struct sk_handle sk_handle_t;

/*
 *  Initialize sk_handle for further usage.
 */
sk_handle_t* connect();

/*
 *  Release resources associated with sk_handle.
 */
void disconnect(sk_handle_t* handle);

/*
 *  sk_create creates data directory and a metadata file for a dataset.
 */
int sk_create(sk_handle_t* handle, sk_dataset_metadata_t metadata);

/*
 *  sk_drop removes whole data directory including metadata and data fies for a dataset.
 */
int sk_drop(sk_handle_t* handle);

/*
 *  sk_open creates a dataset instance and initializes it.
 */
int sk_open(sk_handle_t* handle, const char *path);

/*
 *  sk_close deletes a dataset instance and closes temporary files.
 */
int sk_close(sk_handle_t* handle);

/*
 *  sk_add adds a line with a vector into input file for a dataset.
 */
int sk_add(sk_handle_t* handle, uint64_t id, const char *value);

/*
 *  sk_delete adds a line with a delete marker for an item into input file for a dataset.
 */
int sk_delete(sk_handle_t* handle, uint64_t id);

/*
 *  sk_load loads input file into a dataset.
 */
int sk_load(sk_handle_t* handle);

/*
 *  sk_knn finds K nearest neighbors
 */
int sk_knn(sk_handle_t* handle, const char* vec, uint64_t* ids, uint64_t* ids_count);

/*
 *  sk_get returns a text content of a vector found by id.
 *  If vector is not found, function returns -1.
 */
int sk_get(sk_handle_t* handle, uint64_t id, char* buf, uint64_t buf_size);

/*
 *  sk_error returns error code registered during previous call.
 */
int sk_error(sk_handle_t* handle);

/*
 *  sk_error_message returns error message registered during previous call.
 */
const char* sk_error_message(sk_handle_t* handle);

#ifdef __cplusplus
}
#endif

#endif

