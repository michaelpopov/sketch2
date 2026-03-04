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

typedef struct sk_ret {
    sk_handle_t* handle;
    int code;
    char message[256];
} sk_ret_t;

/*
 *  sk_create creates data directory and a metadata file for a dataset.
 */
sk_ret_t sk_create(sk_dataset_metadata_t metadata);

/*
 *  sk_drop removes whole data directory including metadata and data fies for a dataset.
 */
sk_ret_t sk_drop(const char* dir);

/*
 *  sk_open creates a dataset instance and initializes it.
 */
sk_ret_t sk_open(const char *path);

/*
 *  sk_close deletes a dataset instance and closes temporary files.
 */
sk_ret_t sk_close(sk_handle_t* handle);

/*
 *  sk_add adds a line with a vector into input file for a dataset.
 */
sk_ret_t sk_add(sk_handle_t* handle, uint64_t id, const char *value);

/*
 *  sk_delete adds a line with a delete marker for an item into input file for a dataset.
 */
sk_ret_t sk_delete(sk_handle_t* handle, uint64_t id);

/*
 *  sk_load loads input file into a dataset.
 */
sk_ret_t sk_load(sk_handle_t* handle);

#ifdef __cplusplus
}
#endif

#endif

