// Declares testing-only Sketch2 API helpers used by in-repo tests/extensions.

#ifndef SKETCH2API_TESTING_H
#define SKETCH2API_TESTING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Test helper: returns active KNN engine name for the current process.
 * This declaration is intentionally kept out of sketch2.h so it does not
 * become part of the main public API surface.
 */
const char* sk_knn_engine_name_for_testing(void);
int sk_allowlist_storage_kind_for_testing(const void* ptr);
size_t sk_allowlist_temp_path_for_testing(const void* ptr, char* buf, size_t buf_size);

#ifdef __cplusplus
}
#endif

#endif // SKETCH2API_TESTING_H
