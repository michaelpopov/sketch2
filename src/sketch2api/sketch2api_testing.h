// Declares testing-only Sketch2 API helpers used by in-repo tests/extensions.

#ifndef SKETCH2API_TESTING_H
#define SKETCH2API_TESTING_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Test helper: returns active KNN engine name for the current process.
 * This declaration is intentionally kept out of sketch2.h so it does not
 * become part of the main public API surface.
 */
const char* sk_knn_engine_name_for_testing(void);

#ifdef __cplusplus
}
#endif

#endif // SKETCH2API_TESTING_H
