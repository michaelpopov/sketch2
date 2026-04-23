# Sketch2api

`Sketch2api` is the C API layer for Sketch2 dataset creation, loading,
mutation, and query operations.

## Build Artifact Layout

`Sketch2api` is now built as a static library and linked into the shared
`libsketch2.so` artifact.

For host applications, the runtime library to load/deploy is `libsketch2.so`.
`Sketch2api` is not produced as a standalone shared library in the default build.

Typical release artifacts:

- `build/lib/libsketch2.so`

Typical debug artifacts:

- `build-dbg/lib/libsketch2.so`

## Public C API Shape

The public header is `src/sketch2api/sketch2.h`.
Testing-only declarations live in `src/sketch2api/sketch2api_testing.h`.

The API follows a simple status-code pattern:

- functions return `0` on success and nonzero on failure
- detailed error state is stored on the handle
- callers can inspect failures through `sk_error()` and `sk_error_message()`

The main entry points are:

```c
sk_handle_t* sk_new_handle(const char* db_path);
void sk_release_handle(sk_handle_t* handle);

int sk_create(sk_handle_t* handle, const char* name, const char* dirs, unsigned int dim,
              const char* type, unsigned int range_size, const char* dist_func);
int sk_drop(sk_handle_t* handle, const char* name);
int sk_open(sk_handle_t* handle, const char* name);
int sk_close(sk_handle_t* handle);

int sk_knn(sk_handle_t* handle, const char* vec, unsigned int k,
           uint64_t** ids_out, size_t* count_out);
int sk_knn_items(sk_handle_t* handle, const char* vec, unsigned int k,
                 const void* allowed_ids_blob, size_t allowed_ids_blob_size,
                 uint64_t** ids_out, double** scores_out, size_t* count_out);
int sk_score_ascending_is_better(sk_handle_t* handle, bool* out);
int sk_get(sk_handle_t* handle, uint64_t id, char** value_out);
int sk_start_writing(sk_handle_t* handle);
int sk_write_vector(sk_handle_t* handle, uint64_t id, const char* data);
int sk_write_deleted(sk_handle_t* handle, uint64_t id);
int sk_abort_writing(sk_handle_t* handle);
int sk_complete_writing(sk_handle_t* handle);
int sk_bitset_create(sk_handle_t* handle, const void* blob, size_t blob_size, const char* name);
int sk_bitset_drop(sk_handle_t* handle, const char* name);
int sk_bitset_load(sk_handle_t* handle, const char* name, void** blob_out, size_t* blob_size_out);
int sk_bitset_builder_add(
    void** state, uint64_t id, bool* out_of_memory, const char** error_message_out);
int sk_bitset_builder_finish(
    void** state, void** blob_out, size_t* blob_size_out,
    bool* out_of_memory, const char** error_message_out);
void sk_free(void* ptr);
```

`sk_knn()` and `sk_get()` return allocated results through out-parameters. The
caller owns those returned buffers and must release them with `sk_free()`.

`sk_knn_items()` extends `sk_knn()` by returning scores and accepting an
optional allowlist bitset blob. The blob layout is documented in
`src/sketch2api/BITSET.md`.

For incremental ingest, the staged-writing API accumulates vectors and delete
markers into a temporary input file owned by the open dataset. Calling
`sk_complete_writing()` loads that accumulated input and removes the temporary
file. Calling `sk_abort_writing()` discards the staged session, removes the
temporary input file, and leaves the persisted dataset unchanged.

Example:

```c
uint64_t* ids = NULL;
size_t count = 0;
if (sk_knn(handle, "1.0, 2.0, 3.0, 4.0", 3, &ids, &count) != 0) {
    fprintf(stderr, "knn failed: %s\n", sk_error_message(handle));
} else {
    for (size_t i = 0; i < count; ++i) {
        printf("%llu\n", (unsigned long long)ids[i]);
    }
}
sk_free(ids);
```

Staged-write example:

```c
if (sk_start_writing(handle) != 0) {
    fprintf(stderr, "start_writing failed: %s\n", sk_error_message(handle));
}
if (sk_write_vector(handle, 10, "10.1, 10.1, 10.1, 10.1") != 0) {
    fprintf(stderr, "write_vector failed: %s\n", sk_error_message(handle));
}
if (sk_write_deleted(handle, 11) != 0) {
    fprintf(stderr, "write_deleted failed: %s\n", sk_error_message(handle));
}
if (sk_complete_writing(handle) != 0) {
    fprintf(stderr, "complete_writing failed: %s\n", sk_error_message(handle));
}
```

Abort example:

```c
if (sk_start_writing(handle) != 0) {
    fprintf(stderr, "start_writing failed: %s\n", sk_error_message(handle));
}
if (sk_write_vector(handle, 10, "10.1, 10.1, 10.1, 10.1") != 0) {
    fprintf(stderr, "write_vector failed: %s\n", sk_error_message(handle));
    sk_abort_writing(handle);
}
```

## Startup Initialization

Sketch2 runtime initialization happens automatically from `sk_new_handle()`.

That call applies process-wide runtime configuration before the handle is
created, so callers do not need a separate startup step.

Configuration sources and precedence:

1. built-in defaults
2. `SKETCH2_CONFIG` ini file, if present and readable
3. `SKETCH2_LOG_LEVEL`, overriding `log.level`
4. `SKETCH2_THREAD_POOL_SIZE`, overriding `thread_pool.size`
5. `SKETCH2_LOG_FILE`, selecting the log sink

If `SKETCH2_CONFIG` is missing, that is fine. Defaults and env overrides still
 work. If it is set but unreadable, startup logs a warning and continues with
 direct env overrides.

The native compute path is not part of runtime configuration. It is built into
the library.

After the first successful initialization, the runtime is sealed:

- log level does not change through later startup config attempts
- thread-pool size does not change
- log sink does not change

This prevents process-wide behavior from mutating halfway through execution.

## Python Wrapper Behavior

The Python wrapper in `src/pytest/sketch2_wrapper.py` relies on `sk_new_handle()`
to perform runtime initialization automatically.

If you are using the C API directly from another host, setting environment
variables before `sk_new_handle()` is enough:

```c
setenv("SKETCH2_LOG_LEVEL", "DEBUG", 1);
setenv("SKETCH2_THREAD_POOL_SIZE", "8", 1);

sk_handle_t* handle = sk_new_handle("/tmp/my_db");
```

If `sk_new_handle()` returns `NULL`, handle creation failed before a handle-local
error object existed, so the caller should treat that as a connection/setup
failure rather than trying to read `sk_error_message()`.

## Thread Pool Notes

If `SKETCH2_THREAD_POOL_SIZE` or `thread_pool.size` is greater than `1`,
 `Sketch2api` creates one shared thread pool for the process inside the Sketch2
 runtime.

That shared pool is used by storage code such as range-level dataset loading.
 `Sketch2api` and `vlite` now run through the same `libsketch2.so`, so they see
 the same singleton and therefore the same thread pool.
