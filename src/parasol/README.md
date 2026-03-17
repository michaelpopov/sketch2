# parasol

`parasol` is the C API layer for Sketch2 dataset creation, loading,
mutation, and query operations.

## Build Artifact Layout

`parasol` is now built as a static library and linked into the shared
`libsketch2.so` artifact.

For host applications, the runtime library to load/deploy is `libsketch2.so`.
`parasol` is not produced as a standalone shared library in the default build.

Typical release artifacts:

- `build/lib/libsketch2.so`

Typical debug artifacts:

- `build-dbg/lib/libsketch2.so`

## Startup Initialization

Sketch2 runtime initialization is explicit now.

The important entry point is:

```c
int sk_runtime_init(void);
```

This function applies process-wide runtime configuration once. It is intended
 to run before normal API usage such as `sk_connect()`.

Configuration sources and precedence:

1. built-in defaults
2. `SKETCH2_CONFIG` ini file, if present and readable
3. `SKETCH2_LOG_LEVEL`, overriding `log.level`
4. `SKETCH2_THREAD_POOL_SIZE`, overriding `thread_pool.size`
5. `SKETCH2_LOG_FILE`, selecting the log sink

If `SKETCH2_CONFIG` is missing, that is fine. Defaults and env overrides still
 work. If it is set but unreadable, startup logs a warning and continues with
 direct env overrides.

After the first successful initialization, the runtime is sealed:

- log level does not change through later startup config attempts
- thread-pool size does not change
- log sink does not change

This prevents process-wide behavior from mutating halfway through execution.

## Python Wrapper Behavior

The Python wrapper in `src/pytest/sketch2_wrapper.py` already calls
 `sk_runtime_init()` before `sk_connect()`, so normal demo/test usage gets the
 explicit initialization automatically.

If you are using the C API directly from another host, you should do the same:

```c
setenv("SKETCH2_LOG_LEVEL", "DEBUG", 1);
setenv("SKETCH2_THREAD_POOL_SIZE", "8", 1);

if (sk_runtime_init() != 0) {
    /* handle startup failure */
}

sk_handle_t* handle = sk_connect("/tmp/my_db");
```

## Thread Pool Notes

If `SKETCH2_THREAD_POOL_SIZE` or `thread_pool.size` is greater than `1`,
 `parasol` creates one shared thread pool for the process inside the Sketch2
 runtime.

That shared pool is used by storage code such as range-level dataset loading.
 `parasol` and `vlite` now run through the same `libsketch2.so`, so they see
 the same singleton and therefore the same thread pool.
