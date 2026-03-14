# parasol

`parasol` is the shared C API layer for Sketch2 dataset creation, loading,
mutation, and query operations.

## Runtime Library Dependency

`libparasol.so` now depends on `libutils.so`.

This is required because process-wide runtime state is centralized in the shared
 utilities library:

- global log level
- configured log file sink
- singleton-owned thread pool
- one-time runtime configuration state

Without linking `libutils.so` dynamically, each shared library would get its own
 copy of that state. In practice that caused duplicated singleton startup,
 duplicated `Started thread pool...` logs, separate log-level state, and more
 than one thread pool inside the same Python process.

So when you load or deploy `libparasol.so`, make sure `libutils.so` is
 available too, typically in the same runtime library directory.

Typical release artifacts:

- `build/lib/libparasol.so`
- `build/lib/libutils.so`

Typical debug artifacts:

- `build-dbg/lib/libparasol.so`
- `build-dbg/lib/libutils.so`

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

The Python wrapper in `src/pytest/parasol_wrapper.py` already calls
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
 `parasol` creates one shared thread pool for the process through `libutils.so`.

That shared pool is used by storage code such as range-level dataset loading.
 The point of the shared `libutils.so` dependency is that `parasol` and `vlite`
 see the same singleton and therefore the same thread pool.
