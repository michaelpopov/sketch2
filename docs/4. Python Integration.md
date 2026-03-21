# Python Integration

This document explains how Python integration in Sketch2 is implemented, what
functionality it supports, and how to use the Python wrapper API.

## Overview

Sketch2 provides Python integration through a thin wrapper in:

- `src/pytest/sketch2_wrapper.py`

The wrapper does not implement storage or query logic in Python. Instead, it
loads the native Sketch2 shared library, `libsketch2.so`, and forwards Python
calls to the `parasol` C API.

That design keeps the Python layer small while reusing the same native runtime,
storage engine, compute kernels, and process-wide configuration used by other
integration surfaces.

## How Python Integration Works

The Python wrapper is implemented with `ctypes`.

At startup, the `Sketch2` class:

1. locates `libsketch2.so`
2. loads it with `ctypes.CDLL(...)`
3. configures the C function signatures
4. calls `sk_runtime_init()`
5. opens a Sketch2 handle with `sk_connect()`

After that, Python methods such as `create()`, `upsert()`, `knn()`, and
`merge_accumulator()` call directly into the shared library.

This means Python is an integration surface for the native engine, not a
separate implementation.

## Shared Library Lookup

If `lib_path` is not passed explicitly, the wrapper searches the standard build
locations in this order:

1. `bin/libsketch2.so`
2. `bin-dbg/libsketch2.so`

If no file is found, wrapper construction fails with `FileNotFoundError`.

You can also pass an explicit library path:

```python
from sketch2_wrapper import Sketch2

with Sketch2("/tmp/my_db", lib_path="build/lib/libsketch2.so") as sk:
    ...
```

## Runtime Configuration

The Python wrapper uses the same process-wide runtime initialization as the C
API and SQLite integration.

Important environment variables:

- `SKETCH2_CONFIG`
- `SKETCH2_LOG_LEVEL`
- `SKETCH2_THREAD_POOL_SIZE`
- `SKETCH2_LOG_FILE`

These should be set before the Python process creates a `Sketch2` object.

Example:

```bash
export SKETCH2_LOG_LEVEL=DEBUG
export SKETCH2_THREAD_POOL_SIZE=8
python3 my_script.py
```

## Supported Functionality

The Python wrapper supports the main dataset lifecycle and query operations.

Supported capabilities include:

- connecting to a Sketch2 database root
- creating, opening, closing, and dropping datasets
- inserting and deleting vectors
- bulk loading from generated data or input files
- flushing the accumulator and merging delta files
- running KNN queries
- fetching stored vectors
- printing and stats diagnostics

The wrapper is used by demos and integration tests, but it is also a practical
scripting surface for local workflows and automation.

## Basic Usage

Example:

```python
from sketch2_wrapper import Sketch2

with Sketch2("/tmp/my_workspace") as sk:
    sk.create("items", type_name="f32", dim=4, range_size=1000, dist_func="l2")

    sk.upsert(100, "0.0, 0.0, 0.0, 0.0")
    sk.upsert(101, "1.0, 1.0, 1.0, 1.0")
    sk.upsert(102, "2.0, 2.0, 2.0, 2.0")
    sk.merge_accumulator()

    ids = sk.knn("1.1, 1.1, 1.1, 1.1", 2)
    print(ids)

    vec = sk.get(ids[0])
    print(vec)

    sk.close("items")
```

## Wrapper API Reference

### `Sketch2(db_path, lib_path=None)`

Creates a wrapper instance bound to a Sketch2 database root directory.

Arguments:

- `db_path`: path to the Sketch2 database root
- `lib_path`: optional explicit path to `libsketch2.so`

Behavior:

- loads the native shared library
- initializes the Sketch2 runtime
- connects to the database root

The class supports context-manager usage through `with Sketch2(...) as sk:`.

### `connect`

There is no separate Python `connect()` method in the wrapper.

Connection happens inside `Sketch2(db_path, lib_path=None)`:

- the wrapper loads `libsketch2.so`
- initializes the runtime with `sk_runtime_init()`
- connects to the database root with `sk_connect()`

So, in Python, object construction is the connect step.

### `close_handle()`

Closes any tracked open datasets on the current handle and disconnects the
underlying native handle.

This is also called automatically when the context manager exits.

### `error()`

Returns the current numeric error code from the native handle.

### `error_message()`

Returns the current error message string from the native handle.

### `create(name, type_name="f32", dim=4, range_size=1000, dist_func="l1")`

Creates a dataset and opens it on the current handle.

Arguments:

- `name`: dataset name
- `type_name`: vector element type, such as `f32`, `f16`, or `i16`
- `dim`: vector dimension
- `range_size`: id-range partition size
- `dist_func`: distance function, such as `l1`, `l2`, or `cos`

### `drop(name)`

Drops a dataset and removes its persisted files.

### `open(name)`

Opens an existing dataset on the current handle.

### `close(name)`

Closes an open dataset on the current handle.

### `upsert(item_id, value)`

Inserts or replaces one vector by id.

Arguments:

- `item_id`: vector id
- `value`: vector encoded as text

Example:

```python
sk.upsert(42, "1.0, 2.0, 3.0, 4.0")
```

### `ups2(item_id, value)`

Convenience insertion helper that constructs a vector by repeating one scalar
value across the dataset dimension.

Example:

```python
sk.ups2(42, 1.5)
```

### `delete(item_id)`

Deletes one vector by id.

### `merge_accumulator()`

Flushes the current mutable accumulator state into persisted storage.

### `merge_delta()`

Merges persisted delta content back into compact data files.

### `knn(vec, count) -> list[int]`

Runs a K-nearest-neighbor query and returns result ids.

Arguments:

- `vec`: query vector encoded as text
- `count`: requested number of neighbors

Returns:

- list of result ids in nearest-first order

Example:

```python
ids = sk.knn("1.0, 2.0, 3.0, 4.0", 5)
```

### `get(item_id) -> str`

Fetches one stored vector and returns its text representation.

### `print()`

Prints dataset vectors through the native runtime.

This is mainly a diagnostic helper.

### `generate(count, start_id, pattern)`

Generates synthetic text-form input data and loads it into the current dataset.

This is used heavily by tests and demos because the generated content is
deterministic and scales to large volumes.

### `generate_bin(count, start_id, pattern)`

Generates synthetic binary-form input data and loads it into the current
dataset.

This is useful for larger demo and performance-oriented flows.

### `load_file(path)`

Loads vectors from an external input file into the current dataset.

The input file can contain either synthetically generated data or a prepared
set of vectors from an external pipeline. This is the primary bulk-ingest entry
point for production-style workflows where vector data is prepared outside the
wrapper and then loaded into Sketch2.

### `stats()`

Prints storage statistics through the native runtime.

This is mainly a diagnostic and inspection helper.

## Error Handling

The wrapper raises `Sketch2Error` when a native API call returns failure.

`Sketch2Error` includes:

- `operation`: name of the failed native operation
- `code`: numeric error code
- `message`: native error message

This keeps Python code straightforward:

```python
from sketch2_wrapper import Sketch2, Sketch2Error

try:
    with Sketch2("/tmp/my_db") as sk:
        sk.open("missing_dataset")
except Sketch2Error as e:
    print(e.operation, e.code, e.message)
```

## Related Files

Other Python-side files in `src/pytest`:

- `shell.py`: interactive helper shell
- `demo.py`: Python demo and demo workload driver
- `integ_helpers.py`: shared helpers used by integration tests
- `test_*.py`: wrapper and integration tests

## Notes

- the wrapper is intentionally thin and mirrors the native runtime surface
- Python is used for scripting, demos, and integration testing, not for
  reimplementing the storage engine
- the wrapper depends on a prebuilt `libsketch2.so`, so the native project must
  be built before Python code can use it
