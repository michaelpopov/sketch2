# Test

This document explains how testing in Sketch2 is organized and how the test
suite is executed.

## Synthetic Test Data

Sketch2 includes built-in synthetic data generation, and the tests use it
actively.

- generated vectors can be produced in large volumes
- generated content is deterministic
- expected KNN results and storage outcomes remain predictable

This matters because many tests need controlled datasets that are large enough
to exercise range splitting, updates, merges, deletes, and multi-process
access without relying on opaque random fixtures.

## Running The Main Test Targets

The root `Makefile` exposes the usual entry points.

Run the release C++ suite:

```bash
make test
```

Run the debug C++ suite explicitly:

```bash
make test TYPE=dbg
```

Run the Python test suite:

```bash
make pytest
```

Run the default validation flow:

```bash
make validate
```

`make validate` combines the default C++ suite with the Python tests and
tutorial scripts.

## Testing Model

Sketch2 uses multiple layers of testing because the project spans storage,
the compute layer, process-wide runtime state, Python bindings, and SQLite
integration.

The main layers are:

- C++ unit tests for core libraries and APIs
- Python tests for the wrapper layer and end-to-end behavior
- SQLite tests for testing query functionality
- integration tests that exercise multi-step workflows, subprocess behavior,
  and reader/writer coordination
- sanitizer runs that execute the C++ suite under memory and UB checkers

This structure keeps low-level correctness checks close to the implementation
while also validating the system as a usable runtime library.

## C++ Unit Tests

The C++ test suite is built around GoogleTest and registered with `ctest`.
Tests are grouped by subsystem into separate executables:

- `utest_utils`
- `utest_stor`
- `utest_compute`
- `utest_sketch2`
- `utest_compute_sketch2api`
- `utest_vlite`

These binaries cover:

- utility code such as config parsing, file locks, logging, singleton state,
  and the update notifier
- storage components such as input generation, input parsing, data file
  layout, readers, writers, and dataset logic
- compute kernels and scanner logic
- the `sketch2api` C API, including compute-oriented API coverage
- the SQLite integration layer

There is also a dedicated thread-pool test binary:

- `utest_thread_pool`

It is intentionally kept out of the default `ctest` run because it adds
noticeable latency. It is available for focused runs when thread-pool behavior
is the thing being validated.

## Python And Integration Tests

The Python test tree lives under `src/pytest`, but the test runner is the
standard library `unittest` module, not `pytest`.

The Python layer serves two roles:

- wrapper/API tests for `sketch2_wrapper.py`
- integration tests that drive the shared library through realistic workflows

Representative coverage includes:

- basic dataset lifecycle and wrapper behavior
- error handling
- score-function behavior
- bulk load and incremental compaction
- delete-heavy workloads
- update-notifier visibility
- multi-range datasets
- concurrent readers and reader/writer interaction
- SQLite-facing integration behavior

Many of these tests use temporary directories and subprocesses so they can
exercise cross-process behavior instead of only in-process control flow.

## SQLite Integration Tests

SQLite integration is tested at two levels.

### C++ SQLite Tests

The C++ executable `utest_vlite` tests the SQLite extension layer directly.
These are unit-style integration tests: they create temporary datasets, load
`libsketch2.so` into an in-memory SQLite database, create a `vlite` virtual
table, and run SQL queries against it.

These tests validate:

- extension loading and virtual table creation
- KNN result ids and scores returned through SQL
- correct use of dataset-configured score functions such as `dot`, `l2`, and
  `cos`
- SQL constraint handling and error reporting
- SQLite-facing helper functionality such as the bitset-related interfaces

The important point is that these tests exercise the real SQLite extension
boundary rather than only calling lower-level C++ functions.

### Python SQLite Tests

The Python integration test `test_integ_vlite_interfaces.py` validates the
same SQLite surface from the higher-level runtime workflow used by the rest of
the system.

These tests use the Python `sqlite3` module to:

- open an in-memory SQLite connection
- load `libsketch2.so` as an extension
- create a `vlite` virtual table against a real Sketch2 dataset
- execute SQL queries that combine virtual-table behavior with the extension's
  helper functions

This layer checks SQL-visible behavior such as:

- `bitset_agg(id)` validation rules and unordered-id bitset filtering
- `allowed_ids` filtering when a pointer or aligned BLOB is supplied
- baseline behavior when `allowed_ids` is `NULL`
- rejection of invalid non-BLOB inputs

Together, the C++ and Python SQLite tests verify both the low-level extension
plumbing and the higher-level SQL workflow that users actually run.

## What The Tests Validate

At a high level, the suite checks these properties:

- persisted storage files are written and read correctly
- mutable state in pending updates behaves correctly
- merges preserve logical dataset contents
- compute kernels and scanner logic return correct nearest neighbors
- compiled runtime behavior stays correct across supported builds
- public APIs behave correctly through `Sketch2api`, Python, and SQLite
- reader/writer coordination and update notification work across processes

The suite is therefore not just checking isolated functions. It is validating
that Sketch2 works as a coherent storage-and-compute system.

## Notes

- Python integration tests rely on the wrapper and helper utilities in
  `src/pytest`
- some integration tests intentionally spawn subprocesses to validate
  cross-process behavior rather than only single-process execution
