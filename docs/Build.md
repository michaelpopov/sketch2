# Build

This document is focused on building Sketch2 and running its main entry
points.

## Platform And Toolchain

Sketch2 is currently a Linux-only project.

Minimum build requirements:

- CMake 3.20 or newer
- a C++20 compiler
- GCC or Clang on x86/x86_64 and AArch64 for the Google Highway and NumKong multi-target build

Practical tools used by this repository:

- `cmake`
- `ninja-build`
- `make`
- `python3` for the Python wrapper, shell, and demo scripts

A fresh machine also needs normal outbound network access during the first
configure because GoogleTest is downloaded as part of the standard build.

On Ubuntu, the repository `Makefile` provides:

```bash
make prepare
```

That target runs:

```bash
sudo apt update && sudo apt install -y build-essential cmake ninja-build
```

If `python3` is not already installed on the machine, install it separately.

## Build Model

The project uses CMake as the primary build system. The root `Makefile` is a
thin convenience layer over the main CMake commands.

Build directories are engine-specific because `SKETCH2_COMPUTE_ENGINE` is cached at
configure time. If you want a NumKong build, configure a fresh build directory
for it instead of reusing a previously configured `highway` tree.

The default compute engine is `highway`. Set `-DSKETCH2_COMPUTE_ENGINE=numkong` at
configure time if you want the NumKong-backed build instead.

Important build types:

- `Debug`
- `Release`

If no build type is provided, the top-level CMake configuration defaults to
`Debug`.

On x86/x86_64, Highway builds can compile higher-ISA targets through these
options:

- `SKETCH_ENABLE_AVX2`
- `SKETCH_ENABLE_AVX512F`
- `SKETCH_ENABLE_AVX512VNNI`

All three are enabled by default. On AArch64, the build enables
`-march=armv8.2-a+fp16+dotprod`. `+dotprod` is in the baseline because every
server target we ship to (Apple Silicon, Ampere Altra/AmpereOne, Graviton 2/3/4)
supports SDOT/UDOT, and the integer kernels materially benefit from it.

SVE is intentionally not enabled by default on AArch64 because many `arm64`
systems, including Apple Silicon, do not support it. If you are building for an
SVE-capable Linux target such as Graviton 3/3E/4 or AmpereOne and want the
extra kernels, configure with:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSKETCH_ENABLE_ARM_SVE=ON
```

Or from the Makefile:

```bash
make build-arm-sve
```

For Neoverse-specific instruction scheduling, set `SKETCH_ARM_MCPU` to the
target core. Common values are `neoverse-n1` (Graviton2, Ampere Altra),
`neoverse-v1` (Graviton3/3E), `neoverse-v2` (Graviton4), or `native` when
building on the deployment host:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSKETCH_ENABLE_ARM_SVE=ON -DSKETCH_ARM_MCPU=neoverse-v1
```

The Makefile exposes the same knobs as `ARM_SVE=1` and `ARM_MCPU=<tune>`:

```bash
make build-arm-sve ARM_MCPU=neoverse-v1
```

To build the NumKong engine explicitly from a fresh clone or on another
machine, use a dedicated build directory:

```bash
cmake -S . -B build-nk-dbg -G Ninja -DCMAKE_BUILD_TYPE=Debug -DSKETCH2_COMPUTE_ENGINE=numkong
cmake --build build-nk-dbg
ctest --test-dir build-nk-dbg --output-on-failure
```

## Main Artifacts

The main runtime artifact is:

- `libsketch2.so`

This shared library contains the Sketch2 runtime, the `Sketch2api` C API, and the
SQLite `vlite` extension entry points.

Typical output layout:

- highway release runtime binaries: `bin-hwy/`
- highway debug runtime binaries: `bin-dbg-hwy/`
- numkong release runtime binaries: `bin-nk/`
- numkong debug runtime binaries: `bin-dbg-nk/`
- release shared libraries: `build/lib/`
- debug shared libraries: `build-dbg/lib/`

The most important library paths are therefore:

- release: `build/lib/libsketch2.so`
- debug: `build-dbg/lib/libsketch2.so`

## Building With Make

The simplest entry points are the root `Makefile` targets.

Build debug:

```bash
make
```

Build release:

```bash
make rel
```

These commands configure the corresponding build directory if needed and then
run `cmake --build` with parallel jobs.

Install the release artifacts for reuse by other projects:

```bash
make install-hwy
```

For the NumKong release artifacts, use:

```bash
make install-nk
```

`install-hwy` depends on `rtest`, while `install-nk` depends on `rtest-nk`.
Each target first builds and tests its release configuration, then creates an
engine-specific install tree in the repository root and copies the public C
header plus the release shared library and its Python wrapper into it.

Installed layout:

- `install-hwy/include/sketch2.h`
- `install-hwy/bin/libsketch2.so`
- `install-hwy/bin/sketch2_wrapper.py`
- `install-nk/include/sketch2.h`
- `install-nk/bin/libsketch2.so`
- `install-nk/bin/sketch2_wrapper.py`

The `install-hwy/` and `install-nk/` directories are meant to hold the files
that consumers need without having to know the repository's internal build
directories.

Examples:

- a Python integration script that needs a stable directory for `SKETCH2_LIB`
- the tutorial harness and similar local demos that import `sketch2_wrapper.py`
- a C or C++ consumer that includes `sketch2.h` and links against `libsketch2.so`

## Building With CMake Directly

If you want to work without the `Makefile`, use CMake directly.
The repository-standard build directories (`build/`, `build-dbg/`, `build-nk/`,
`build-nk-dbg/`) are intended to be Ninja trees.

Debug:

```bash
cmake -S . -B build-dbg -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build-dbg
```

Release:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

`bench_compute` is built as part of the normal native build. To build it
explicitly in a release tree, use:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target bench_compute
```

## Using Sketch2 From Another Project

When Sketch2 is checked out locally, it is convenient to point dependent
projects at the repository root through `SKETCH2_ROOT`.

Example:

```bash
export SKETCH2_ROOT=/absolute/path/to/sketch2
```

If you are already in the repository root, you can also set it as:

```bash
export SKETCH2_ROOT=$(pwd)
```

After `make install-hwy` or `make install-nk`, the reusable install inputs are:

- `$SKETCH2_ROOT/install-hwy/include`
- `$SKETCH2_ROOT/install-hwy/bin`
- `$SKETCH2_ROOT/install-nk/include`
- `$SKETCH2_ROOT/install-nk/bin`

Example for Python tooling:

```bash
export SKETCH2_LIB="$SKETCH2_ROOT/install-hwy/bin"
python3 tutorial/run_all.py
```

For C or C++ consumers built from a checked-out Sketch2 tree, use the public
header from the install tree:

```bash
g++ -std=c++20 app.cpp \
  -I"$SKETCH2_ROOT/install-hwy/include" \
  -L"$SKETCH2_ROOT/install-hwy/bin" \
  -lsketch2
```

This keeps Python-facing consumers pointed at a stable runtime directory instead
of the build-specific directories such as `build/lib` or `bin-hwy`.

## Using The Shared Library During Dev And Test Runs

If a development binary or test binary loads `libsketch2.so` at runtime, add
the installed library directory to `LD_LIBRARY_PATH`.

Example:

```bash
export LD_LIBRARY_PATH="$SKETCH2_ROOT/install-hwy/bin:${LD_LIBRARY_PATH}"
```

With that in place, locally built tools, integration tests, or other dependent
executables can locate `libsketch2.so` during development and test runs.

For a one-off command, you can also set it inline:

```bash
LD_LIBRARY_PATH="$SKETCH2_ROOT/install-hwy/bin:${LD_LIBRARY_PATH}" ./my_test_binary
```

## Running The Python Entry Points

The Python wrapper lives in `src/pytest/sketch2_wrapper.py`. It loads
`libsketch2.so` through `ctypes` and searches the standard build locations in
this order:

1. `bin-dbg-hwy/libsketch2.so`
2. `bin-hwy/libsketch2.so`
3. `bin-dbg-nk/libsketch2.so`
4. `bin-nk/libsketch2.so`

Run the interactive Python shell helper:
The search is temporary for the current development stage. It will be fixed
in the future.

```bash
python3 src/pytest/shell.py --db-root /tmp/sketch2_db
```

## Running The SQLite Entry Point

Sketch2 also ships a SQLite virtual table extension through `libsketch2.so`.
When using SQLite, load the shared library and then create a virtual table that
points at an existing Sketch2 dataset ini file.

Example:

```sql
.load /absolute/path/to/build/lib/libsketch2.so
CREATE VIRTUAL TABLE nn USING vlite('/absolute/path/to/dataset.ini');
```

The repository also builds a SQLite shell binary in the runtime output
directory. In a debug build that binary is typically:

- `bin-dbg-hwy/sqlite3`

## Runtime Configuration

Sketch2 runtime initialization is process-wide. The main environment variables
for running the system are:

- `SKETCH2_CONFIG`
- `SKETCH2_LOG_LEVEL`
- `SKETCH2_THREAD_POOL_SIZE`
- `SKETCH2_LOG_FILE`

These variables should be set before starting a Python process that loads
`libsketch2.so` or before loading the SQLite extension.

Example:

```bash
export SKETCH2_LOG_LEVEL=DEBUG
export SKETCH2_THREAD_POOL_SIZE=8
export SKETCH2_LOG_FILE=/tmp/sketch2.log
```

## Notes

- `Release` uses the compiler's standard CMake release flags, which typically
  means optimized code with `-DNDEBUG`
- each configured build contains one top-level compute engine selected by
  `SKETCH2_COMPUTE_ENGINE`
- Highway builds can still contain multiple ISA targets inside that engine,
  while NumKong builds keep their own internal capability-specific dispatch
