# Build

This document is focused on building Sketch2 and running its main entry
points.

## Platform And Toolchain

Sketch2 is currently a Linux-only project.

Minimum build requirements:

- CMake 3.20 or newer
- a C++20 compiler
- GCC or Clang on x86/x86_64 if AVX2 or AVX-512 runtime backends are enabled
- AArch64 is supported with the architecture options needed for NEON and fp16

Practical tools used by this repository:

- `cmake`
- `ninja-build`
- `make`
- `python3` for the Python wrapper, shell, and demo scripts

On Ubuntu, the repository `Makefile` provides:

```bash
make install
```

That target runs:

```bash
sudo apt update && sudo apt install -y build-essential cmake ninja-build
```

If `python3` is not already installed on the machine, install it separately.

## Build Model

The project uses CMake as the primary build system. The root `Makefile` is a
thin convenience layer over the main CMake commands.

Important build types:

- `Debug`
- `Release`
- `Sanitizer`

If no build type is provided, the top-level CMake configuration defaults to
`Debug`.

On x86/x86_64, the build enables runtime-dispatched SIMD backends through these
options:

- `SKETCH_ENABLE_AVX2`
- `SKETCH_ENABLE_AVX512F`
- `SKETCH_ENABLE_AVX512VNNI`

All three are enabled by default. On AArch64, the build enables
`-march=armv8.2-a+fp16`.

## Main Artifacts

The main runtime artifact is:

- `libsketch2.so`

This shared library contains the Sketch2 runtime, the `parasol` C API, and the
SQLite `vlite` extension entry points.

Typical output layout:

- release runtime binaries: `bin/`
- debug runtime binaries: `bin-dbg/`
- sanitizer runtime binaries: `bin-san/`
- release shared libraries: `build/lib/`
- debug shared libraries: `build-dbg/lib/`
- sanitizer shared libraries: `build-san/lib/`

The most important library paths are therefore:

- release: `build/lib/libsketch2.so`
- debug: `build-dbg/lib/libsketch2.so`
- sanitizer: `build-san/lib/libsketch2.so`

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

Build sanitizer:

```bash
make san
```

These commands configure the corresponding build directory if needed and then
run `cmake --build` with parallel jobs.

## Building With CMake Directly

If you want to work without the `Makefile`, use CMake directly.

Debug:

```bash
cmake -S . -B build-dbg -DCMAKE_BUILD_TYPE=Debug
cmake --build build-dbg
```

Release:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Sanitizer:

```bash
cmake -S . -B build-san -DCMAKE_BUILD_TYPE=Sanitizer
cmake --build build-san
```

If you want benchmark binaries as part of a release build, configure with:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSKETCH_ENABLE_BENCHMARKS=ON
cmake --build build --target bench_comp gbench_comp
```

## Running The Python Entry Points

The Python wrapper lives in `src/pytest/sketch2_wrapper.py`. It loads
`libsketch2.so` through `ctypes` and searches the standard build locations in
this order:

1. `bin/libsketch2.so`
2. `bin-dbg/libsketch2.so`

Run the interactive Python shell helper:
The search is temporary for the current development stage. It will be fixed
in the future.

```bash
make pyshell
```

Or run it directly:

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

- `bin-dbg/sqlite3`

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
- `Sanitizer` builds use AddressSanitizer, UndefinedBehaviorSanitizer, and LeakSanitizer
- on x86, higher SIMD backends are compiled into one binary and selected at
  runtime on supported CPUs
