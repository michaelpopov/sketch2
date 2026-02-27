# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

All commands run from the project root (directory containing `CMakeLists.txt`).

**Debug build (preferred for development):**
```sh
cmake -S . -B build-dbg -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=g++
cmake --build build-dbg --parallel $(nproc)
```

**Release build:**
```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build build --parallel $(nproc)
```

**Build a specific target:**
```sh
cmake --build build-dbg --parallel $(nproc) --target utest_stor
```

**Discover all targets:**
```sh
cmake --build build-dbg --target help
```

## Running Tests

**Run all unit tests:**
```sh
ctest --test-dir build-dbg --output-on-failure
```

**Run a single test suite by name:**
```sh
ctest --test-dir build-dbg -R utest_stor --output-on-failure
```

**Build and run a specific unit test binary directly:**
```sh
cmake --build build-dbg --parallel $(nproc) --target utest_stor && ./bin-dbg/utest_stor
```

Unit test binaries (Debug): `bin-dbg/utest_stor`, `bin-dbg/utest_membuf`, `bin-dbg/utest_comp`, `bin-dbg/utest_cntrl`

## Architecture

Sketch2 is a vector database with a client/server architecture. The project is in early development (Stage 1 focuses on the storage layer).

### Executables

| Target | Output | Purpose |
|--------|--------|---------|
| `server` | `bin[-dbg]/server` | Database daemon |
| `client` | `bin[-dbg]/client` | CLI client (uses readline) |
| `tester` | `bin[-dbg]/tester` | Integration test driver (server+client combined, uses readline) |

### Core Libraries (in dependency order)

| Library | CMake target | Purpose |
|---------|-------------|---------|
| `src/core/storage` | `stor` | On-disk storage: data files, delta files, readers/writers |
| `src/core/membuf` | `membuf` | In-memory buffers; depends on `stor` |
| `src/core/compute` | `comp` | Distance computations (L1, L2), KNN scanner; depends on `stor`, `membuf` |
| `src/core/control` | `cntrl` | Top-level orchestration; depends on all three above |

Both `server` and `tester` link against all four core libraries.

### Storage Design

The storage layer uses two file types:
- **Data files**: immutable, store bulk vector data. Layout: `[DataFileHeader][vectors][ids]`. Each vector entry is a `u64` id followed by raw float data.
- **Delta files**: smaller mutable files tracking updates/deletes. Merged into data files when they exceed a size threshold.

Vectors are identified by a `uint64_t` vector id. A vector's file is deterministic from its id (each data file covers a range of ids). Supported data types: `f32` (Float), `f16` (Float16); dimension range: 4–4096.

Stage 1 components being implemented (see `src/core/storage/DESIGN.md`):
- `InputGenerator` — generates test datasets in text format
- `InputReader` — memory-maps and parses input text files
- `DataWriter` — converts `InputReader` content to binary data files
- `DataReader` — reads binary data files, with iterator and lookup by id
- `Scanner` — K-nearest-neighbor search over a `DataReader`

Functions that can fail return a `Ret` struct (to be defined in `src/core/utils/`).

### Include Paths

`src/` is the include root. Use paths like:
```cpp
#include "core/storage/data_file.h"
```

### Namespace

All code lives under the `sketch2` namespace (e.g., `sketch2::storage`).

### Compiler Flags

`-Wall -Wextra -Wpedantic -Werror` — all warnings are errors.

### External Dependencies

- **GoogleTest** v1.15.2 — downloaded automatically via `FetchContent` on first configure
- **readline** + **ncurses** — used by `client` and `tester`; install `libreadline-dev` if missing
