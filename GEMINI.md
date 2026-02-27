# GEMINI.md - Project Instructions

This file provides foundational mandates for Gemini CLI when working in the `sketch2` repository. These instructions take absolute precedence over general defaults.

## Project Context
`sketch2` is a high-performance vector database written in C++. It is currently in Stage 1, focusing on the storage layer and basic K-NN search.

## Tech Stack & Conventions
- **Language:** C++17 or higher.
- **Build System:** CMake (preferred with Ninja generator).
- **Compiler:** `g++` (strict flags: `-Wall -Wextra -Wpedantic -Werror`).
- **Testing:** GoogleTest (GTest).
- **Namespace:** All code must reside within the `sketch2` namespace (e.g., `sketch2::storage`).
- **Includes:** Use paths relative to `src/` (e.g., `#include "core/storage/data_file.h"`).
- **Error Handling:** Functions that can fail should return the `sketch2::Ret` struct (defined in `src/core/utils/shared_types.h`). Use the `CHECK(ret)` macro for propagation.

## Build & Test Commands

### Debug Build (Standard Development)
```sh
cmake -S . -B build-dbg -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=g++
cmake --build build-dbg
```

### Running Tests
```sh
# Run all tests
ctest --test-dir build-dbg --output-on-failure

# Run a specific test suite
ctest --test-dir build-dbg -R <test_name> --output-on-failure
```

## Engineering Standards
- **Style:** Adhere to `.clang-format`. Use `snake_case` for filenames and variables, `PascalCase` for classes.
- **Surgical Changes:** Prioritize targeted modifications. Always verify changes by running relevant unit tests.
- **Documentation:** Maintain `README.md` and `DESIGN.md` files in subdirectories when architectural changes occur.
- **Safety:** Never commit secrets or large binary test data.
