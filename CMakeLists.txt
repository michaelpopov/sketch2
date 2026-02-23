cmake_minimum_required(VERSION 3.20)

# Fix FetchContent timestamp warning in CMake >= 3.28
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

project(sketch2 LANGUAGES CXX)

#----------------------------
# Support for f16
#----------------------------
add_library(sketch_compile_options INTERFACE)
option(SKETCH_ENABLE_AVX2 "Enable AVX2 optimizations on x86_64" ON)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  target_compile_options(sketch_compile_options INTERFACE -march=armv8.2-a+fp16)
endif()

if(SKETCH_ENABLE_AVX2 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  target_compile_options(sketch_compile_options INTERFACE -mavx2)
endif()

# ---------------------------
# Global build configuration
# ---------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Warnings / errors (GCC / Clang)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# Threading (safe default for executables/tests)
find_package(Threads REQUIRED)

# Output directories (Ninja is single-config; this uses CMAKE_BUILD_TYPE)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
endif()

string(TOLOWER "${CMAKE_BUILD_TYPE}" _bt_lower)
if (_bt_lower STREQUAL "debug")
  set(_RUNTIME_OUT_DIR "${CMAKE_SOURCE_DIR}/bin-dbg")
else()
  set(_RUNTIME_OUT_DIR "${CMAKE_SOURCE_DIR}/bin")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${_RUNTIME_OUT_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# ---------------------------
# Dependencies
# ---------------------------
include(FetchContent)

# GoogleTest (downloaded automatically)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_subdirectory(src)
