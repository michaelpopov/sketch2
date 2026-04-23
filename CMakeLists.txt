cmake_minimum_required(VERSION 3.20)

# Fix FetchContent timestamp warning in CMake >= 3.28
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

project(sketch2 LANGUAGES C CXX)

# Shared GoogleTest runner used by all unit test binaries.
set(SHARED_UTEST_MAIN "${CMAKE_SOURCE_DIR}/src/core/utils/utest_main.cpp")

#----------------------------
# Support for f16
#----------------------------
add_library(sketch_compile_options INTERFACE)
option(SKETCH_ENABLE_AVX2 "Enable AVX2 runtime-dispatched x86_64 compute backends" ON)
option(SKETCH_ENABLE_AVX512F "Enable AVX-512F runtime-dispatched x86_64 compute backends" ON)
option(SKETCH_ENABLE_AVX512VNNI "Enable AVX-512 VNNI runtime-dispatched x86_64 compute backends" ON)
option(SKETCH_ENABLE_ARM_SVE "Enable Arm SVE/SVE-half kernels for AArch64 builds on SVE-capable CPUs" OFF)
target_compile_definitions(sketch_compile_options INTERFACE
  SKETCH_ENABLE_AVX2=$<IF:$<BOOL:${SKETCH_ENABLE_AVX2}>,1,0>
  SKETCH_ENABLE_AVX512F=$<IF:$<BOOL:${SKETCH_ENABLE_AVX512F}>,1,0>
  SKETCH_ENABLE_AVX512VNNI=$<IF:$<BOOL:${SKETCH_ENABLE_AVX512VNNI}>,1,0>
  SKETCH_ENABLE_ARM_SVE=$<IF:$<BOOL:${SKETCH_ENABLE_ARM_SVE}>,1,0>
  SKETCH2_COMPUTE_ENGINE_HIGHWAY=1)

if((SKETCH_ENABLE_AVX2 OR SKETCH_ENABLE_AVX512F OR SKETCH_ENABLE_AVX512VNNI)
    AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR
      "SKETCH_ENABLE_AVX2/SKETCH_ENABLE_AVX512F/SKETCH_ENABLE_AVX512VNNI require "
      "GCC or Clang on x86 because the compute kernels use per-function target "
      "attributes for runtime dispatch.")
  endif()
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(SKETCH_ARM_MCPU "" CACHE STRING
    "Optional -mcpu tuning for AArch64 builds (e.g. neoverse-n1, neoverse-n2, neoverse-v1, neoverse-v2, native). Empty uses a generic -march baseline.")
  if(SKETCH_ARM_MCPU STREQUAL "")
    # +dotprod is safe on every Armv8.2-A target we ship to (Apple Silicon,
    # Ampere Altra/AmpereOne, Graviton 2/3/4). Keep it in the baseline so SDOT/UDOT
    # are available to the integer kernels without opting into SVE.
    target_compile_options(sketch_compile_options INTERFACE -march=armv8.2-a+fp16+dotprod)
  else()
    # GCC rejects -march together with -mcpu. -mcpu already selects a specific
    # ISA superset for the chosen core, so emit it on its own.
    target_compile_options(sketch_compile_options INTERFACE -mcpu=${SKETCH_ARM_MCPU})
  endif()
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sketch_compile_options INTERFACE -msse2)
  endif()

  set(SKETCH_X86_MARCH "" CACHE STRING
    "Optional -march value for x86_64 builds (e.g. native, znver4, icelake-server). Empty uses the compiler default. Non-Highway TUs honour this; Highway kernels keep their per-function target attributes.")
  set(SKETCH_X86_MTUNE "" CACHE STRING
    "Optional -mtune value for x86_64 builds (e.g. native, znver4, icelake-server). Empty uses the compiler default. Safe to set even when targeting an older -march.")
  if(NOT SKETCH_X86_MARCH STREQUAL "")
    target_compile_options(sketch_compile_options INTERFACE -march=${SKETCH_X86_MARCH})
  endif()
  if(NOT SKETCH_X86_MTUNE STREQUAL "")
    target_compile_options(sketch_compile_options INTERFACE -mtune=${SKETCH_X86_MTUNE})
  endif()
endif()

# ---------------------------
# Global build configuration
# ---------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Build position-independent code so static libraries can be linked into shared libraries.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# First-party warning policy. Link this target into project-owned targets rather
# than setting global compiler flags that vendored code has to undo later.
add_library(sketch_warnings INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(sketch_warnings INTERFACE -Wall -Wextra -Wpedantic -Werror)
endif()

# Threading (safe default for executables/tests)
find_package(Threads REQUIRED)

# Output directories (Ninja is single-config; this uses CMAKE_BUILD_TYPE)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)

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
  URL_HASH SHA256=f179ec217f9b3b3f3c6e8b02d3e7eda997b49e4ce26d6b235c9053bec9c0bf9f
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_subdirectory(src)
