cmake_minimum_required(VERSION 3.20)

# Fix FetchContent timestamp warning in CMake >= 3.28
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

project(sketch2 LANGUAGES C CXX)

#----------------------------
# Support for f16
#----------------------------
add_library(sketch_compile_options INTERFACE)
option(SKETCH_ENABLE_AVX2 "Enable AVX2 runtime-dispatched x86_64 compute backends" ON)
target_compile_definitions(sketch_compile_options INTERFACE
  SKETCH_ENABLE_AVX2=$<IF:$<BOOL:${SKETCH_ENABLE_AVX2}>,1,0>)

if(SKETCH_ENABLE_AVX2 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR
      "SKETCH_ENABLE_AVX2 requires GCC or Clang on x86 because the AVX2 kernels "
      "use per-function target attributes for runtime dispatch.")
  endif()
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  target_compile_options(sketch_compile_options INTERFACE -march=armv8.2-a+fp16)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sketch_compile_options INTERFACE -msse2)
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

# Warnings / errors (GCC / Clang)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# Sanitizer build type for GCC/Clang.
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(SKETCH_SANITIZER_FLAGS
      "-fsanitize=address,undefined,leak -fno-omit-frame-pointer"
      CACHE STRING "Compiler and linker flags used by the Sanitizer build type")
endif()

# Threading (safe default for executables/tests)
find_package(Threads REQUIRED)

# Output directories (Ninja is single-config; this uses CMAKE_BUILD_TYPE)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug, Release, RelWithDebInfo, MinSizeRel, Sanitizer)" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel Sanitizer)

string(TOLOWER "${CMAKE_BUILD_TYPE}" _bt_lower)

if (_bt_lower STREQUAL "sanitizer")
  if (NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "The Sanitizer build type is supported only with GCC or Clang.")
  endif()

  set(CMAKE_C_FLAGS_SANITIZER
      "${CMAKE_C_FLAGS_DEBUG} ${SKETCH_SANITIZER_FLAGS}"
      CACHE STRING "Flags used by the C compiler during Sanitizer builds" FORCE)
  set(CMAKE_CXX_FLAGS_SANITIZER
      "${CMAKE_CXX_FLAGS_DEBUG} ${SKETCH_SANITIZER_FLAGS}"
      CACHE STRING "Flags used by the C++ compiler during Sanitizer builds" FORCE)
  set(CMAKE_EXE_LINKER_FLAGS_SANITIZER
      "${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${SKETCH_SANITIZER_FLAGS}"
      CACHE STRING "Flags used for linking executables during Sanitizer builds" FORCE)
  set(CMAKE_SHARED_LINKER_FLAGS_SANITIZER
      "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} ${SKETCH_SANITIZER_FLAGS}"
      CACHE STRING "Flags used for linking shared libraries during Sanitizer builds" FORCE)
  set(CMAKE_MODULE_LINKER_FLAGS_SANITIZER
      "${CMAKE_MODULE_LINKER_FLAGS_DEBUG} ${SKETCH_SANITIZER_FLAGS}"
      CACHE STRING "Flags used for linking loadable modules during Sanitizer builds" FORCE)
endif()

if (_bt_lower STREQUAL "debug")
  set(_RUNTIME_OUT_DIR "${CMAKE_SOURCE_DIR}/bin-dbg")
elseif (_bt_lower STREQUAL "sanitizer")
  set(_RUNTIME_OUT_DIR "${CMAKE_SOURCE_DIR}/bin-san")
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
