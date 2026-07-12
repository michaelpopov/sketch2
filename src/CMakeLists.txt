add_subdirectory(third-party/croaring)
add_subdirectory(core)
add_subdirectory(db)
add_subdirectory(libsketch2)
add_subdirectory(sketch2api)
add_subdirectory(third-party/sqlite)
# Google Highway — vendored SIMD library.
# Disable tests, examples, contrib, and install to keep the build lightweight.
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(HWY_WARNINGS_ARE_ERRORS OFF CACHE BOOL "" FORCE)
add_subdirectory(third-party/google_highway)

# llama.cpp — git submodule (third-party/llama_cpp), CPU backend only. As a
# subdirectory LLAMA_STANDALONE is OFF, so tools/examples/tests default off.
set(GGML_CUDA   OFF CACHE BOOL "" FORCE)   # CPU only: no CUDA toolkit required
set(GGML_METAL  OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
# GGML_NATIVE=OFF enables a fixed x86-64-v3-class set (SSE4.2/AVX/AVX2/BMI2/
# F16C/FMA) in the ggml units only — a deliberate AVX2-class (Haswell 2013+,
# Zen 2017+) deployment requirement of the embedding API. sk_embedder_open()
# checks the CPU at runtime and fails cleanly instead of SIGILL. NATIVE stays
# forced OFF: -march=native exposes no feature list the guard could check.
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
# aarch64 would default to plain armv8-a, disabling the fp16/dotprod kernels
# that dominate quantized inference; match the project baseline (see
# SKETCH_ARM_MCPU). sk_embedder_open() verifies at runtime. Not FORCEd —
# override with -DGGML_CPU_ARM_ARCH=... for a different target.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(GGML_CPU_ARM_ARCH "armv8.2-a+fp16+dotprod" CACHE STRING "")
endif()
# ggml's std::thread pool instead of OpenMP: no libgomp runtime dependency.
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
# Build llama/ggml as static archives; global CMAKE_POSITION_INDEPENDENT_CODE is on.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
# Belt-and-suspenders: keep the auxiliary targets off even if upstream flips a default.
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
add_subdirectory(third-party/llama_cpp)

# Compile llama/ggml with per-function sections so the libsketch2 link can
# discard unreferenced code (--gc-sections) once the symbols are localized.
foreach(_llama_target llama ggml ggml-base ggml-cpu)
  target_compile_options(${_llama_target} PRIVATE -ffunction-sections -fdata-sections)
endforeach()

# Link interface for the bundled llama.cpp; libsketch2 localizes the symbols
# (--exclude-libs) and trims unreferenced code (--gc-sections).
add_library(sketch2_llama INTERFACE)
target_link_libraries(sketch2_llama INTERFACE
  llama ggml ggml-base ggml-cpu Threads::Threads)
