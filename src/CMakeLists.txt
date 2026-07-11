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

# llama.cpp — vendored via git submodule (third-party/llama_cpp), CPU backend only.
# As a subdirectory LLAMA_STANDALONE is OFF, so tools/examples/server/tests/common
# already default off; we mainly pin the backend and portability knobs. This builds
# just the ggml and llama libraries — link them into a target when you need them.
set(GGML_CUDA   OFF CACHE BOOL "" FORCE)   # CPU only: no CUDA toolkit required
set(GGML_METAL  OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
# Portable baseline: don't bake -march=native into ggml (matches this project's
# SKETCH_X86_MARCH default). Set GGML_NATIVE=ON for a machine-tuned build.
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
# Build llama/ggml as static archives; global CMAKE_POSITION_INDEPENDENT_CODE is on.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
# Belt-and-suspenders: keep the auxiliary targets off even if upstream flips a default.
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
add_subdirectory(third-party/llama_cpp)
