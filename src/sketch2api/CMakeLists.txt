add_library(sketch2api STATIC
  internal.cpp
  internal_bitset.cpp
  internal_embed.cpp
  sketch2.cpp
  sketch2api_utils.cpp
)

set_target_properties(sketch2api PROPERTIES
  OUTPUT_NAME sketch2api
)

target_include_directories(sketch2api
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# internal_embed.cpp compiles against llama.h; llama symbols are provided by
# the consumer (libsketch2 via sketch2_llama, unit tests link llama directly).
target_include_directories(sketch2api
  PRIVATE
    ${CMAKE_SOURCE_DIR}/src/third-party/llama_cpp/include
    ${CMAKE_SOURCE_DIR}/src/third-party/llama_cpp/ggml/include
)

target_link_libraries(sketch2api
  PRIVATE
    bitset
    compute
    stor
    utils
    sketch_warnings
)

add_executable(utest_sketch2
  ${SHARED_UTEST_MAIN}
  utest_sketch2api.cpp
  utest_bitset.cpp
)
target_include_directories(utest_sketch2 PRIVATE ${CMAKE_SOURCE_DIR}/src/core)
target_link_libraries(utest_sketch2 PRIVATE sketch2api compute bitset llama GTest::gtest Threads::Threads sketch_warnings)
add_test(NAME utest_sketch2 COMMAND utest_sketch2)

add_executable(utest_compute_sketch2api
  ${SHARED_UTEST_MAIN}
  utest_compute_sketch2api.cpp
)
target_include_directories(utest_compute_sketch2api PRIVATE ${CMAKE_SOURCE_DIR}/src/core)
target_link_libraries(utest_compute_sketch2api PRIVATE sketch2api compute bitset llama GTest::gtest Threads::Threads sketch_warnings)
add_test(NAME utest_compute_sketch2api COMMAND utest_compute_sketch2api)

add_executable(utest_embed
  ${SHARED_UTEST_MAIN}
  utest_embed.cpp
)
target_include_directories(utest_embed PRIVATE ${CMAKE_SOURCE_DIR}/src/core)
target_link_libraries(utest_embed PRIVATE sketch2api compute bitset llama GTest::gtest Threads::Threads sketch_warnings)
add_test(NAME utest_embed COMMAND utest_embed)

# E2E embedding tests need a real GGUF model: a pinned, checksum-verified
# fixture cached under the build tree (SKETCH2_EMBED_TEST_MODEL overrides).
# Without it the tests skip; CI sets SKETCH2_REQUIRE_EMBED_TESTS=1 to fail
# instead.
set(SKETCH2_EMBED_TEST_MODEL_PATH
  ${CMAKE_BINARY_DIR}/test-models/all-MiniLM-L6-v2-Q4_K_M.gguf)
target_compile_definitions(utest_embed PRIVATE
  SKETCH2_EMBED_TEST_MODEL_DEFAULT="${SKETCH2_EMBED_TEST_MODEL_PATH}")
add_custom_target(fetch-embed-test-model
  COMMAND ${CMAKE_SOURCE_DIR}/scripts/fetch-embed-test-model.sh
          ${SKETCH2_EMBED_TEST_MODEL_PATH}
  COMMENT "Fetching the pinned embedding test model"
  VERBATIM)
