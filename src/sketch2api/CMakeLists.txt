add_library(sketch2api STATIC
  allowlist_control.cpp
  internal.cpp
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

target_link_libraries(sketch2api
  PRIVATE
    compute
    stor
    utils
    sketch_warnings
)

add_executable(utest_sketch2
  ${SHARED_UTEST_MAIN}
  utest_sketch2api.cpp
)
target_include_directories(utest_sketch2 PRIVATE ${CMAKE_SOURCE_DIR}/src/core)
target_link_libraries(utest_sketch2 PRIVATE sketch2api compute GTest::gtest Threads::Threads sketch_warnings)
add_test(NAME utest_sketch2 COMMAND utest_sketch2)

add_executable(utest_compute_sketch2api
  ${SHARED_UTEST_MAIN}
  utest_compute_sketch2api.cpp
)
target_include_directories(utest_compute_sketch2api PRIVATE ${CMAKE_SOURCE_DIR}/src/core)
target_link_libraries(utest_compute_sketch2api PRIVATE sketch2api compute GTest::gtest Threads::Threads sketch_warnings)
add_test(NAME utest_compute_sketch2api COMMAND utest_compute_sketch2api)
