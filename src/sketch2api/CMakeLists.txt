add_library(sketch2api STATIC sketch2api.cpp)

set_target_properties(sketch2api PROPERTIES
  OUTPUT_NAME sketch2api
)

target_include_directories(sketch2api
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(sketch2api
  PRIVATE
    comp
    stor
    utils
)

add_executable(utest_sketch2
  ${SHARED_UTEST_MAIN}
  utest_sketch2api.cpp
)
target_include_directories(utest_sketch2 PRIVATE ${CMAKE_SOURCE_DIR}/src/core)
target_link_libraries(utest_sketch2 PRIVATE sketch2api GTest::gtest Threads::Threads)
add_test(NAME utest_sketch2 COMMAND utest_sketch2)
