add_library(parasol STATIC parasol.cpp)

set_target_properties(parasol PROPERTIES
  OUTPUT_NAME parasol
)

target_include_directories(parasol
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(parasol
  PRIVATE
    comp
    stor
    utils
)

add_executable(utest_parasol
  ${SHARED_UTEST_MAIN}
  utest_parasol.cpp
)
target_include_directories(utest_parasol PRIVATE ${CMAKE_SOURCE_DIR}/src/core)
target_link_libraries(utest_parasol PRIVATE parasol GTest::gtest Threads::Threads)
add_test(NAME utest_parasol COMMAND utest_parasol)
