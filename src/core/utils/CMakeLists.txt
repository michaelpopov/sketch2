add_library(utils STATIC utils.cpp ini_reader.cpp)

target_include_directories(utils PUBLIC ${CMAKE_SOURCE_DIR}/src/core)

set_target_properties(utils PROPERTIES OUTPUT_NAME "utils")

target_link_libraries(utils PUBLIC sketch_compile_options)

add_executable(utest_utils utest_main.cpp utest_ini_reader.cpp)
target_link_libraries(utest_utils PRIVATE utils GTest::gtest Threads::Threads)
add_test(NAME utest_utils COMMAND utest_utils)
