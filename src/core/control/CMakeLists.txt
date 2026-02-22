add_library(cntrl STATIC control.cpp)
set_target_properties(cntrl PROPERTIES OUTPUT_NAME "cntrl")
target_link_libraries(cntrl PUBLIC comp membuf stor sketch_compile_options Threads::Threads)

add_executable(utest_cntrl utest_main.cpp)
target_link_libraries(utest_cntrl PRIVATE cntrl stor comp membuf GTest::gtest Threads::Threads)
add_test(NAME utest_cntrl COMMAND utest_cntrl)
