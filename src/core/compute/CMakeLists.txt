add_library(comp STATIC compute.cpp scanner.cpp)
set_target_properties(comp PROPERTIES OUTPUT_NAME "comp")
target_link_libraries(comp PUBLIC stor sketch_compile_options Threads::Threads)

add_executable(utest_comp utest_main.cpp utest_scanner.cpp utest_compute_l1.cpp)
target_link_libraries(utest_comp PRIVATE comp stor membuf GTest::gtest Threads::Threads)
add_test(NAME utest_comp COMMAND utest_comp)
