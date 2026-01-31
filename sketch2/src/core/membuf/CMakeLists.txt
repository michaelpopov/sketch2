add_library(membuf STATIC membuf.cpp)
set_target_properties(membuf PROPERTIES OUTPUT_NAME "membuf")
target_link_libraries(membuf PUBLIC Threads::Threads)

add_executable(utest_membuf utest_main.cpp)
target_link_libraries(utest_membuf PRIVATE membuf stor GTest::gtest Threads::Threads)
add_test(NAME utest_membuf COMMAND utest_membuf)
