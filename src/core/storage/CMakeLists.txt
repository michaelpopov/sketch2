add_library(stor STATIC storage.cpp)
set_target_properties(stor PROPERTIES OUTPUT_NAME "stor")
target_link_libraries(stor PUBLIC Threads::Threads)

add_executable(utest_stor utest_main.cpp)
target_link_libraries(utest_stor PRIVATE stor GTest::gtest Threads::Threads)
add_test(NAME utest_stor COMMAND utest_stor)
