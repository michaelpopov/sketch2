add_library(stor STATIC storage.cpp input_generator.cpp input_reader.cpp data_writer.cpp)

target_include_directories(stor PUBLIC ${CMAKE_SOURCE_DIR}/src)

set_target_properties(stor PROPERTIES OUTPUT_NAME "stor")
target_link_libraries(stor PUBLIC utils Threads::Threads)

add_executable(utest_stor utest_main.cpp utest_input_generator.cpp utest_input_reader.cpp utest_data_writer.cpp)
target_link_libraries(utest_stor PRIVATE stor GTest::gtest Threads::Threads)
add_test(NAME utest_stor COMMAND utest_stor)
