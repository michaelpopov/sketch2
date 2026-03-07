add_library(stor STATIC storage.cpp input_generator.cpp input_reader.cpp data_writer.cpp data_reader.cpp data_merger.cpp dataset.cpp accumulator.cpp accumulator_wal.cpp)

target_include_directories(stor PUBLIC ${CMAKE_SOURCE_DIR}/src)

set_target_properties(stor PROPERTIES OUTPUT_NAME "stor")
target_link_libraries(stor PUBLIC utils sketch_compile_options Threads::Threads)

add_executable(utest_stor utest_main.cpp utest_input_generator.cpp utest_input_reader.cpp utest_data_file_layout.cpp utest_data_writer.cpp utest_data_reader.cpp utest_data_merger.cpp utest_dataset.cpp utest_full_cycle.cpp utest_accumulator.cpp utest_accumulator_wal.cpp)
target_link_libraries(utest_stor PRIVATE stor GTest::gtest Threads::Threads)
add_test(NAME utest_stor COMMAND utest_stor)
