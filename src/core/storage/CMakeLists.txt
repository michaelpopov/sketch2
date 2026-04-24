add_library(stor STATIC
  input_generator.cpp
  input_reader.cpp
  input_writer.cpp
  data_writer.cpp
  data_reader.cpp
  data_merger.cpp
  dataset.cpp
  dataset_reader.cpp
  dataset_writer.cpp
  dataset_node.cpp
  compact_ids_offsets.cpp
  compact_ids_misses.cpp
  compact_ids_bitset.cpp
  compact_ids.cpp
  compact_ids_shared.cpp
)

target_include_directories(stor PUBLIC ${CMAKE_SOURCE_DIR}/src)

set_target_properties(stor PROPERTIES OUTPUT_NAME "stor")
target_link_libraries(stor PUBLIC utils sketch_compile_options Threads::Threads)
target_link_libraries(stor PRIVATE sketch_warnings)

add_executable(utest_stor
  ${SHARED_UTEST_MAIN}
  utest_input_generator.cpp
  utest_input_reader.cpp
  utest_input_writer.cpp
  utest_data_file_layout.cpp
  utest_data_writer.cpp
  utest_data_reader.cpp
  utest_data_merger.cpp
  utest_dataset.cpp
  utest_full_cycle.cpp
  utest_compact_ids_offsets.cpp
  utest_compact_ids_ext.cpp
)
target_link_libraries(utest_stor PRIVATE stor croaring GTest::gtest Threads::Threads sketch_warnings)
add_test(NAME utest_stor COMMAND utest_stor)
