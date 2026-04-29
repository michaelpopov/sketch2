add_library(bitset STATIC
  dynamic_bitset.cpp
  roaring_ids.cpp
  chunked_bits.cpp
  bitset_filter_control.cpp
  bitset_file_cache.cpp
)

target_include_directories(bitset PUBLIC ${CMAKE_SOURCE_DIR}/src)

set_target_properties(bitset PROPERTIES OUTPUT_NAME "bitset")

target_link_libraries(bitset
  PUBLIC
    utils
    sketch_compile_options
    Threads::Threads
    croaring
  PRIVATE
    sketch_warnings
)

add_executable(utest_bitset
  ${SHARED_UTEST_MAIN}
  utest_dynamic_bitset.cpp
  utest_roaring_ids.cpp
  utest_chunked_bits.cpp
  utest_bitset_filter_control.cpp
  utest_bitset_file_cache.cpp
)

target_link_libraries(utest_bitset
  PRIVATE
    bitset
    GTest::gtest
    Threads::Threads
    sketch_warnings
)

add_test(NAME utest_bitset COMMAND utest_bitset)
