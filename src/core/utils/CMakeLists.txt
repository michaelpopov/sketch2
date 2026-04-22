add_library(utils STATIC
    ini_reader.cpp string_utils.cpp dynamic_bitset.cpp compute_unit.cpp singleton.cpp log.cpp update_notifier.cpp file_path_lock.cpp mapped_region.cpp)

target_include_directories(utils PUBLIC ${CMAKE_SOURCE_DIR}/src/core)

set_target_properties(utils PROPERTIES OUTPUT_NAME "utils")

target_link_libraries(utils PUBLIC sketch_compile_options Threads::Threads)
target_link_libraries(utils PRIVATE sketch_warnings)

add_executable(utest_utils utest_main.cpp utest_ini_reader.cpp utest_string_utils.cpp utest_dynamic_bitset.cpp utest_file_lock.cpp utest_log.cpp utest_singleton.cpp utest_update_notifier.cpp utest_mapped_region.cpp utest_high_perf_heap.cpp)
target_link_libraries(utest_utils PRIVATE utils GTest::gtest Threads::Threads sketch_warnings)
add_test(NAME utest_utils COMMAND utest_utils)

# Keep thread-pool tests available for focused runs, but do not include them in
# the routine utils test target because they add noticeable latency.
add_executable(utest_thread_pool utest_main.cpp utest_thread_pool.cpp)
target_link_libraries(utest_thread_pool PRIVATE utils GTest::gtest Threads::Threads sketch_warnings)
