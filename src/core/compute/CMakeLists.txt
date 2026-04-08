add_executable(bench_comp bench_compute.cpp)
target_link_libraries(bench_comp PRIVATE stor Threads::Threads)

if(TARGET benchmark::benchmark)
  add_executable(gbench_comp gbench_compute.cpp)
  target_link_libraries(gbench_comp PRIVATE benchmark::benchmark calc stor Threads::Threads)
endif()

add_executable(utest_comp
  ${SHARED_UTEST_MAIN}
  utest_compute_dot.cpp
  utest_compute_l2.cpp
  utest_compute_cos.cpp
  utest_compute_runtime.cpp
)
target_link_libraries(utest_comp PRIVATE stor GTest::gtest Threads::Threads)
add_test(NAME utest_comp COMMAND utest_comp)
