set(COMPUTE_SOURCES
  highway.cpp
)

set(COMPUTE_UTEST_SOURCES
  utest_hwy_kernels.cpp
  utest_scanner_ex.cpp
)

add_library(compute STATIC ${COMPUTE_SOURCES})
set_target_properties(compute PROPERTIES OUTPUT_NAME "compute")
target_link_libraries(compute PUBLIC stor hwy sketch_compile_options Threads::Threads)
target_link_libraries(compute PRIVATE sketch_warnings)

add_executable(bench_compute bench_compute.cpp)
target_link_libraries(bench_compute PRIVATE compute stor Threads::Threads sketch_warnings)

# Highway's foreach_target expands into this first-party translation unit and can
# trigger warnings that are too noisy for the strict project warning target.
set_source_files_properties(highway.cpp PROPERTIES COMPILE_OPTIONS "-Wno-error;-Wno-pedantic")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  if(SKETCH_ENABLE_ARM_SVE AND SKETCH_ARM_MCPU STREQUAL "")
    set_source_files_properties(highway.cpp PROPERTIES
      COMPILE_OPTIONS "-Wno-error;-Wno-pedantic;-march=armv8.2-a+sve+fp16+dotprod")
  endif()
endif()

add_executable(utest_compute
  ${SHARED_UTEST_MAIN}
  ${COMPUTE_UTEST_SOURCES}
)
target_link_libraries(utest_compute PRIVATE compute stor GTest::gtest Threads::Threads sketch_warnings)
add_test(NAME utest_compute COMMAND utest_compute)
