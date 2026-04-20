set(COMPUTE_SOURCES
  compute_engine.cpp
  scanner.cpp
)

set(COMPUTE_UTEST_SOURCES
  utest_scanner_ex.cpp
)

if(SKETCH_COMPUTE_ENGINE STREQUAL "highway")
  list(APPEND COMPUTE_SOURCES
    highway.cpp
  )
  list(APPEND COMPUTE_UTEST_SOURCES
    utest_hwy_kernels.cpp
  )
elseif(SKETCH_COMPUTE_ENGINE STREQUAL "numkong")
  list(APPEND COMPUTE_SOURCES
    numkong.cpp
  )
  list(APPEND COMPUTE_UTEST_SOURCES
    utest_nk_kernels.cpp
  )
else()
  message(FATAL_ERROR "Unsupported SKETCH_COMPUTE_ENGINE='${SKETCH_COMPUTE_ENGINE}'")
endif()

add_library(compute STATIC ${COMPUTE_SOURCES})
set_target_properties(compute PROPERTIES OUTPUT_NAME "compute")
target_link_libraries(compute PUBLIC stor hwy sketch_compile_options Threads::Threads)
if(SKETCH_COMPUTE_ENGINE STREQUAL "numkong")
  target_include_directories(compute SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/src/third-party/numkong)
endif()

add_executable(bench_compute bench_compute.cpp)
target_link_libraries(bench_compute PRIVATE compute stor Threads::Threads)

if(SKETCH_COMPUTE_ENGINE STREQUAL "numkong" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  set_source_files_properties(numkong.cpp PROPERTIES
    COMPILE_DEFINITIONS "NK_TARGET_HASWELL=1;NK_TARGET_SKYLAKE=1")
elseif(SKETCH_COMPUTE_ENGINE STREQUAL "numkong" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  if(SKETCH_ENABLE_ARM_SVE)
    # SVE is not part of the portable arm64 baseline. Keep it opt-in so builds
    # remain runnable on Apple Silicon and other non-SVE AArch64 CPUs/VMs.
    set_source_files_properties(numkong.cpp PROPERTIES
      COMPILE_OPTIONS "-march=armv8.2-a+sve+fp16+dotprod")
  endif()
elseif(SKETCH_COMPUTE_ENGINE STREQUAL "numkong" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(wasm32|wasm64)$")
  set_source_files_properties(numkong.cpp PROPERTIES
    COMPILE_DEFINITIONS "NK_TARGET_V128RELAXED=1")
endif()

# Highway's foreach_target generates code that may trigger warnings promoted to
# errors by the project's global -Werror / -Wpedantic flags.
if(SKETCH_COMPUTE_ENGINE STREQUAL "highway")
  set_source_files_properties(highway.cpp PROPERTIES COMPILE_OPTIONS "-Wno-error;-Wno-pedantic")
endif()
if(SKETCH_COMPUTE_ENGINE STREQUAL "highway" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  if(SKETCH_ENABLE_ARM_SVE)
    set_source_files_properties(highway.cpp PROPERTIES
      COMPILE_OPTIONS "-Wno-error;-Wno-pedantic;-march=armv8.2-a+sve+fp16+dotprod")
  endif()
endif()

add_executable(utest_compute
  ${SHARED_UTEST_MAIN}
  ${COMPUTE_UTEST_SOURCES}
)
if(SKETCH_COMPUTE_ENGINE STREQUAL "numkong")
  target_include_directories(utest_compute SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/src/third-party/numkong)
endif()
target_link_libraries(utest_compute PRIVATE compute stor GTest::gtest Threads::Threads)
add_test(NAME utest_compute COMMAND utest_compute)
