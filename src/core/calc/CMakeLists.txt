set(CALC_SOURCES
  calc_engine.cpp
  scanner_ex.cpp
)

set(CALC_UTEST_SOURCES
  utest_scanner_ex.cpp
)

if(SKETCH_CALC_ENGINE STREQUAL "highway")
  list(APPEND CALC_SOURCES
    hwy_kernels.cpp
    scanner_hw.cpp
  )
  list(APPEND CALC_UTEST_SOURCES
    utest_hwy_kernels.cpp
  )
elseif(SKETCH_CALC_ENGINE STREQUAL "numkong")
  list(APPEND CALC_SOURCES
    nk_kernels.cpp
    scanner_nk.cpp
  )
  list(APPEND CALC_UTEST_SOURCES
    utest_nk_kernels.cpp
  )
else()
  message(FATAL_ERROR "Unsupported SKETCH_CALC_ENGINE='${SKETCH_CALC_ENGINE}'")
endif()

add_library(calc STATIC ${CALC_SOURCES})
set_target_properties(calc PROPERTIES OUTPUT_NAME "calc")
target_link_libraries(calc PUBLIC stor hwy sketch_compile_options Threads::Threads)
if(SKETCH_CALC_ENGINE STREQUAL "numkong")
  target_include_directories(calc SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/src/third-party/numkong)
endif()

add_executable(bench_calc bench_calc.cpp)
target_link_libraries(bench_calc PRIVATE calc stor Threads::Threads)

if(SKETCH_CALC_ENGINE STREQUAL "numkong" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  set_source_files_properties(nk_kernels.cpp PROPERTIES
    COMPILE_DEFINITIONS "NK_TARGET_HASWELL=1;NK_TARGET_SKYLAKE=1")
elseif(SKETCH_CALC_ENGINE STREQUAL "numkong" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  if(SKETCH_ENABLE_ARM_SVE)
    # SVE is not part of the portable arm64 baseline. Keep it opt-in so builds
    # remain runnable on Apple Silicon and other non-SVE AArch64 CPUs/VMs.
    set_source_files_properties(nk_kernels.cpp PROPERTIES
      COMPILE_OPTIONS "-march=armv8.2-a+sve+fp16+dotprod")
  endif()
elseif(SKETCH_CALC_ENGINE STREQUAL "numkong" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(wasm32|wasm64)$")
  set_source_files_properties(nk_kernels.cpp PROPERTIES
    COMPILE_DEFINITIONS "NK_TARGET_V128RELAXED=1")
endif()

# Highway's foreach_target generates code that may trigger warnings promoted to
# errors by the project's global -Werror / -Wpedantic flags.
if(SKETCH_CALC_ENGINE STREQUAL "highway")
  set_source_files_properties(hwy_kernels.cpp PROPERTIES COMPILE_OPTIONS "-Wno-error;-Wno-pedantic")
endif()
if(SKETCH_CALC_ENGINE STREQUAL "highway" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  if(SKETCH_ENABLE_ARM_SVE)
    set_source_files_properties(hwy_kernels.cpp PROPERTIES
      COMPILE_OPTIONS "-Wno-error;-Wno-pedantic;-march=armv8.2-a+sve+fp16+dotprod")
  endif()
endif()

add_executable(utest_calc
  ${SHARED_UTEST_MAIN}
  ${CALC_UTEST_SOURCES}
)
if(SKETCH_CALC_ENGINE STREQUAL "numkong")
  target_include_directories(utest_calc SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/src/third-party/numkong)
endif()
target_link_libraries(utest_calc PRIVATE calc stor GTest::gtest Threads::Threads)
add_test(NAME utest_calc COMMAND utest_calc)
