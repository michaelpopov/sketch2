add_library(calc STATIC
  calc_engine.cpp
  compute_kernels.cpp
  hwy_kernels.cpp
  nk_kernels.cpp
  scanner_ex.cpp
)
set_target_properties(calc PROPERTIES OUTPUT_NAME "calc")
target_link_libraries(calc PUBLIC stor hwy sketch_compile_options Threads::Threads)
target_include_directories(calc SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/src/third-party/numkong)

add_executable(bench_calc bench_calc.cpp)
target_link_libraries(bench_calc PRIVATE calc stor Threads::Threads)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  set_source_files_properties(nk_kernels.cpp PROPERTIES
    COMPILE_DEFINITIONS "NK_TARGET_HASWELL=1;NK_TARGET_SKYLAKE=1")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set_source_files_properties(nk_kernels.cpp PROPERTIES
    COMPILE_DEFINITIONS "NK_TARGET_NEON=1;NK_TARGET_NEONFHM=1;NK_TARGET_SVE=1;NK_TARGET_SVEHALF=1")
  set_source_files_properties(nk_kernels.cpp PROPERTIES
    COMPILE_OPTIONS "-march=armv8.2-a+sve+fp16+dotprod")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(wasm32|wasm64)$")
  set_source_files_properties(nk_kernels.cpp PROPERTIES
    COMPILE_DEFINITIONS "NK_TARGET_V128RELAXED=1")
endif()

# Highway's foreach_target generates code that may trigger warnings promoted to
# errors by the project's global -Werror / -Wpedantic flags.
set_source_files_properties(hwy_kernels.cpp PROPERTIES COMPILE_OPTIONS "-Wno-error;-Wno-pedantic")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set_source_files_properties(hwy_kernels.cpp PROPERTIES
    COMPILE_OPTIONS "-Wno-error;-Wno-pedantic;-march=armv8.2-a+sve+fp16+dotprod")
endif()

if(DEFINED ENV{DUMMY_CALC} AND NOT "$ENV{DUMMY_CALC}" STREQUAL "")
  target_compile_definitions(calc PRIVATE DUMMY_CALC)
endif()

add_executable(utest_calc
  ${SHARED_UTEST_MAIN}
  utest_hwy_kernels.cpp
  utest_nk_kernels.cpp
  utest_scanner_ex.cpp
)
target_include_directories(utest_calc SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/src/third-party/numkong)
target_link_libraries(utest_calc PRIVATE calc stor GTest::gtest Threads::Threads)
add_test(NAME utest_calc COMMAND utest_calc)
