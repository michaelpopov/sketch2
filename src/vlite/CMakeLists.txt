add_library(vlite STATIC vlite.cpp)

set_target_properties(vlite PROPERTIES
  OUTPUT_NAME vlite
)

target_include_directories(vlite
  PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/third-party/sqlite
)

target_link_libraries(vlite
  PRIVATE
    comp
    stor
    utils
    sketch_compile_options
    Threads::Threads
)

add_executable(utest_vlite utest_main.cpp utest_vlite.cpp)
target_link_libraries(utest_vlite
  PRIVATE
    stor
    utils
    sqlite3_lib
    GTest::gtest
    sketch_compile_options
    Threads::Threads
    dl
)
target_compile_definitions(utest_vlite PRIVATE VLITE_EXTENSION_PATH=\"$<TARGET_FILE:sketch2>\")
add_dependencies(utest_vlite sketch2)
add_test(NAME utest_vlite COMMAND utest_vlite)
