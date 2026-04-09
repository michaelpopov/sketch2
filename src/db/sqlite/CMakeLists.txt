add_library(sk2sqlite STATIC vlite.cpp)

set_target_properties(sk2sqlite PROPERTIES
  OUTPUT_NAME sk2sqlite
)

target_include_directories(sk2sqlite
  PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/third-party/sqlite
)

target_link_libraries(sk2sqlite
  PRIVATE
    calc
    stor
    utils
    sketch_compile_options
    Threads::Threads
)

add_executable(utest_vlite
  ${SHARED_UTEST_MAIN}
  utest_vlite.cpp
)
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
