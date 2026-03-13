add_library(vlite SHARED vlite.cpp)

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

# Keep the SQLite extension artifact next to runtime binaries.
add_custom_command(TARGET vlite POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:vlite>
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<TARGET_FILE_NAME:vlite>
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
target_compile_definitions(utest_vlite PRIVATE VLITE_EXTENSION_PATH=\"$<TARGET_FILE:vlite>\")
add_dependencies(utest_vlite vlite)
add_test(NAME utest_vlite COMMAND utest_vlite)
