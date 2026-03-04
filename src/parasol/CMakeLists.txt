add_library(parasol SHARED parasol.cpp)

set_target_properties(parasol PROPERTIES
  OUTPUT_NAME parasol
)

target_include_directories(parasol
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(parasol
  PRIVATE
    cntrl
    comp
    membuf
    stor
    utils
)

# Keep shared C API artifact next to runtime binaries.
add_custom_command(TARGET parasol POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:parasol>
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<TARGET_FILE_NAME:parasol>
)

add_executable(utest_parasol utest_main.cpp utest_parasol.cpp)
target_link_libraries(utest_parasol PRIVATE parasol GTest::gtest Threads::Threads)
add_test(NAME utest_parasol COMMAND utest_parasol)
