add_library(sketch2 SHARED sketch2.cpp)

set_target_properties(sketch2 PROPERTIES OUTPUT_NAME "sketch2")

target_link_libraries(sketch2
  PRIVATE
    -Wl,--whole-archive
    parasol
    vlite
    -Wl,--no-whole-archive
    comp
    stor
    utils
)

add_custom_command(TARGET sketch2 POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:sketch2>
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<TARGET_FILE_NAME:sketch2>
)
