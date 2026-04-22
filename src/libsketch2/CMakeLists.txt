add_library(sketch2 SHARED sketch2.cpp)

set_target_properties(sketch2 PROPERTIES OUTPUT_NAME "sketch2")

target_link_libraries(sketch2
  PRIVATE
    -Wl,--whole-archive
    sketch2api
    sk2sqlite
    -Wl,--no-whole-archive
    compute
    stor
    utils
    hwy
)

add_custom_command(TARGET sketch2 POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:sketch2>
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<TARGET_FILE_NAME:sketch2>
  COMMENT "Syncing libsketch2.so into the runtime output directory"
)

add_custom_target(sync_sketch2_runtime ALL
  DEPENDS sketch2
  COMMAND ${CMAKE_COMMAND} -E make_directory
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:sketch2>
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<TARGET_FILE_NAME:sketch2>
  COMMENT "Ensuring libsketch2.so is synced into the runtime output directory"
)
