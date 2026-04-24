add_library(croaring STATIC roaring.c)

set_target_properties(croaring PROPERTIES
  OUTPUT_NAME roaring
  C_STANDARD 11
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
)

target_include_directories(croaring
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(croaring PRIVATE -O3)
