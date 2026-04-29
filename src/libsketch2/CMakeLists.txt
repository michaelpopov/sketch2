add_library(sketch2 SHARED sketch2.cpp)

set_target_properties(sketch2 PROPERTIES OUTPUT_NAME "sketch2")

target_link_libraries(sketch2
  PRIVATE
    -Wl,--whole-archive
    sketch2api
    sk2sqlite
    -Wl,--no-whole-archive
    bitset
    compute
    stor
    utils
    hwy
    sketch_warnings
)
