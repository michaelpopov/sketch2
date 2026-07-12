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
    sketch2_llama
    sketch_warnings
)

# Localize the llama/ggml symbols and let --gc-sections discard unreferenced
# code. The archive list must stay explicit: --exclude-libs,ALL would also
# localize the sk_* API coming from the whole-archive sketch2api.
target_link_options(sketch2
  PRIVATE
    -Wl,--exclude-libs,libllama.a:libggml.a:libggml-base.a:libggml-cpu.a
    -Wl,--gc-sections
)
