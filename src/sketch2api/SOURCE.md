# Source Index

- `CMakeLists.txt`: Build rules for the sketch2api library and tests.
- `BITSET.md`: Sketch2 API ownership doc for the binary bitset filter format used by query helpers.
- `DESIGN.md`: Design notes for the sketch2api API layer.
- `README.md`: Overview of the sketch2api C API and usage model.
- `internal_embed.cpp`: C API implementation for the sk_embedder text-embedding functions, backed by the bundled llama.cpp (CPU only).
- `sketch2.cpp`: C API implementation for dataset lifecycle, updates, queries, and text/binary bulk generation or loading.
- `sketch2.h`: Public C API declarations exposed by sketch2api, including text and binary bulk-ingest helpers and the sk_embedder embedding API.
- `sketch2api_testing.h`: Testing-only API declarations intentionally kept out of the main public header.
- `utest_main.cpp`: Shared GoogleTest entry point (`src/core/utils/utest_main.cpp`) for the sketch2api test binary.
- `utest_sketch2api.cpp`: Unit tests for the sketch2api API.
- `utest_bitset.cpp`: Unit tests for sketch2api bitset builder, load/drop/cache, and bitset-filtered KNN behavior.
- `utest_embed.cpp`: Unit tests for the sk_embedder API; end-to-end tests run only when SKETCH2_EMBED_TEST_MODEL points at a GGUF embedding model.
