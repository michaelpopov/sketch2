# Source Index

- `CMakeLists.txt`: Build rules for the bitset library and bitset-focused unit tests.
- `SOURCE.md`: Index of the bitset module sources and test helpers.
- `bitset_filter.h`: Shared bitset filter struct exposed by scanner APIs.
- `bitset_filter_control.cpp`: Materialization and storage management for serialized bitset filters.
- `bitset_filter_control.h`: Ownership and loading declarations for persisted or in-memory bitset filters.
- `chunked_bits.cpp`: Chunked bitset serialization and view implementation.
- `chunked_bits.h`: Chunked bitset builder and zero-copy view declarations.
- `dynamic_bitset.cpp`: Growable bitset implementation used by storage readers.
- `dynamic_bitset.h`: Growable bitset declarations.
- `roaring_ids.cpp`: RoaringIds implementation backed by CRoaring bitmaps.
- `roaring_ids.h`: RoaringIds container declarations for compact uint64_t id sets.
- `utest_bitset_filter_control.cpp`: Unit tests for bitset filter storage ownership and persistence.
- `utest_chunked_bits.cpp`: Unit tests for chunked bitset serialization and iteration.
- `utest_chunked_bits_helpers.h`: Shared helpers for building and inspecting serialized chunked bitsets in tests.
- `utest_dynamic_bitset.cpp`: Unit tests for the dynamic bitset.
- `utest_roaring_ids.cpp`: Unit tests for RoaringIds container behavior and serialization.
- `utest_roaring_ids_helpers.h`: Shared helpers for tests that build Roaring id trailers.
