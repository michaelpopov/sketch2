# Storage Layer Instructions

This file provides specialized guidance for the `src/core/storage` directory.

## Storage Design Principles
- **Data Files:** Immutable bulk vector data. Layout: `[DataFileHeader][vectors][ids]`.
- **Delta Files:** Small, mutable files tracking updates/deletes.
- **Merging:** Updates are initially written to delta files and merged into data files when they grow too large.
- **Vector ID:** A `uint64_t` identifier. Data files cover a deterministic range of IDs.

## Core Components (Stage 1)
- `InputGenerator`: Generates test datasets.
- `InputReader`: Memory-maps and parses input text files.
- `DataWriter`: Converts `InputReader` content to binary data files.
- `DataReader`: Reads binary data files with iterators and ID lookups.
- `Scanner`: Performs K-nearest-neighbor searches.
- `Dataset`: Orchestrates file access and manages ID-to-file mapping.

## Implementation Standards
- **Memory Mapping:** Use `mmap` (via `InputReader` or similar) for efficient file access.
- **Interface Consistency:** Ensure all storage components use the `sketch2::Ret` class for error reporting.
- **Data Types:** Support `f32` (Float) and `f16` (Float16) data types (refer to `src/core/utils/shared_types.h`).
- **Namespace:** All classes should be in `sketch2::storage`.

## Testing
- Each storage component must have a corresponding `utest_<component_name>.cpp` file.
- Use `InputGenerator` to create deterministic test data for storage unit tests.
