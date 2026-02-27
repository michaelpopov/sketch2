# Compute Layer Instructions

This file provides specialized guidance for the `src/core/compute` directory.

## Compute Layer Principles
- **K-NN Search:** Implement efficient K-nearest-neighbor search algorithms (Stage 1 focuses on L1 distance).
- **Distance Functions:** Support multiple distance metrics (L1, L2, etc.).
- **SIMD Optimization:** Utilize SIMD (AVX2, NEON) for performance-critical distance computations (refer to `compute_l1_avx2.h`, `compute_l1_neon.h`).

## Core Components (Stage 1)
- `ICompute`: Interface for distance computation.
- `ComputeL1`: Implements L1 distance calculation for supported data types.
- `Scanner`: Performs K-nearest-neighbor searches over a `DataReader`.

## Implementation Standards
- **Interface Consistency:** All compute components must use the `ICompute` interface.
- **Data Types:** Support `f32` (Float) and `f16` (Float16) data types (refer to `src/core/utils/shared_types.h`).
- **Namespace:** All classes should be in `sketch2::compute`.

## Testing
- Each compute component must have a corresponding `utest_<component_name>.cpp` file.
- Use `InputGenerator` to create deterministic test data for compute unit tests.
