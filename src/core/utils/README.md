# Utilities

Shared helper classes, platform abstractions, and common data structures used
across the `sketch2` core.

## Architecture and SIMD Detection

The project uses a centralized system for detecting hardware capabilities and
managing architecture-specific optimizations (AVX2, AVX-512, NEON).

### `arch_detection.h`

This header is the single point of truth for:
- **Architecture Detection**: Defines `SKETCH_ARCH_X86` and `SKETCH_ARCH_ARM64`.
- **Feature Flags**: Provides canonical flags like `SKETCH_HAS_AVX2`,
  `SKETCH_HAS_AVX512F`, and `SKETCH_HAS_NEON` based on both build-time
  configuration and target platform.
- **Target Attributes**: Defines `SKETCH_AVX2_TARGET` and other compiler-specific
  attributes used to enable SIMD instruction generation for specific functions.

All vectorized compute kernels and architecture-dependent code should include
`arch_detection.h` instead of manually checking compiler macros.

## Shared Types and Constants

- `shared_types.h`: Defines core enums like `DataType` (f32, f16, i16) and
  the `Ret` status code used throughout the project.
- `shared_consts.h`: Contains versioning and alignment constants.

## Runtime State

- `singleton.h`: Manages global process state, including the active
  `ComputeUnit` (hardware backend selection) and configuration.
- `compute_unit.h`: Encapsulates the logic for selecting the best available
  compute backend at runtime.
