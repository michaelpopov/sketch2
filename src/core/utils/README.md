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

- `shared_types.h`: Defines core enums like `DataType` (`f32`, `f16`, `i16`,
  `f8`) and the `Ret` status code used throughout the project.  Its persistent
  numeric mapping is append-only: `f16=0`, `f32=1`, `i16=2`, and `f8=3`.
- `shared_consts.h`: Contains versioning and alignment constants.

### E5M2 `f8`

`f8` permanently names the one-byte E5M2 format.  Its bit pattern is the high
byte of the corresponding IEEE binary16 (`f16`) bit pattern, so decoding an
`f8` byte is exactly equivalent to decoding `uint16_t(byte) << 8` as `f16`.
The largest finite value is `57344`.

Checked numeric conversion uses round-to-nearest, ties-to-even in two stages:
`f32 -> f16 -> f8`.  Python starts by explicitly rounding its binary64 input
to `f32`, then follows those same two stages.  Textual and numeric ingest reject
NaN, infinities, and finite values that overflow E5M2; a binary input payload
is raw stored bytes and is deliberately trusted rather than revalidated.

The usual vector display uses two decimal places for people, not for a
serialization contract.  The lossless generator and round-trip path requests
the `digits == 0` form, which uses `%.9g` so an `f8` value can be parsed back to
the same byte.

## Runtime State

- `singleton.h`: Manages global process state including configuration and the
  shared thread pool.
