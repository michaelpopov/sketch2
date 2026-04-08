# Sketch2 Compute Configuration

This document describes the three primary compute options available in `sketch2api` and provides instructions on how to configure them.

## Overview

The `sketch2` compute layer is responsible for high-speed score calculations. It supports three distinct implementations (engines) that can be selected at runtime to optimize performance for different hardware and use cases.

## The Three Compute Options

### 1. Google Highway (`highway`)
Built on the [Google Highway](https://github.com/google/highway) library, this engine provides a portable SIMD abstraction.
*   **Best for**: Cross-platform compatibility and consistent performance across diverse architectures.
*   **Configuration**: Set the engine to `highway`.

### 2. NumKong (`numkong`)
A specialized numerical kernel library optimized for high-performance vector operations.
*   **Best for**: Maximum performance on supported f32 and f16 workloads (L2 and Cosine metrics).
*   **Configuration**: Set the engine to `numkong`.

### 3. Custom SIMD Functions
Hand-written, project-specific kernels optimized for various CPU architectures. These provide direct access to specific instruction sets.
*   **Best for**: Targeted optimization and cases where generic libraries may not provide the desired instruction-level control.
*   **Available Backends**:
    *   `avx512_vnni`: AVX-512 with Vector Neural Network Instructions.
    *   `avx512f`: AVX-512 Foundation instructions.
    *   `avx2`: Intel/AMD AVX2 with FMA and F16C support.
    *   `neon`: ARM NEON for AArch64 (Apple Silicon, AWS Graviton).
    *   `scalar`: Standard non-vectorized implementation (fallback).
*   **Configuration**: Set the engine to the specific backend name (e.g., `avx2`, `neon`). This selects `ScannerEx(CalcEngine::compute)`, which then uses the matching custom SIMD backend at runtime.

---

## Configuration Instructions

You can configure the compute engine using either an INI configuration file or environment variables.

### 1. Using the INI Configuration File
In your `sketch2.ini` file, add or modify the `[compute]` section:

```ini
[compute]
# Options: highway, numkong, avx2, avx512f, avx512_vnni, neon, scalar, auto
engine=numkong
```

Point to your configuration file using the `SKETCH2_CONFIG` environment variable:
```bash
export SKETCH2_CONFIG=/path/to/your/sketch2.ini
```

### 2. Using Environment Variables
Environment variables take the highest precedence and override settings in the INI file.

*   **`SKETCH2_COMPUTE_ENGINE`**: Set this to one of the engine or backend names.
    ```bash
    # Example: Select the AVX2 custom SIMD backend
    export SKETCH2_COMPUTE_ENGINE=avx2

    # Example: Select Google Highway
    export SKETCH2_COMPUTE_ENGINE=highway
    ```

### 3. Automatic Selection (`auto`)
If set to `auto`, `sketch2` probes the host CPU and selects the highest-performing supported backend among the **Custom SIMD functions**. In `sketch2api`, `auto` means `ScannerEx(CalcEngine::compute)` is used with automatic backend selection inside the custom SIMD runtime.

## Summary of Configuration Precedence
1.  **`SKETCH2_COMPUTE_ENGINE`** environment variable.
2.  **`engine`** setting in the `[compute]` section of the INI file.
3.  **Automatic Detection** (Default).

## Runtime Dispatch Summary

- `highway` selects `ScannerEx(CalcEngine::highway)`.
- `numkong` selects `ScannerEx(CalcEngine::numkong)`.
- `auto`, `avx2`, `avx512f`, `avx512_vnni`, `neon`, and `scalar` select `ScannerEx(CalcEngine::compute)`.
- Invalid values log an `ERROR` message and degrade to the default compute path instead of terminating the host process.

## Verifying the Active Engine
Set the `SKETCH2_LOG_LEVEL` environment variable to `INFO` to see which engine was selected during initialization:
```text
[INFO] Compute engine set to 'highway' because configuration explicitly requested it.
[INFO] Compute backend set to 'avx512_vnni' because auto-detected AVX-512 VNNI.
```
