# Sketch2 Compute Configuration

This document describes the build-selected compute model used by `sketch2api`.

## Build-Time Model

Sketch2 does not ship a runtime-switchable mix of top-level compute engines in
one binary.

Each configured build selects exactly one top-level engine through
`SKETCH2_COMPUTE_ENGINE`:

- `highway`
- `numkong`

That compiled engine is what `sketch2api` uses for KNN queries.

## Runtime Behavior

There is no runtime setting for the top-level compute engine.

`SKETCH2_CONFIG` and the other startup settings still control runtime concerns
such as logging and thread-pool sizing, but they do not change the engine used
for query execution.

## Hardware Specialization Inside The Compiled Engine

The top-level engine is fixed per build, but hardware specialization still
happens inside that engine:

- Highway builds use Google Highway multi-target dispatch to pick the active ISA
- NumKong builds resolve capability-specific kernels inside the NumKong backend

So there is still runtime CPU specialization, just not runtime switching
between `highway` and `numkong`.

## Verifying The Active Engine

Set `SKETCH2_LOG_LEVEL=INFO` to see what happened during initialization.

Examples:

```text
[INFO] Compute backend set to 'highway' because it is the default calc engine.
```
