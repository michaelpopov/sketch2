# Sketch2 Compute Configuration

This document describes the build-selected compute model used by `sketch2api`.

## Build-Time Model

Sketch2 does not ship a runtime-switchable mix of top-level compute paths in
one binary.

Each configured build contains one native compute path, and that is what
`sketch2api` uses for KNN queries.

## Runtime Behavior

There is no runtime setting for the top-level compute path.

`SKETCH2_CONFIG` and the other startup settings still control runtime concerns
such as logging and thread-pool sizing, but they do not change the path used
for query execution.

## Hardware Specialization Inside The Compiled Runtime

The top-level path is fixed per build, but hardware specialization still
happens inside that runtime:

- the build uses ISA-specialized kernels where available
- runtime dispatch still picks the best supported ISA on the host CPU

So there is still runtime CPU specialization, just not runtime switching
between alternate top-level implementations.

## Verifying The Active Runtime

Set `SKETCH2_LOG_LEVEL=INFO` to see what happened during initialization.

Examples:

```text
[INFO] Compute backend initialized for the active native runtime path.
```
