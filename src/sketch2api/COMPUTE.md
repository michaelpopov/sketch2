# Sketch2 Compute Configuration

This document describes the Highway-based compute model used by `sketch2api`.

## Build-Time Model

Sketch2 builds use the Highway compute path for KNN queries.

## Runtime Behavior

There is no runtime setting for the top-level compute path.

`SKETCH2_CONFIG` and the other startup settings still control runtime concerns
such as logging and thread-pool sizing, but they do not change the Highway path
used for query execution.

## Hardware Specialization Inside The Highway Runtime

Highway still specializes execution for the host CPU inside that runtime:

- the build uses ISA-specialized kernels where available
- runtime dispatch still picks the best supported ISA on the host CPU

So there is still runtime CPU specialization inside Highway, but no
separate top-level compute implementation to switch between.

## Verifying The Active Runtime

Set `SKETCH2_LOG_LEVEL=INFO` to see what happened during initialization.

Examples:

```text
[INFO] Compute backend set to 'highway' because this build always uses the Highway backend.
```
