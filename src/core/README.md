# core


Split functionality to multiple libraries to allow focusing on specific funcationality and
doing proper unit testing.

## storage

Data formats.
Writing data.
Loading data and accessing records.
Splitting to multiple files.
Managing these files.

## membuf

Managing memory buffers.
Control layer of loading files defined in "storage".

## compute

All the processing related to vector math.
Optimizations.
SIMD.
GPU.

## control

Getting it all together into coherrent system capable of processing
commands, queries, jobs like "writer" and "indexer".


