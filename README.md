# Sketch2

Sketch2 is a vector storage and compute engine built for high-throughput KNN
workloads.

It is designed as an embeddable system rather than a general-purpose database.
The project focuses on storing vector data efficiently, scanning it directly in
its stored representation, and using hardware-aware compute paths to execute
distance calculations with low overhead.

## What Sketch2 Is

Sketch2 combines:

- a custom persisted storage format for vector data
- runtime-dispatched SIMD compute backends
- integration surfaces for host applications and tools

The current system is centered on brute-force KNN over stored vectors. Support
for IFV-PQ indexes and ANN search will be implemented in the next version.

## Integration Surfaces

Sketch2 currently exposes several ways to integrate with other systems:

- a native C API through the shared runtime library
- a Python wrapper for scripting, demos, and automation
- a SQLite virtual table for SQL-based nearest-neighbor queries

This allows Sketch2 to act as a specialized vector engine inside broader
application stacks.

## Project Focus

The project emphasizes:

- storage and compute designed together
- predictable read behavior with batched write and merge flows
- support for multiple CPU backends in one binary
- integration with existing systems

It currently targets Linux and supports `f32`, `f16`, and `i16` vector data
with `l1`, `l2`, and `cos` distance functions.

## Documentation

- [Manifesto](docs/0.%20Manifesto.md)
- [Design](docs/1.%20Design.md)
- [Build](docs/2.%20Build.md)
- [Test](docs/3.%20Test.md)
- [Python Integration](docs/4.%20Python%20Integration.md)
- [SQLite Integration](docs/5.%20SQLite%20Integration.md)
- [Demo](docs/6.%20Demo.md)
