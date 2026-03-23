# Sketch2

Sketch2 is an open-source, embeddable vector engine built for fast, large-scale similarity search.
It’s designed to live inside your application — not as a separate service — giving you direct control over how vectors are stored, scanned, and computed.

Instead of treating vector search like a database problem, Sketch2 focuses on what matters most: moving less data and doing less work per query. It stores vectors in a compact, compute-friendly format and executes distance calculations using hardware-aware paths for maximum throughput.

💫 Why Sketch2?

- **High-Throughput by Design**  
  Optimized for workloads where you need to evaluate large candidate sets quickly — not just index and pray.
- **Embeddable & Lightweight**  
  No servers, no orchestration. Link it into your system and run vector search where your data already lives.
- **Compute-Centric Architecture**  
  Designed around efficient scanning and distance computation, minimizing memory movement and overhead.
- **Storage Meets Execution**  
  Vectors aren’t just stored — they’re laid out for fast, direct processing without expensive transformations.
- **Built for Control**  
  Ideal for systems where you want to own the full query pipeline, from filtering to scoring.

🚀 Where it fits

Sketch2 is a great fit for:

- Custom search and ranking systems
- Recommendation engines
- Retrieval pipelines (RAG, semantic search)
- High-performance analytics over embeddings
- Systems where tight integration and performance matter more than turnkey features.

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

| Document | Path |
| --- | --- |
| Manifesto | docs/0.%20Manifesto.md |
| Design | docs/1.%20Design.md |
| Build | docs/2.%20Build.md |
| Test | docs/3.%20Test.md |
| Python Integration | docs/4.%20Python%20Integration.md |
| SQLite Integration | docs/5.%20SQLite%20Integration.md |
| Demo | docs/6.%20Demo.md |
