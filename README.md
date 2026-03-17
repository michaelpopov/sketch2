### Project Sketch2

## Goal

Design a system where vector data is stored and processed in a way that matches the underlying hardware—rather than adapting it to general-purpose abstractions.

In short: apply **mechanical sympathy** to vector storage and computation, end to end.

---

The phrase *“mechanical sympathy”*— applied to software by Martin Thompson — describes systems built with a deep understanding of how machines actually work.

In software, that means aligning with hardware realities: memory layout, CPU caches, vectorized execution, and I/O behavior. The goal is simple: **minimize friction, minimize wasted work, maximize flow**.

---

Sketch2 is a **vector storage engine with a built-in compute layer**, designed specifically for vector workloads.

- Data layout minimizes storage overhead and maximizes I/O throughput  
- Memory is organized for cache locality and efficient data movement  
- Vectors are aligned for SIMD execution  
- Compute units use platform-specific instructions (Intel, AMD, ARM)

Storage and computation are tightly integrated so that vector processing operates directly on data in its optimal form.

---

## Current State

- Core storage and compute mechanisms are in place  
- Processing is currently brute-force (no indexing yet)  
- Initial integrations:
  - Python  
  - SQLite  

---

## Roadmap

- Add indexing structures (IVF, PQ)  
- Expand integrations (Postgres, MySQL)  
- Explore interoperability with FAISS  

---
