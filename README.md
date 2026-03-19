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

## Core

**Performance.**
Runtime SIMD dispatch selects the best available backend — AVX-512 VNNI, AVX-512F, AVX2, or NEON — at startup, without recompilation. A single binary runs optimally across Intel, AMD, and ARM hardware.

**Reliability.**
An in-memory accumulator with a Write-Ahead Log (WAL) ensures that recent writes survive process crashes. Data is never silently lost.

**Scalability.** 
Vectors are partitioned by ID range and spread across multiple directories, allowing the dataset to grow beyond a single disk or filesystem.

**Flexibility.** 
Supported element types: f32, f16, i16. 
Supported  distance functions (L1, L2, Cosine).

---

## Quick Start / Building

### Prerequisites

Sketch2 is built and supported on **Linux only**. Other operating systems are not
supported.

| Requirement  | Minimum version | Notes                                         |
|--------------|-----------------|-----------------------------------------------|
| OS           | Linux           | Only supported platform.                      |
| C++ compiler | C++20           | GCC or Clang required. MSVC is not supported. |
| CMake        | 3.20            | Used to configure all build types.            |

On x86-64, the AVX2 / AVX-512 SIMD compute backends require GCC or Clang because
they use per-function target attributes for runtime dispatch. The build will fail
at configuration time if a non-GCC/Clang compiler is detected with those backends
enabled.

### Build types

There are three CMake build types, each with a corresponding Makefile shorthand:

| Type      | Directory    | Makefile target        | Use for                               |
|-----------|--------------|------------------------|---------------------------------------|
| Debug     | `build-dbg/` | `make` or `make build` | Day-to-day development and unit tests |
| Release   | `build/`     | `make rel`             | Performance work, benchmarks, demos   |
| Sanitizer | `build-san/` | `make san`             | Memory and UB error detection         |

### Configure & build

For each build directory, configure with CMake and then build. Replace the build directory name and `CMAKE_BUILD_TYPE` flag to match the target configuration (see the table above) before running the commands below:

```bash
cmake -S . -B build-dbg -DCMAKE_BUILD_TYPE=Debug
cmake --build build-dbg
```

Once a build directory exists you can rerun `cmake --build <dir>` as needed to compile updated sources. Use `cmake --build <dir> --target help` to list available targets (benchmarks, unit tests, demos, etc.). `JOBS=N` overrides the default parallelism.

For release or sanitizer builds change the build directory and `CMAKE_BUILD_TYPE` flag accordingly (e.g., `-B build -DCMAKE_BUILD_TYPE=Release`).

### Common targets

```
make              Build debug (default)
make rel          Build release
make san          Build with AddressSanitizer / UBSan

make test         Build debug, run unit tests
make rtest        Build release, run unit tests
make santest      Build sanitizer, run unit tests
make pytest       Run Python API tests

make demo         Build release, run bulk-load KNN demo (10M vectors, dim=256)
make bench        Build release with benchmarks enabled, run essential benchmark suite
make benchext     Run extended benchmark suite

make clean        Delete .o/.obj files from the debug build directory
```

Parallel jobs default to the number of online CPUs. Override with `JOBS=N`:

```bash
make JOBS=4
```

## Creating datasets for vlite

`vlite` expects an existing Sketch2 dataset INI file (see `src/vlite/SQLITE.md`). The dataset generator lives in the same repository: either import the Python wrapper under `src/pytest/sketch2_wrapper.py` or run the bundled demo script.

1. Add `src/pytest` to `PYTHONPATH` so Python can find the wrapper:

   ```bash
   PYTHONPATH=$(pwd)/src/pytest python - <<'PY'
   from sketch2_wrapper import Sketch2

   with Sketch2("/tmp/sketch2_workspace") as sk:
       sk.create("my_dataset", type_name="f32", dim=128, range_size=1000, dist_func="l2")
       sk.upsert(1, "1.0, 0.0, 0.0, 0.0")
       sk.merge_accumulator()
       sk.close("my_dataset")
   PY

   # point vlite at /tmp/sketch2_workspace/my_dataset.ini
   ```

   The workspace directory contains `my_dataset.ini`, the data folders, and the WAL file after the writer shuts down. Point `vlite` (or the demo's `vlite('/path/to/my_dataset.ini')`) at the INI file.

2. The helper script `src/pytest/demo.py` orchestrates this end-to-end flow: run

   ```bash
   PYTHONPATH=$(pwd)/src/pytest python src/pytest/demo.py --keep
   ```

   It keeps the generated workspace, prints its path, and leaves a `dataset.ini` you can reuse with the SQLite example.

For production demos you can also use the C API (`libsketch2.so`/`parasol`) directly; the dataset INI still lives in the workspace root with the dataset name + `.ini`.

## Current State

- Core storage and compute mechanisms are in place  
- Processing is currently brute-force (no indexing yet)  
## Initial Integrations

- Python
- SQLite

## Usage Examples

### SQLite

Load the Sketch2 extension and query an existing dataset through a virtual table:

```sql
-- Load the extension (path to libsketch2.so)
.load ./build/lib/libsketch2.so

-- Create a virtual table pointing to a Sketch2 dataset INI
CREATE VIRTUAL TABLE nn USING vlite('/path/to/my_dataset.ini');

-- Find the 5 nearest neighbors
SELECT id, distance
FROM nn
WHERE query = '1.0, 2.0, 3.0, 4.0' AND k = 5
ORDER BY distance;
```

### Python

The `Sketch2` wrapper lives under `src/pytest/sketch2_wrapper.py`. Run the snippet with `PYTHONPATH=$(pwd)/src/pytest` (or execute the demo from the previous section) so Python can import the module.

Use the `Sketch2` wrapper to manage datasets and perform searches:

```python
from sketch2_wrapper import Sketch2

# Initialize and connect to a workspace
with Sketch2("/tmp/my_workspace") as sk:
    # Create a new dataset
    sk.create("items", type_name="f32", dim=128, dist_func="l2")

    # Insert a vector
    sk.upsert(101, "1.0, 0.5, 0.2, ...")

    # Search for nearest neighbors
    results = sk.knn("1.0, 0.5, 0.2, ...", count=10)
    print(f"Nearest neighbor IDs: {results}")
```

## Project Structure

- `src/core`: The core storage engine and high-performance compute kernels (L1, L2, Cosine).
- `src/parasol`: The C API and shared library entry points for `libsketch2.so`.
- `src/vlite`: The `vlite` SQLite virtual table extension.
- `src/pytest`: Python bindings, demo scripts, and comprehensive integration tests.

## Documentation

### Architecture & Design
- **[Storage Engine](src/core/storage/README.md)**: Deep dive into the range-partitioned storage engine, WAL-backed accumulator, and binary file formats.
- **[Compute Architecture](src/core/compute/README.md)**: Details on the runtime-dispatched SIMD kernels (AVX-512, AVX2, NEON) and the specialized scan layer.
- **[Parasol C API](src/parasol/README.md)**: Overview of the C API used for dataset management, mutation, and KNN queries.
- **[Storage Design Specs](src/core/storage/DESIGN.md)**: Technical specifications for input generation, data merging, and partitioning.

### User Guides & Integration
- **[SQLite Virtual Table (vlite)](src/vlite/SQLITE.md)**: Complete guide to using Sketch2 within SQLite, including schema details and SQL examples.
- **[Compute Operations](src/core/compute/OPERATIONS.md)**: Manual for controlling and verifying the runtime compute backends via environment variables.
- **[Python Wrapper](src/pytest/SOURCE.md)**: Index of the Python integration layers and demo scripts.

### Performance & Benchmarking
- **[Benchmark Guide](src/core/compute/BENCHMARKS.md)**: Instructions for running the Google Benchmark suite and interpreting throughput results.

### Technical References
- **[Source Index](src/SOURCE.md)**: A high-level map of all source components in the repository.

## Roadmap

- Add indexing structures (IVF, PQ)  
- Expand integrations (Postgres, MySQL)  
- Explore interoperability with FAISS  

---
