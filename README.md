# Sketch2

Sketch2 is a vector storage and compute engine focused on high-throughput, SIMD-accelerated KNN workloads.

## What It Provides

- Range-partitioned vector storage with low-overhead binary formats
- Runtime SIMD backend dispatch (AVX-512 VNNI, AVX-512F, AVX2, NEON, scalar)
- WAL-backed accumulator for crash-safe recent writes
- Multi-file / multi-directory scaling by id range
- Integrations:
  - SQLite virtual table extension (`vlite`)
  - Python wrapper used by integration tests and demos

Storage and compute are designed together so search runs directly against the stored representation.

## Current Scope

- Brute-force KNN search (no ANN index yet)
- Supported vector element types: `f32`, `f16`, `i16`
- Supported distance functions: `l1`, `l2`, `cos`
- Linux-only build and runtime support

## Build Requirements

| Requirement | Minimum | Notes                   |
|-------------|---------|-------------------------|
| OS          | Linux   | Only supported platform |
| Compiler    | C++20   | GCC or Clang            |
| CMake       | 3.20    | Build system            |


## Build Quick Start

Debug build:

```bash
cmake -S . -B build-dbg -DCMAKE_BUILD_TYPE=Debug
cmake --build build-dbg
```

Release build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Sanitizer build:

```bash
cmake -S . -B build-san -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-san
```

## Makefile Shortcuts

```text
make              Build debug (default)
make rel          Build release
make san          Build sanitizer build

make test         Build debug and run C++ tests
make rtest        Build release and run C++ tests
make santest      Build sanitizer and run C++ tests
make pytest       Run Python integration tests
make cover        Run broader validation/coverage flow

make demo         Run release demo workload
make bench        Run benchmark suite
make benchext     Run extended benchmark suite

make clean        Clean debug build objects
```

## Usage

### Python

`Sketch2` wrapper location: `src/pytest/sketch2_wrapper.py`

Run with `PYTHONPATH=$(pwd)/src/pytest`:

```python
from sketch2_wrapper import Sketch2

with Sketch2("/tmp/my_workspace") as sk:
    # 1) Create and open a dataset
    sk.create("items", type_name="f32", dim=4, range_size=1000, dist_func="l2")

    # 2) Ingest vectors
    sk.upsert(100, "0.0, 0.0, 0.0, 0.0")
    sk.upsert(101, "1.0, 1.0, 1.0, 1.0")
    sk.upsert(102, "2.0, 2.0, 2.0, 2.0")
    sk.upsert(103, "3.0, 3.0, 3.0, 3.0")

    # 3) Update and delete
    sk.upsert(101, "1.1, 1.1, 1.1, 1.1")  # overwrite id=101
    sk.delete(103)                         # tombstone id=103

    # 4) Flush accumulator to storage files
    sk.merge_accumulator()

    # 5) Run KNN
    query = "1.2, 1.2, 1.2, 1.2"
    ids = sk.knn(query, count=3)
    print("top-3 ids:", ids)

    # 6) Fetch exact stored vector text for a returned id
    if ids:
        print("vector for best match:", sk.get(ids[0]))

    # 7) Close
    sk.close("items")
```

### SQLite (`vlite`)

```sql
.load /absolute/path/to/libsketch2.so
CREATE VIRTUAL TABLE nn USING vlite('/absolute/path/to/dataset.ini');

SELECT id, distance
FROM nn
WHERE query = '1.0, 2.0, 3.0, 4.0' AND k = 5
ORDER BY distance;
```

`vlite` also supports optional `allowed_ids` filtering via bitset BLOBs produced
by `bitset_agg(id)`. See `src/vlite/SQLITE.md` and `src/vlite/BITSET.md`.

#### Join KNN Results With Metadata

Store metadata in a regular SQLite table and join by vector id:

```sql
CREATE TABLE items_meta (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT
);

SELECT m.title, m.category
FROM nn AS n
JOIN items_meta AS m ON m.id = n.id
WHERE n.query = '1.0, 2.0, 3.0, 4.0' AND n.k = 5
ORDER BY n.distance;
```

Filter by metadata first, then join with KNN:

```sql
SELECT m.id, m.title, n.distance
FROM nn AS n
JOIN items_meta AS m ON m.id = n.id
WHERE n.match_expr MATCH '0.1, 0.2, 0.3, 0.4'
  AND n.k = 20
  AND m.category = 'books'
ORDER BY n.distance
LIMIT 10;
```

#### Push a list of "allowed_ids" to the KNN search

Use `bitset_agg(id)` from a relational table to constrain candidates:

```sql
SELECT n.id, n.distance
FROM nn AS n
WHERE n.match_expr MATCH '2.1, 2.1, 2.1, 2.1'
  AND n.k = 10
  AND n.allowed_ids = (
        SELECT bitset_agg(id)
        FROM items_meta
        WHERE category = 'books'
      )
ORDER BY n.distance;
```

## Repository Layout

- `src/core` — storage engine and compute kernels
- `src/parasol` — C API and shared-library entry points
- `src/vlite` — SQLite virtual table extension
- `src/pytest` — Python wrapper, demos, integration tests

## Documentation Index

### Architecture And Design

- [Storage Engine](src/core/storage/README.md)
- [Compute Architecture](src/core/compute/README.md)
- [Storage Design Specs](src/core/storage/DESIGN.md)

### APIs And Integrations

- [Parasol C API](src/parasol/README.md)
- [SQLite `vlite` Guide](src/vlite/SQLITE.md)
- [Bitset Filter Format](src/vlite/BITSET.md)
- [Python Integration Source Index](src/pytest/SOURCE.md)

### Performance

- [Benchmark Guide](src/core/compute/BENCHMARKS.md)

### Source Navigation

- [Top-Level Source Index](src/SOURCE.md)

## Roadmap

- Add ANN indexing (IVF, PQ)
- Expand database integrations (e.g. PostgreSQL, MySQL)
- Evaluate interoperability with FAISS
