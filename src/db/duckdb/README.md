# DuckDB Integration

Sketch2 also has a DuckDB integration, but that integration is developed in a
separate repository:

- GitHub: `https://github.com/michaelpopov/sketch2duckdb`

That separate repository exists intentionally. DuckDB extension development is
organized around DuckDB's extension template and build flow, so keeping the
extension in its own repo is the practical model encouraged by DuckDB for
developing, testing, and packaging extensions.

## Why A DuckDB Extension Exists

Sketch2 is focused on vector storage and vector search. DuckDB is strong at:

- SQL querying
- joins
- metadata filtering
- analytics

The DuckDB extension connects those responsibilities cleanly:

- Sketch2 performs nearest-neighbor search
- DuckDB provides the relational and analytical layer around that search

This makes it possible to use Sketch2 as a specialized vector engine inside a
DuckDB workflow instead of treating Sketch2 as a general-purpose database.

## What The Extension Provides

The current DuckDB integration is query-oriented. It assumes that a Sketch2
dataset already exists and exposes the read/query path inside DuckDB.

The main SQL surface is:

- `sketch2(arg)`
  Returns extension-related information such as Sketch2 version or the current
  opened dataset name.
- `sketch2_open(database_path, dataset_name)`
  Opens a Sketch2 dataset for the current DuckDB connection.
- `sketch2_knn(query_vector, k, bitset_filter_ref)`
  Returns nearest neighbors as `(id, score)`.
- `sketch2_bitset_filter(id)`
  Aggregates DuckDB ids into a Sketch2 allow-list filter that can be reused by
  `sketch2_knn`.

Supported query-vector formats include:

- Sketch2 text vectors as `VARCHAR`
- DuckDB `FLOAT[]`
- DuckDB `DOUBLE[]`
- DuckDB float/double `ARRAY`

## How It Works

The separation of responsibilities is similar in spirit to the SQLite
integration in this repo:

1. Sketch2 owns dataset files, vector parsing, filtering, and scoring.
2. DuckDB owns SQL execution, joins, metadata predicates, and query shaping.
3. The extension stores a Sketch2 handle in DuckDB connection-local state.
4. `sketch2_knn(...)` calls Sketch2 through the C API and returns rows back to
   DuckDB.
5. `sketch2_bitset_filter(...)` builds a compact Sketch2 allow-list blob from
   ids produced by DuckDB queries.

Important behavior:

- one DuckDB connection tracks one opened Sketch2 dataset at a time
- bitset filter references are connection-local and ephemeral
- the integration is currently read/query oriented
- dataset creation, staged writes, deletes, and merges still happen through
  Sketch2 itself

## Typical Usage

Open a dataset:

```sql
CALL sketch2_open('/mnt/nvme/sketch2/db', 'items');
```

Run a KNN query:

```sql
SELECT id, score
FROM sketch2_knn([1.0, 2.0, 3.0, 4.0]::FLOAT[], 5, NULL);
```

Join KNN results with DuckDB metadata:

```sql
SELECT n.id, n.score, m.title
FROM sketch2_knn([7.4, 7.4, 7.4, 7.4]::FLOAT[], 5, NULL) AS n
JOIN metadata AS m ON m.id = n.id;
```

Build an allow-list in DuckDB and push it down into Sketch2:

```sql
SELECT sketch2_bitset_filter(id)
FROM metadata
WHERE category = 'books';
```

The returned filter reference can then be passed as the third argument to
`sketch2_knn(...)`.

## Building The Extension

The DuckDB extension repository depends on an external Sketch2 build.

Before building the extension, set:

```sh
export SKETCH2_ROOT=/path/to/sketch2
```

The extension build expects the Sketch2 public header in the source tree and an
installed runtime directory for the shared library:

- `"$SKETCH2_ROOT/src/sketch2api"`
- `"$SKETCH2_ROOT/install-hwy/bin"`

Build steps and extension-specific tests live in the separate
`sketch2duckdb` repository.

## What Is Not In This Repo

This Sketch2 repository contains documentation for the DuckDB integration, but
the DuckDB extension source code, its DuckDB-specific build system, and its SQL
test suite live in the separate `sketch2duckdb` repository.

If you want:

- DuckDB extension source code
- DuckDB extension build artifacts
- SQLLogicTests for the DuckDB integration
- deployment/install instructions for DuckDB extension binaries

use the sibling repository instead.

## See Also

- `src/db/sqlite/README.md` for the SQLite integration in this repo
- For the full DuckDB extension documentation `https://github.com/michaelpopov/sketch2duckdb`
