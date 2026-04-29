# DuckDB Integration

Sketch2 has a DuckDB extension that lives in a separate repository:

- GitHub: `https://github.com/michaelpopov/sketch2duckdb`

That split is intentional. The DuckDB side follows DuckDB's extension
template, build system, packaging flow, and SQL test conventions, so it is
maintained separately from the core Sketch2 repository.

## Why This Integration Exists

Sketch2 is the vector engine:

- vector storage
- nearest-neighbor search
- named bitset filters

DuckDB is the SQL and analytics layer:

- SQL execution
- joins with relational metadata
- filtering, grouping, and analytics around vector results

Together, this lets Sketch2 handle vector search while DuckDB handles the rest
of the query plan.

## Current SQL Surface

The DuckDB extension currently exposes these functions and pragmas:

- `PRAGMA sketch2_open(database_path, dataset_name)`
  Opens a Sketch2 dataset for the current DuckDB connection.
- `PRAGMA sketch2_close`
  Closes the currently opened Sketch2 dataset for the current DuckDB
  connection.
- `sketch2_version() -> VARCHAR`
  Returns the Sketch2 library version.
- `sketch2_dataset() -> VARCHAR`
  Returns the dataset name currently opened on this connection.
- `sketch2_knn(query_vector, k, bitset_filter_name) -> TABLE(id UBIGINT, score DOUBLE)`
  Runs nearest-neighbor search in Sketch2 and returns result rows to DuckDB.
- `sketch2_bitset_filter(id, name) -> VARCHAR`
  Aggregates DuckDB ids into a persisted named Sketch2 bitset filter.
- `sketch2_bitset_load(name) -> VARCHAR`
  Validates a persisted named filter and warms Sketch2's named-filter cache.
- `sketch2_bitset_cache_remove(name) -> INTEGER`
  Removes one named filter from the process-global cache.
- `sketch2_bitset_cache_clear() -> BOOLEAN`
  Clears the process-global named-filter cache.
- `sketch2_bitset_drop(name) -> INTEGER`
  Deletes a persisted named filter. Returns `1` when something was removed and
  `0` when the name was already absent.

## Supported Query Patterns

The integration is query-oriented. It assumes the Sketch2 dataset already
exists and focuses on reading/searching it from DuckDB.

### 1. Open A Dataset And Run KNN

```sql
PRAGMA sketch2_open('/mnt/nvme/sketch2_db', 'items');

SELECT sketch2_version(), sketch2_dataset();

SELECT id, score
FROM sketch2_knn([1.0, 1.0, 1.0, 1.0]::FLOAT[], 5, NULL)
ORDER BY score, id;
```

### 2. Join Sketch2 Results With DuckDB Metadata

```sql
SELECT n.id, n.score, m.title, m.category
FROM sketch2_knn([7.4, 7.4, 7.4, 7.4]::FLOAT[], 5, NULL) AS n
JOIN metadata AS m ON m.id = n.id
ORDER BY n.score, n.id;
```

### 3. Post-Filter In DuckDB

```sql
SELECT n.id, n.score, m.category
FROM sketch2_knn([7.4, 7.4, 7.4, 7.4]::FLOAT[], 6, NULL) AS n
JOIN metadata AS m ON m.id = n.id
WHERE m.category = 'books'
ORDER BY n.score, n.id;
```

### 4. Push Metadata-Derived Filters Down Into Sketch2

Build a named filter from DuckDB rows:

```sql
SELECT sketch2_bitset_filter(id, 'books_filter')
FROM metadata
WHERE category = 'books';
```

Then reuse it in KNN:

```sql
SELECT n.id, n.score
FROM sketch2_knn([7.4, 7.4, 7.4, 7.4]::FLOAT[], 6, 'books_filter') AS n
ORDER BY n.score, n.id;
```

This is important because pushdown can change the neighbor set compared to
running an unrestricted KNN first and filtering afterward in DuckDB.

### 5. Reuse And Manage Persisted Named Filters

```sql
SELECT sketch2_bitset_load('books_filter');
SELECT sketch2_bitset_cache_remove('books_filter');
SELECT sketch2_bitset_cache_clear();
SELECT sketch2_bitset_drop('books_filter');
```

The tutorials and tests cover the full lifecycle:

- create a named filter once
- reuse it across multiple KNN queries
- explicitly warm the cache
- evict one cached filter or clear the cache entirely
- drop the persisted filter from storage

## Query Vector Input Types

`sketch2_knn` accepts these query-vector formats:

- Sketch2 text vectors as `VARCHAR`
- DuckDB `FLOAT[]`
- DuckDB `DOUBLE[]`
- DuckDB fixed-size float/double `ARRAY`

Examples:

```sql
SELECT * FROM sketch2_knn('1.0, 1.0, 1.0, 1.0', 5, NULL);
SELECT * FROM sketch2_knn([1.0, 1.0, 1.0, 1.0]::FLOAT[], 5, NULL);
SELECT * FROM sketch2_knn([1.0, 1.0, 1.0, 1.0]::DOUBLE[], 5, NULL);
```

## Important Behavior And Limits

- The opened Sketch2 dataset is connection-local. One DuckDB connection tracks
  one opened Sketch2 dataset at a time.
- `sketch2_knn` requires `PRAGMA sketch2_open(...)` to run first on that
  connection.
- `k` must be greater than `0` and at most `1,000,000`.
- `bitset_filter_name` is optional, but when provided it must be a non-empty
  string naming a persisted Sketch2 filter.
- `sketch2_bitset_filter(id, name)` requires a constant, non-NULL, non-empty
  filter name.
- `sketch2_bitset_filter` accepts unsorted ids and rejects negative ids.
- When `sketch2_bitset_filter` sees no input rows, it returns `NULL` instead of
  creating a filter.
- Named-filter cache operations act on Sketch2's process-global cache.

## What The Integration Demonstrates Today

The DuckDB tutorials in `sketch2duckdb/tutorial/` currently cover:

- basic KNN from SQL
- joining KNN results with ordinary DuckDB tables
- metadata filtering after the join
- metadata-derived filter pushdown into Sketch2
- reuse and cleanup of persisted named bitset filters

The automated tests also verify that DuckDB queries see the current state of an
already-created Sketch2 dataset, including staged writes and staged deletes
performed through Sketch2 itself.

## What The Extension Does Not Currently Provide

The DuckDB SQL surface is still read/query focused. Dataset management remains
in Sketch2 itself. In practice that means:

- dataset creation is not done from DuckDB SQL
- staged writes are not done from DuckDB SQL
- deletes are not done from DuckDB SQL
- merge/maintenance operations are not done from DuckDB SQL

Those workflows still happen through Sketch2 APIs and tools; DuckDB is used to
query the resulting datasets.

## Build And Repository Notes

The extension repository depends on a Sketch2 build outside the DuckDB repo.
Before building the extension, set:

```sh
export SKETCH2_ROOT=/path/to/sketch2
```

The DuckDB extension sources, build instructions, SQLLogicTests, Python
integration tests, and tutorial scripts all live in the separate
`sketch2duckdb` repository.

## See Also

- `src/db/sqlite/README.md` for the SQLite integration
- `https://github.com/michaelpopov/sketch2duckdb` for the DuckDB extension
  source, tests, and tutorials
