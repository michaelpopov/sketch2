# storage

Assumptions:
1. There will be initial load of a large volume of data.
2. After that there will be more data added periodically.
3. Deleting or updating existing data records are relatively rare operations.
4. Data added, updated or deleted in batches that contain multiple vectors.
5. There is no "point updates", i.e. multiple frequent operations to add, delete or update specific records.

Ideas:
1. Store most data in inmutable files "data files".
2. There are many such files. This allows processing data in parts and doing this in parallel.
3. Each record is identified in the input data by a 64-bit unsigned integer "vector id".
   This number links the vector to a record with metadata for this entity.
4. After vector is stored in the database, it can be accessed by
   - vector id
   - record id, that consists of the file id (file that contains the vector) and position in the file.
5. After vector with a certain vector id is stored in the specific data file, it always will be stored
   in the data file with the same file id.
6. Access by vector id will require following steps:
    - search a data file that contains the vector
    - search the id in the file and corresponding position inside the file.
7. Access by record id is a direct path to data file by its file_id and the vector by its position.
8. Changes in data files are implemented by maintaining "delta files".
9. These "delta files" are much smaller than "data files". They contain records about deleted vectors, updated
   records with their updated vector data.
10. When a "delta file" grows above a certain threshold, it is merged with a "data file". In this case a new
   version of "data file" is generated.
11. When "data files" are loaded for processing, "delta files" are also loaded and dynamically merged with the
   in-memory copy of the data file.
12. "Loader" performs loading and merging of data files and delta files. After that "processor" can see up-to-date
   data loaded into memory.
13. "Writer" processes input data in batches and generates new "data files" and new updated versions of "delta files".
14. Data is inserted/updated in batches. Each batch processing job:
    - creates new data files;
    - creates new delta files or new versions of existing delta files;
    - creates a new version of a catalogue with information about new data files and new/versioned delta files.
15. While data is processed by "writer", all "readers" continue processing data using existing version of catalogue and
  existing data/delta files. When query starts it is linked to the existing version of catalogue and existing versions
  of delta files. Thus "reader" doing processing for the query see consistent data. After newer catalogue is generated
  and stored, future "query processors" will use this new version of catalogue.
16. The same principle regarding index. Information about current version of index is stored in index catalogue.
17. There is a "garbage collector" that deletes older versions of catalogues, delta files and older versions of data files.
18. Bottom line: we pay the price in overutilization of storage but we have an uninterruped query processing while
  "writer" and "indexer" can do their jobs in the background.

Main File Types:
1. Data file
2. Delta file
3. Catalogue

Temporary Work Files:
1. Sample data file.
2. Residuals data file.

IVF Index File Types:
1. Index catalogue.
2. Cluster map file - For each data file, it contains cluster id for each record in the data file.
3. Centroids file.
4. PQ Centroids file.
5. PQ data file - For each data file, it contains PQ versions of the original vectors.
