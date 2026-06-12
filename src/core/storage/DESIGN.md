Storage Design

Stage 1.

InputFormat
-------------------------
Data is loaded into Sketch2 storage from input files.
There are four possible formats of these files:
  - text with vector values delimited by coma
  - text with vector values delimited by space
  - binary with vector values in binary format matching vector data type
  - indexed binary with vector values in binary format matching vector data type.

The first line of the input file is always in text format. It is finished with
end-of-line symbol '\n'. This line contains information about vector data type,
vector dimensions and optional indicators for binary and indexed binary format.
For example:
f32,128\n
f32,128,bin\n
f32,128,binind\n

Supported data types: f32, f16, i16
Supported dimensions range: 4 .. 4096

Text format consists of id and vector value delimited by colon ':'
id : value
  where id is uint64_t in text representation
        value is a set of numbers in square brackets [ ... ]

Empty vector value [] indicates that vector with this id is deleted.

Example of text format with comma-separated values:
f32,128
1 : [ 1.1, 1.1, ... 1.1, 1.1 ]
2 : [ 2.1, 2.1, ... 2.1, 2.1 ]
3 : []
...
999: [ 999.1, 999.1, ... 999.1, 999.1 ]

Example of text format with space-separated values:
f32,128
1 : [ 1.1 1.1, ... 1.1 1.1 ]
2 : [ 2.1 2.1, ... 2.1 2.1 ]
3 : []
...
999: [ 999.1 999.1 ... 999.1 999.1 ]

Binary format consists of sequence of pairs id and value, where id is uint64_t
number and value is a set of numbers of a corresponding data type. The count
of numbers in a value matches vector's dimensions.

 |------|-------------------------------|
    id        vector value

Binary format cannot contain information about deleted vectors.

To overcome this limitation there is an indexed binary format.
Indexed binary format consists of blocks 64 items, where each item
can be either uint64_t id or a pair of uint64_t id and vector value
similar to binary format.
In front of each block, there are 8 bytes containing a bitset describing
the following data:
 - if a bit is set to 1, then the corresponding item consists only of
   uint64_t id indicating deleted vector id
 - if a bit is set to 0, the  the corresponding item consists of
   uint64_t id and vector value.

In the end of each full block there is a control footer consisting of:
 - uint32_t counter equal to the index of the last record in the block
 - uint32_t CRC32 checksum of the preceding full block payload

The counter values should be like 64, 128, 192, 256, etc.

 |-------|------|-------------------------------|------|------|-------------------------------|--------|--------|
  bitset   id        vector value                  id     id      vector value                  counter  crc32

The control footers are used for checking the correctness of file format and detecting corruption
of binary data. The control footers are added after each full block of exactly 64 items. If in the
end there is incomplete block of items, there is no footer to check.

The CRC32 checksum covers the whole preceding full block payload, starting from
the 64-bit bitset at the beginning of the block and ending with the last byte
of the 64th item. The footer bytes themselves are not included in the checksum.

InputGenerator
-------------------------
For development and testing purposes we need datasets. There is a dataset generator that writes
files in input data format. It can be configured to write files with different patterns.


InputReader
-------------------------
There is a class that provides access to data in an input file.
It maps file to memory.
It loads metadata.
It scans the whole file and loads information about each line.
struct LineInfo
    u64 id
    u64 offset // where the first number of vector data starts
vector<LineInfo> info
It implements interface:
    init(path)
    count()  number of lines
    type()
    dim()
    data(index)   u8* to vector<u8> that contains parsed vector
    size()  u8 size of a vector. sizeof(type) * dim

Functions that can fail return struct Ret defined in utils directory


DataWriter
-------------------------
A class that gets generates sealed data file based on the content of InputReader.
Format:

|--------|------------------------------------------|--------------------|---------------------|
  header       aligned vector records with optional    frozen RoaringIds     frozen RoaringIds
               inline stored norm                      active ids           deleted ids

header is a struct DataFileHeader.
Each vector record stores the vector payload plus optional inline norm data in the same stride-sized slot.
ids are stored as frozen RoaringIds trailers. Active ids and deleted ids are
stored in separate trailers, both encoded as uint32 offsets from `min_range_id`.
For cosine datasets each active record also stores an inline `f32` inverse norm, `1.0 / ||vector||`.
Zero vectors store `0.0`.
The values are intentionally stored as `f32` instead of `f64` to keep each record compact and
cheap to stream during scans. This is a precision/performance tradeoff: cosine distances that use
the stored inline value can differ slightly from recomputing norms in `double`, especially for near-ties.
For files managed as part of a cosine dataset, Sketch2 requires this inline value to be present on every
persisted data/delta record.

|----------------------------|
      data

Interface:
    init(input_path, output_path)
    exec()

The position of an active id in the active RoaringIds set matches the position
of the corresponding vector record.

Create an instance of InputReader and init it with input_path.
Init DataFileHeader (data_file.h)
Write output file:
  - write header
  - iterate over all data(index) in InputReader and write each vector record
  - for cosine datasets, compute one inverse norm per active vector and store it inline in that record
  - write frozen RoaringIds trailers for active ids and deleted ids.


DataReader
-------------------------
A class that reads data from a data file or a data file with an attached delta
file. See format defined for DataWriter.
Interface:
    init(path)
    type()
    dim()
    size()  vector size
    count() number of vectors
    begin() get iterator
    get_norm(index) f32 stored inline norm for the matching active vector; throws when inline norms are absent
    get(id) u8*
    at(index) u8*

Iterator
    next()
    eof()

The reader memory-maps vector records and frozen RoaringIds trailers directly
from the file. The active ids trailer supports id lookup, positional lookup, and
ordered iteration. The deleted ids trailer records tombstones.

When a delta DataReader is attached, the base reader builds a hidden-row bitset
whose size matches the base active-id count. A bit is set when the delta either
deletes the base id or provides a newer active row for the same id. Base
iteration skips hidden rows, and delta rows are exposed from the delta reader.
There is no per-id hash map in the current design.

For files with stored norm metadata, the reader exposes the inline stored norm
for each visible vector.


Scanner
-------------------------
The query path now lives under `core/compute`.

`Scanner` is the main facade used by higher-level code. It delegates a query
to the Highway compute layer and then runs one specialized path for:

  - distance function
  - vector element type
  - backend-specific kernel set

The scanner implementation lives in:

  - `highway.cpp` for the Highway-backed build

Shared scanner helpers are split by responsibility instead of being folded into
one monolithic implementation. For example:

  - dataset traversal and visible-row handling live in `scanner_dataset_scan.h`
  - hot scan loops live in `scanner_scan_loops.h`
  - heap/result extraction helpers live in `scanner_heap_utils.h`

Public scanner operations return either ids or `(id, score)` pairs and clear
caller-provided output buffers before reporting a failure so failed queries do
not leave stale results behind.

For cosine datasets scanner precomputes the query norm once per search. If a data file
or accumulator entry contains stored inline cosine inverse norms, scanner uses:

    cosine_distance = 1 - dot(a, q) * inv_norm(a) * inv_norm(q)

That avoids recomputing the stored-vector norm inside the hot scan loop.
Because persisted inverse norms are stored inline as `f32`, this fast path may produce slightly different
results than recomputing norms in `double`. The expected benefit is lower storage cost and lower
per-candidate work in cosine-heavy scans.
Cosine datasets reject persisted files that do not contain inline inverse norms.

For L2 datasets scanner likewise precomputes the query squared norm once per search. If a data file
or accumulator entry contains stored inline squared norms, scanner uses:

    l2_distance_sq = norm_sq(a) + norm_sq(q) - 2 * dot(a, q)

That avoids recomputing the stored-vector squared norm inside the hot scan loop.
Because persisted squared norms are stored inline as `f32`, this fast path may produce slightly different
results than recomputing norms in `double`. The expected benefit is lower per-candidate work in
L2-heavy scans.
L2 datasets reject persisted files that do not contain inline squared norms.


Dataset
-----------------------
This class controls access to files. It contains meta data about files locations.
Metadata:
  - set of directories where data files are located
  - range of ids per data file
Dataset is initialized either manually by setting metadata parameters or it can
load the values from config.ini file, which is formatted as ini file.

Dataset implements 
  - init(const vector<string>& paths, uint64_t size), where paths are directories for data files
       and size defines a range of ids per data file, for example size=1000 item with id=123 goes to file 0
       and item with id 2100 goes to file 2.
  - store(path), where path is a path of input data file
       check the ids in the input of the file
       create data file for each range of ids
       each data file goes to its directory, which is defined by (file_id % number_of_dirs)
       for each data file create DataWriter and write the file with data from the input file
       each DataWriter writes only the items that belong to its range

Dataset can run in two modes: Owner and Guest.
As Owner a dataset can make modifications in the data. As Guest a dataset can only query data.
Dataset is Owner by default. Guest mode can be enabled via `Dataset::set_guest_mode()`, which checks for pending writes before switching.
Guest mode rejects write operations such as `store()` and `merge()`.

Dataset caches opened files after they are accessed. The following access operations do not require
scanning directories, looking for files and opening them again.

Dataset allows checking whether a specific deleted id exists in its buffered state.
This functionality is used in Scanner: function find_ that gets Dataset ref as an argument checks
if an id was deleted and skips adding it to the heap.

Dataset supports an iterator over buffered vectors. This iterator is used
in the Scanner when it scans Dataset data.

DataMerger
---------------------------
We have two types of files:
  - data files
  - delta files.

Data files contain the bulk of data. Delta files contain changes that need to be applied to data files.
For example: data file 123.data and its delta file 123.delta.
There are following possible cases:
  1) no files at all - neither data file nor delta file
  2) there is a data file but no delta file
  3) there is a data file and delta file

Transitions between these case:
  1 -> 2 -> 3 // No data in the beginning, new data file is created, new updates are written into a delta file
       |
       + -> 2 // Instead of writing updates to a delta file, data file and and updates are merged into a new data file
       |
       + -> 3 -> 3 // Updates are written to a delta file, then new updates are merged into the delta file
       |
       + -> 3 -> 2 // Updates are written into a delta file, then the delta file is merged with a data file.

"No files at all" is the intitial state. After data is written at least once, it does not reappear.
"Only data file" is the state after the first time the data is written.
    The following updates can result in maintaining this state if the volume of updates is "close" to the volume of data
    in the data file. In this case the updates are merged into data file without generating delta file.
    If a delta files grows to the size "close" to the data file, then a delta file is merged into data file, which results
    in the "only data file" state again.
"Data file and delta file" is the state after some updates for the data file is written into a delta file.

The logic of making merging decisions is in Dataset::store() function.
The merge functionality is in the DataMerger class.


UpdateNotifier
--------------------------
There are two different parts of the system: one writes data periodically, another one processes queries.
They might be executed in different processes.
The writer can write data while readers continue reading because data/delta files are immutable.
But the reader caches open files to reuse them on following queries.
There is a need to flush this cache when data is updated.
Class UpdateNotifier in utils library.
It is running in two modes: (1) updater mode, (2) checker mode.
The writer process call UpdateNotifier::update() function after it completes changes in the data/delta files.
The reader process before processing query calls UpdateNotifier::check_updated() function. 
This function returns true if data/delta files were changed and the reader flushes cache before processing a query.
Implementation:
(1) In updater mode, the writer calls UpdateNotifier::init_updater(). This function either creates a file if it doesn't
exist or opens file in RW mode and reads 8-bytes uint64_t number from the file. It stores this number in its data member.
In UpdateNotifier::update() it increments the number, writes it into the file and fdatasync() the file.
(2) In checker mode, the reader calls UpdateNotifier::check_updated().
If a file is not opened, the function opens the file, reads 8-byte number, stores it in data member and returns true.
If a file is already opened, the function reads the 8-byte number again and compares it to the stored value. If values
are equal, it returns false. Otherwise, it sets a data member to a new value and returns true.
There is a lock file used for acquiring lock on a dataset. Use this file for storing data for this mechanism.
UpdateNotifier is a data member of Dataset. std::unique_ptr<UpdateNotifier>
In Dataset functions store() and merge()
  - if UpdateNotifier is not initialized, initialize it as updater
  - Call UpdateNotifier::update()
In Dataset function reader()
  - if UpdateNotifier is not initialized, initialize it as checker
  - Call UpdateNotifier::check_updated() and flush the cache if needed.

Multiprocess/Multithreaded Support
---------------------------------------------
The intention is to have the system running in two different processes:
  - Writer periodically loads new batches of data into datasets
  - Reader running multiple concurrent queries.

There is an UpdateNotifer that allows Reader to learn about data updates by Writer and adjust Reader's caches.
Writer is not supposed to be multithreaded but let's have "paranoid" protection that ensures it doesn't crash or corrupt
data even if somebody initiates write operations from multiple threads on the same dataset.
Reader is supposed to be multithreaded: multiple queries on the same dataset can run concurrently. It requires careful
protection of mutable data used in queries.

In this system it is ok to see slightly stale data. If query completes on data version that existed 10 millisecond ago, it's fine. 
The crucial point of multithreading/multiprocess support is preventing system crashes. If the system completes query successfully
without seeing data that was added after the query started, it is a correct behavior.

DatasetNode
----------------------------------------------
The system used to have "a God object" called Dataset. It could read and write data for a dataset. In order to keep
things manageable I split this functionality to three classes:
 - Dataset - a base class that can handle initialization
 - DatasetReader - a class that inherrits from Dataset and can handle all functionality related to reading data
 - DatasetWriter - a class that inherrits from Dataset and can handle all functionality related to writing data.

Now we have clear responsibility for each class.
We lost some functionality, which is acceptable. For example, it was possible to read buffered pending changes
before they are persisted to data/delta files. It is not possible anymore and that's by design: DatasetReader can only
read data persisted in data/delta files. Period.

On the other hand, there are some parts of the system that depend on having an object that can provide read
and write functionality. For example, the Sketch2api library supports both types of functionality. There is a large
number of unit tests and integration tests that need both types of functionality.

In order to support test scenarios and Sketch2api functionality, let's introduce a new class DatasetNode.
It is declared and implemented in storage/dataset_node.h storage/dataset_node.cpp.
It has two private data members
   std::unique_ptr<DatasetReader> reader_;
   std::unique_ptr<DatasetWriter> writer_;
It has public functions init(...) that allows initializing internal reader_ and writer_.
It exposes public functions required for test scenarios and Sketch2api functionality. These functions
call corresponding functions of reader_ and writer_.

Then we replace DatasetWriter usage in unit and integration tests and in Sketch2api with using DatasetNode,
which provides all functionality required they require.
