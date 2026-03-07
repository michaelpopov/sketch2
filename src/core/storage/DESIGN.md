Storage Design

Stage 1.

InputGenerator
-------------------------
For development and testing purposes we need datasets. There is a dataset generator that writes
files in input data format. It can be configured to write files with different patterns.
The simple pattern contains sorted, sequentially increasing ids and vectors with floats derived 
from ids like id+0.1 First line contains metadata: type, dimension.
Example:

f32,128
1 : [ 1.1, 1.1, ... 1.1, 1.1 ]
2 : [ 2.1, 2.1, ... 2.1, 2.1 ]
...
999: [ 999.1, 999.1, ... 999.1, 999.1 ]

At the stage 1 
 - there are no "delete" lines with empty vectors
 - there are no "update" lines with ids that already exist.

Supported types: f32, f16
Dimension range: 4 .. 4096


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

|--------|-----------------------------|-------------|-------------|
  header       array of vectors          array of ids   array of
                                                        deleted ids

header is a struct DataFileHeader.
vector is data (see below).
ids is an array of u64.

|----------------------------|
      data

Interface:
    init(input_path, output_path)
    exec()

Position of id in array ids matches position of a corresponding vector
in array of vectors.

Create an instance of InputReader and init it with input_path.
Create a vector<u64>, resize it with InputReader::count() and populate with ids.
Init DataFileHeader (data_file.h)
Write output file:
  - write header
  - iterate over all data(index) in InputReader and write each vector
  - write vector of ids.


DataReader
-------------------------
A class that reads data from a data file.
See format defined for DataWriter.
Interface:
    init(path)
    type()
    dim()
    size()  vector size
    count() number of vectors
    begin() get iterator
    get(id) u8*
    at(index) u8*

Iterator
    next()
    eof()

There is a bitset of boolean flags. The size matches the number of vectors
in the file. Each vector has a corresponding bit. 0 (false) means the vector
is valid as it is. 1 (true) means the vector was modified. 

There is an additional map that points to modified values. 
     id -> modified value position
If the flag is on in the bitset, then find an uint32 value in the hash map 
by vector's id. If it is found, then use the value identifued by the offset.
If it is not found, then the vector is deleted.

Bitset and the map are populated only if a delta DataReader is provided.

Iterator skips deleted vectors by checking the bitset and look up in the map.


Scanner
-------------------------
Add an interface ICompute in compute/compute.h
It has function 
   virtual double dist(const u8*, const u8*, type, dim) = 0

Add class ComputeL1 in compute/compute_l1.h
It implements ICompute
It implements dist()
It has private functions that convert u8* to corresponding types based on type parameter.
They implement L1 distance calculation for the type.
dist() calls one of these internal functions and return result.

A class that can find K nearest neighbors in data that is provided by Data Reader.
At stage 1 it can use only the distance function L1.
This class is required for closing testing loop and making sure that data interfaces
make sense.
Interface:
    init(path)
    find(func, count, vector)
        return array of ids
        func enum of distance functions L1, L2, ...
        count a requested number of ids in a returned array
        vector u8* a query vector


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
Dataset is Owner by default. Guest mode can be enabled only via `Dataset::set_guest_mode()`.
`set_guest_mode()` fails if the accumulator contains pending updates.
Guest mode rejects `store()`, `store_accumulator()`, `merge()`, `add_vector()`, and `delete_vector()`.

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

Accumulator
-----------------
Fixed size in-memory buffer. Similar to input file in a sense that items can be added. Then the content of "accumulator"
is loaded into data file similar to load from input file.

Accumulator contains two collections:
  - std::unordered_map<uint64_t, std::vector<uint8_t>> vectors_
  - std::unordered_set<uint64_t> deleted_ids_

It has a data member uint64_t data_size_ that controls the overall size of data.
When size of all items in the collections reaches data_size no new data can be added to accumulator.

When delete_vector(id) is called, accumulator checks the content of vectors_ and remove element with that id.
When add_vector(id, data) is called, accumulator checks the content of deleted_ids_ and remove element with value id.
It is ok to replace existing element in vectors_ with a new value when add_vector() is called.

Accumulator interface:
  init(size)
  add_vector(id, data)
  delete_vector(id)
  vectors_count()
  deleted_count()

  std::vector<uint64_t> get_vector_ids() --- return a sorted vector of keys from vectors_
  std::vector<uint64_t> get_deleted_ids() --- return a sorted vector of elements from deleted_id_

  const uint8_t* get_vector(id)


WAL
-----------------------
Accumulator has a Write-Ahead Log. It is stored in a local file with extension .wal
It has a following structure:

    |--------+---------+---------+---------+------- ...   --|
      header   record    record     record    record ...
  
Header is of type BaseFileHeader.
Record consists of the following parts:
  - operation identifier 1 byte: 
       - 4 bits "vector" or "delete id"
       - 4 bits "add" or "remove"
  - uint64_t id
  - if operation is "vector" "add" then it follows by vector data

When Accumulator is created, it tries to open existing .wal file. If the file is present, Accumulator
replays activities registered in the wal and restores its state. After that the wal records are truncated.

When accumulator is updated, it writes a new record into wal:
  - when a new vector added
      - if a "delete id" needs to be removed, write "delete id" "remove" "id" record
      - write "vector" "add" "id" "data" record
  - when a new "delete id" is added
      - if a vector in accumulator needs to be removed, write "vector" "remove" "id" record
      - write "delete id" "add" "id" record.

Records to wal are written before corresponding changes are done in Accumulator in-memory data structures.

After Accumulator data is merged, the records in wal are truncated.
