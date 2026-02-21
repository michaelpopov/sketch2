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

|--------|-----------------------------|-------------|
  header       array of vectors          array of ids

header is a struct DataFileHeader.
vector is data (see below).
ids is an array of u64.

Vector consists of a header u64 and data.
Header contains vector id.

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

There is a vector of boolean flags. The size matches the number of vectors
in the file. Each vector has a corresponding bit. 0 (false) means the vector
is valid as it is. 1 (true) means the vector was modified.

There are two modes:
   - in place
   - reference

If the mode is "in place" then "modified" vector means it is deleted.
If the mode is "reference" then it is required to look at the content of the vector,
the first 8 bytes converted to u8*:
   - nullptr means the vector is deleted
   - otherwise it's a pointer to a new value of the vector.

Iterator skips deleted vectors by checking the bitset and the value of the vector.

The bitset is optional. If it is not present, there are no modifications to the content
of the file.


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

