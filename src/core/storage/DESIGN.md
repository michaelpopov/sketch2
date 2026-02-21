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
vector is u64 + data (see below).
ids is an array of u64.

Vector consists of a header u64 and data.
Header contains vector id.

|-------|-------------------------|
   u64      data

Interface:
    init(const InputReader&, path)
    exec()

Position of id in array ids matches position of a corresponding vector
in array of vectors.


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



Scanner
-------------------------
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

