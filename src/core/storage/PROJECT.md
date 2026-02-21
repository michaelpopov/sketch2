Storage Projects

Stage 1.

X Add src/core/utils directory
    Add .h file for Ret
    Add log.h

X InputGenerator
  - Create .h/.cpp files for InputGenerator class
  - Implement generating input files with simple pattern
  - Develop unit tests validating InputGenerator functionality.

X InputReader
  - Create .h/.cpp files for InputReader class
  - Implement reading input files with simple pattern
  - Develop unit tests validating InputReader functionality.

X DataWriter
  - Create .h/.cpp files for DataWriter class
  - Implement reading input files with simple pattern
  - Develop unit tests validating DataWriter functionality.

X DataReader
At the fist stage it can read data only from sealed data file. It will be extended later
to merge data from delta files. This needs to be kept in mind when designing DataReader.
  - Create .h/.cpp files for DataReader class
  - Implement reading input files with simple pattern
  - Develop unit tests validating DataReader functionality.

Scanner
  - Create .h/.cpp files for Scanner class
  - Implement access to data using DataReader
  - Implement search of K nearest neighbors using L1 distance
  - Develop unit tests validating Scanner functionality.
