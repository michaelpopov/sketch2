Parasol
======================
Parasol is a dynamic library that can be used by Python scritps to get access to Sketch2 functionality.

Parasol supports the following functions:

1. connect
    Functionality: prepares database root directory, meaning ensures that it exists or created.
    Arguments: db_path string - database root path used in other commands.
    Result: handler that can be used in other commands.

2. disconnect 
    Functionality: releases all resources associated with the handler
    Arguments: handler
    Result: void

3. create
    Functionality: in the database root directory creates dataset's ini file and lock file, 
                   creates dataset's directory
    Arguments:  handler obtained in connect
                name string - used as a <name>.ini file and <name>.lock files, <db_root>/name 
                  data directory. Ini file has "dataset" section for values.
                dim int - value written into ini file, validate it's positive
                type string - value written into ini file, validate it's one of "f16", "f32", "i16",
                  check "f16" is acceptable on the platform
                range int - value written into ini file, validate it's more than 10
                Ini file also gets a string value "dirs" with path to data directory.
    Result: 0 success, -1 failure, handler has error code and error message set

4. drop
    Functionality:  closes open dataset and releases all its resources
                    deletes ini file, lock file, data directory.
    Arguments: handler, name of the dataset (same as in "create" function)
    Result: 0 success, -1 failure, handler has error code and error message set

5. open
    Functionality: opens a dataset and makes it ready for data manipulation and queries.
    Arguments:  handler
                name string (same as in "create")
    Result: 0 success, -1 failure, handler has error code and error message set
    If another dataset was opened on the handler and not closed, the function fails.

6. close
    Functionality:  closes open dataset and releases all its resources
    Arguments: handler, name of the dataset (same as in "create" function)
    Result: 0 success, -1 failure, handler has error code and error message set
    If no dataset was opened on the handler, the function fails.

7. upsert
    Functionality: adds vector and id to dataset
    Arguments:  handler,
                id uint64_t
                vector string in format "[ val, val, ... ]"
    Result: 0 success, -1 failure, handler has error code and error message set
    If no dataset was opened on the handler, the function fails.

7. ups2
    Functionality: adds vector and id to dataset
    Arguments:  handler,
                id uint64_t
                vector as a number, which is used to construct vector having dim copies for the number
    Result: 0 success, -1 failure, handler has error code and error message set
    If no dataset was opened on the handler, the function fails.

8. del
    Functionality: delete vector by id
    Arguments:  handler
                id uint64_t
    Result: 0 success, -1 failure, handler has error code and error message set
    If no dataset was opened on the handler, the function fails.

9. knn 
    Functionality: finds K nearest neighbors for the vector
        Clears previous knn result if it was there.
    Arguments:  handler
                vector string in format "[ val, val, ... ]"
                K int
    Result: 0 success, -1 failure, handler has error code and error message set
        Set of ids is stored on the handler and accessed using other
    If no dataset was opened on the handler, the function fails.

10. kres
    Functionality: queries knn result stored on handler
    Arguments:  handler
                index
                    -1  returns number of stored ids
                    N   returns Nth value in the array of stored ids
    Result: 0 success, -1 failure, handler has error code and error message set
    If no knn result was stored, the function fails.


11. macc
    Functionality: merge dataset accumulator
    Arguments:  handler
    Result: 0 success, -1 failure, handler has error code and error message set

12. mdelta
    Functionality: merge delta file
    Arguments:  handler
    Result: 0 success, -1 failure, handler has error code and error message set

13. get
    Functionality: gets vector value
        Clears previous get result if it was there.
    Arguments:  handler
                id uint64_t
    Result: 0 success, -1 failure, handler has error code and error message set
        Vector value is stored on the handler.

14. gres
    Functionality: gets vector value stored on the handler
    Arguments:  handler
    Result: string with vector

15. gid
    Functionality: gets vector id by vector value
    Arguments:  handler
                vector string in format "[ val, val, ... ]"
    Result: 0 success, -1 failure, handler has error code and error message set
        Id value is stored on handler

15. ires
    Functionality: gets vector id stored on handler
    Arguments:  handler
    Result: uint64_t

16. print
    Functionality: iterates through all vectors in dataset and prints them in format
                        id : [ v1, v2, ... , vdim]\n
                    to stdout.
    Arguments:  handler
    Result: 0 success, -1 failure, handler has error code and error message set

17. generate
    Functionality:  vectors input file with vectors and loads them into current dataset.
    Arguments:  handler
                count
                start_id
                PatternType
    Result: 0 success, -1 failure, handler has error code and error message set

18. stats
    Functionality:  prints info for each data and delta file and accumulator.
        Prints to stdout in the following format:
            <accumulator|file name>:\n
                Vectors count\n
                Deleted count\n
                \n
    Arguments:  handler
    Result: 0 success, -1 failure, handler has error code and error message set
