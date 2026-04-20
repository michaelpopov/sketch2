// Declares the writer that builds binary data files from input readers.

#pragma once
#include "utils/shared_types.h"
#include <string>

namespace sketch2 {

class InputReaderView;
class DatasetWriter;

// DataWriter exists to materialize the project's binary data-file format from
// text or binary input records. It builds headers, aligned vector sections, optional
// norms metadata, and the sorted id/delete tables written to disk.
class DataWriter {
public:
    Ret exec_for_testing(const std::string& input_path, const std::string& output_path,
        uint64_t min_range_id, uint64_t start = 0, uint64_t end = 0, DistFunc dist_func = DistFunc::DOT);

    Ret write(const InputReaderView& reader, const std::string& output_path, DistFunc dist_func,
        uint64_t min_range_id);
};

} // namespace sketch2
