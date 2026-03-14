// Declares the writer that builds binary data files from input readers.

#pragma once
#include "utils/shared_types.h"
#include <string>

namespace sketch2 {

class InputReaderView;

// DataWriter exists to materialize the project's binary data-file format from
// text or binary input records. It builds headers, aligned vector sections, optional
// cosine metadata, and the sorted id/delete tables written to disk.
class DataWriter {
public:
    Ret init(const std::string& input_path, const std::string& output_path,
        uint64_t start=0, uint64_t end=0, bool write_cosine_inv_norms=false);
    Ret exec();

    Ret load(const InputReaderView& reader, const std::string& output_path,
        bool write_cosine_inv_norms = false);

private:
    std::string input_path_;
    std::string output_path_;
    uint64_t start_ = 0;
    uint64_t end_ = 0;
    bool write_cosine_inv_norms_ = false;
};

} // namespace sketch2
