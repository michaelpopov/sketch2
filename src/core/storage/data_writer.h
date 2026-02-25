#pragma once
#include "utils/shared_types.h"
#include <string>

namespace sketch2 {

class DataWriter {
public:
    Ret init(const std::string& input_path, const std::string& output_path,
        uint64_t start=0, uint64_t end=0);
    Ret exec();

private:
    std::string input_path_;
    std::string output_path_;
    uint64_t start_;
    uint64_t end_;
};

} // namespace sketch2
