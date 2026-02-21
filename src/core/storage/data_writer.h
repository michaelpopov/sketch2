#pragma once
#include "utils/shared_types.h"
#include <string>

namespace sketch2 {

class DataWriter {
public:
    Ret init(const std::string& input_path, const std::string& output_path);
    Ret exec();

private:
    std::string input_path_;
    std::string output_path_;
};

} // namespace sketch2
