#pragma once
#include "utils/shared_types.h"

namespace sketch2 {

enum class PatternType {
    Sequential,
    Random,
};

struct GeneratorConfig {
    PatternType pattern_type;
    size_t count;
    size_t min_id;
    DataType type;
    size_t dim;
};

Ret generate_input_file(const std::string& path, const GeneratorConfig& config);

} // namespace sketch2