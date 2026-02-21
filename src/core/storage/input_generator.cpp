#include "input_generator.h"
#include <experimental/scope>
#include <cstdint>

namespace sketch2 {

static inline void print_float_line(FILE* f, uint64_t id, size_t dim) {
    double value = static_cast<double>(id) + 0.1;
    fprintf(f, "%lu : [ ", id);
    for (size_t d = 0; d < dim; ++d) {
        if (d < dim - 1) {
            fprintf(f, "%.1f, ", value);
        } else {
            fprintf(f, "%.1f", value);
        }
    }
    fprintf(f, " ]\n");
}

static inline void print_int_line(FILE* f, uint64_t id, size_t dim) {
    fprintf(f, "%lu : [ ", id);
    for (size_t d = 0; d < dim; ++d) {
        if (d < dim - 1) {
            fprintf(f, "%lu, ", id);
        } else {
            fprintf(f, "%lu", id);
        }
    }
    fprintf(f, " ]\n");
}

Ret generate_input_file(const std::string& path, const GeneratorConfig& config) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        return Ret("Failed to open file for writing: " + path);
    }
    std::experimental::scope_exit file_guard([f]() { fclose(f); });

    if (config.pattern_type != PatternType::Sequential) {
        return Ret("Unsupported pattern type");
    }

    fprintf(f, "%s,%lu\n", to_string(config.type), config.dim);
    
    for (size_t i = 0; i < config.count; ++i) {
        uint64_t id= config.min_id + i;
        if (config.type == DataType::f32 || config.type == DataType::f16) {
            print_float_line(f, id, config.dim);
        } else if (config.type == DataType::i32) {
            print_int_line(f, id, config.dim);
        } else {
            return Ret("Unsupported data type");
        }
    }

    return Ret(0);
}

} // namespace sketch2