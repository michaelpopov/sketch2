#include "input_generator.h"
#include <experimental/scope>
#include <cstdint>

namespace sketch2 {

static Ret generate_sequential_input_file(const std::string& path, const GeneratorConfig& config);
static Ret generate_detailed_input_file(const std::string& path, const GeneratorConfig& config);

Ret generate_input_file(const std::string& path, const GeneratorConfig& config) {
    if (config.count == 0) {
        return Ret("count must be greater than zero");
    }
    if (config.dim < 4 || config.dim > 4096) {
        return Ret("dim must be in range [4, 4096]");
    }

    switch (config.pattern_type) {
        case PatternType::Sequential: return generate_sequential_input_file(path, config);
        case PatternType::Detailed:   return generate_detailed_input_file(path, config);
        default: return Ret("Unsupported pattern type");
    }
}

template <typename T>
static inline void print_float_line(FILE* f, uint64_t id, const T* value, size_t dim, bool is_array) {
    fprintf(f, "%lu : [ ", id);
    for (size_t d = 0; d < dim; ++d) {
        const size_t index = is_array ? d : 0;
        if (d < dim - 1) {
            fprintf(f, "%.2f, ", static_cast<double>(value[index]));
        } else {
            fprintf(f, "%.2f", static_cast<double>(value[index]));
        }
    }
    fprintf(f, " ]\n");
}

static inline void print_int_line(FILE* f, uint64_t id, const int16_t* value, size_t dim, bool is_array) {
    fprintf(f, "%lu : [ ", id);
    for (size_t d = 0; d < dim; ++d) {
        const size_t index = is_array ? d : 0;
        if (d < dim - 1) {
            fprintf(f, "%d, ", value[index]);
        } else {
            fprintf(f, "%d", value[index]);
        }
    }
    fprintf(f, " ]\n");
}

static Ret generate_sequential_input_file(const std::string& path, const GeneratorConfig& config) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        return Ret("Failed to open file for writing: " + path);
    }
    std::experimental::scope_exit file_guard([f]() { fclose(f); });

    fprintf(f, "%s,%lu\n", data_type_to_string(config.type), config.dim);
    
    for (size_t i = 0; i < config.count; ++i) {
        uint64_t id= config.min_id + i;
        if (config.type == DataType::f32) {
            float value = static_cast<float>(id) + 0.1;
            print_float_line(f, id, &value, config.dim, false);
        } else if (config.type == DataType::f16) {
            float16 value = static_cast<float16>(id) + 0.1;
            print_float_line(f, id, &value, config.dim, false);
        } else if (config.type == DataType::i16) {
            int16_t value = static_cast<int16_t>(id);
            print_int_line(f, id, &value, config.dim, false);
        } else {
            return Ret("Unsupported data type");
        }
    }

    return Ret(0);
}

static Ret generate_detailed_input_file(const std::string& path, const GeneratorConfig& config) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        return Ret("Failed to open file for writing: " + path);
    }
    std::experimental::scope_exit file_guard([f]() { fclose(f); });

    fprintf(f, "%s,%lu\n", data_type_to_string(config.type), config.dim);

    if (config.type == DataType::f32) {
        InputVector<float> v(config.dim, static_cast<float>(config.max_val));
        for (size_t i = 0; i < config.count; ++i) {
            uint64_t id= config.min_id + i;
            print_float_line(f, id, v.data(), config.dim, true);
            v.next();
        }
    } else if (config.type == DataType::f16) {
        InputVector<float16> v(config.dim, static_cast<float>(config.max_val));
        for (size_t i = 0; i < config.count; ++i) {
            uint64_t id= config.min_id + i;
            print_float_line(f, id, v.data(), config.dim, true);
            v.next();
        }
    } else if (config.type == DataType::i16) {
        InputVector<int16_t> v(config.dim, static_cast<int16_t>(config.max_val));
        for (size_t i = 0; i < config.count; ++i) {
            uint64_t id= config.min_id + i;
            print_int_line(f, id, v.data(), config.dim, true);
            v.next();
        }
    } else {
        throw std::runtime_error("generate_detailed_input_file: invalid data type");
    }


    return Ret(0);
}

} // namespace sketch2
