// Implements utilities for generating textual input files used by tests and demos.

#include "input_generator.h"
#include "core/utils/shared_consts.h"
#include "core/utils/singleton.h"
#include "core/utils/thread_pool.h"
#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <experimental/scope>
#include <fcntl.h>
#include <future>
#include <limits>
#include <sys/stat.h>
#include <sys/mman.h>
#include <thread>
#include <type_traits>
#include <unistd.h>

namespace sketch2 {

namespace {

constexpr size_t kBinarySequentialChunkSize = 10000;

Ret make_io_error(const std::string& action, const std::string& path) {
    return Ret(action + ": " + path + ": " + std::strerror(errno));
}

Ret make_temp_output_path(const std::string& path, std::string* temp_path) {
    if (temp_path == nullptr) {
        return Ret("temporary path output is null");
    }

    std::string pattern = path + ".tmp.XXXXXX";
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');

    const int fd = mkstemp(buffer.data());
    if (fd < 0) {
        return make_io_error("Failed to create temporary file", path);
    }
    std::experimental::scope_exit fd_guard([fd]() { close(fd); });

    if (fchmod(fd, 0666) != 0) {
        return make_io_error("Failed to set temporary file permissions", buffer.data());
    }

    *temp_path = buffer.data();
    return Ret(0);
}

template <typename Writer>
Ret write_file_atomically(const std::string& path, Writer&& writer) {
    std::string temp_path;
    CHECK(make_temp_output_path(path, &temp_path));

    std::experimental::scope_exit cleanup([&temp_path]() {
        if (!temp_path.empty()) {
            std::remove(temp_path.c_str());
        }
    });

    const Ret ret = writer(temp_path);
    if (ret.code() != 0) {
        return ret;
    }

    if (std::rename(temp_path.c_str(), path.c_str()) != 0) {
        return make_io_error("Failed to replace output file", path);
    }

    temp_path.clear();
    return Ret(0);
}

template <typename T>
void write_binary_sequential_range(uint8_t* records_base, const GeneratorConfig& config,
        size_t record_size, size_t begin, size_t end) {
    std::vector<T> payload(config.dim);
    for (size_t i = begin; i < end; ++i) {
        const uint64_t id = config.min_id + i;
        uint8_t* record = records_base + i * record_size;
        std::memcpy(record, &id, sizeof(id));

        if constexpr (std::is_same_v<T, float>) {
            const float value = static_cast<float>(id) + 0.1f;
            std::fill(payload.begin(), payload.end(), value);
        } else if constexpr (std::is_same_v<T, float16>) {
            const float16 value = static_cast<float16>(static_cast<float>(id) + 0.1f);
            std::fill(payload.begin(), payload.end(), value);
        } else {
            const T value = static_cast<T>(id);
            std::fill(payload.begin(), payload.end(), value);
        }
        std::memcpy(record + sizeof(id), payload.data(), payload.size() * sizeof(T));
    }
}

template <typename T>
Ret fill_binary_sequential_records(uint8_t* records_base, const GeneratorConfig& config) {
    const size_t record_size = sizeof(uint64_t) + config.dim * sizeof(T);
    const size_t chunk_count = (config.count + kBinarySequentialChunkSize - 1) / kBinarySequentialChunkSize;
    const auto& thread_pool = get_singleton().thread_pool();
    if (chunk_count <= 1 || !thread_pool) {
        write_binary_sequential_range<T>(records_base, config, record_size, 0, config.count);
        return Ret(0);
    }

    std::vector<std::future<void>> futures;
    futures.reserve(chunk_count);
    for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
        const size_t begin = chunk * kBinarySequentialChunkSize;
        const size_t end = std::min(config.count, begin + kBinarySequentialChunkSize);
        futures.push_back(thread_pool->submit([records_base, &config, record_size, begin, end] {
            write_binary_sequential_range<T>(records_base, config, record_size, begin, end);
        }));
    }

    for (auto& future : futures) {
        try {
            future.get();
        } catch (const std::exception& e) {
            return Ret(e.what());
        } catch (...) {
            return Ret("binary sequential generator failed");
        }
    }

    return Ret(0);
}

Ret generate_sequential_input_file_binary_mmap(const std::string& path, const GeneratorConfig& config) {
    const std::string header =
        std::string(data_type_to_string(config.type)) + "," + std::to_string(config.dim) + ",bin\n";
    const size_t type_size = data_type_size(config.type);
    const size_t max_size = std::numeric_limits<size_t>::max();
    if (type_size == 0 || config.dim > (max_size - sizeof(uint64_t)) / type_size) {
        return Ret("binary input record size overflow");
    }

    const size_t record_size = sizeof(uint64_t) + config.dim * type_size;
    if (config.count > (max_size - header.size()) / record_size) {
        return Ret("binary input file size overflow");
    }
    const size_t file_size = header.size() + config.count * record_size;

    const int fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (fd < 0) {
        return make_io_error("Failed to open file for writing", path);
    }
    std::experimental::scope_exit fd_guard([fd]() { close(fd); });

    if (ftruncate(fd, static_cast<off_t>(file_size)) != 0) {
        return make_io_error("Failed to size file", path);
    }

    void* mapped = mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        return make_io_error("Failed to map file", path);
    }

    uint8_t* map = static_cast<uint8_t*>(mapped);
    std::experimental::scope_exit map_guard([&map, file_size]() {
        if (map != nullptr) {
            munmap(map, file_size);
        }
    });

    std::memcpy(map, header.data(), header.size());
    uint8_t* records_base = map + header.size();

    Ret fill_ret(0);
    if (config.type == DataType::f32) {
        fill_ret = fill_binary_sequential_records<float>(records_base, config);
    } else if (config.type == DataType::f16) {
        fill_ret = fill_binary_sequential_records<float16>(records_base, config);
    } else if (config.type == DataType::i16) {
        fill_ret = fill_binary_sequential_records<int16_t>(records_base, config);
    } else {
        return Ret("Unsupported data type");
    }
    if (fill_ret.code() != 0) {
        return fill_ret;
    }

    if (msync(map, file_size, MS_SYNC) != 0) {
        return make_io_error("Failed to flush mapped file", path);
    }
    if (munmap(map, file_size) != 0) {
        return make_io_error("Failed to unmap file", path);
    }
    map = nullptr;

    if (fsync(fd) != 0) {
        return make_io_error("Failed to sync file", path);
    }

    return Ret(0);
}

} // namespace

static Ret generate_sequential_input_file(const std::string& path, const GeneratorConfig& config);
static Ret generate_detailed_input_file(const std::string& path, const GeneratorConfig& config);
static Ret generate_sequential_input_file_binary(const std::string& path, const GeneratorConfig& config);
static Ret generate_detailed_input_file_binary(const std::string& path, const GeneratorConfig& config);
static Ret generate_manual_input_file(const std::string& path, const ManualInputGenerator& gen);

Ret generate_input_file(const std::string& path, const GeneratorConfig& config) {
    if (config.count == 0) {
        return Ret("count must be greater than zero");
    }
    if (config.dim < kMinDimension || config.dim > kMaxDimension) {
        return Ret("dim must be in range [" + std::to_string(kMinDimension) +
            ", " + std::to_string(kMaxDimension) + "]");
    }
    try {
        validate_type(config.type);
    } catch (const std::exception& e) {
        return Ret(e.what());
    }
    if (config.binary && config.every_n_deleted > 0) {
        return Ret("binary input format does not support deleted items");
    }

    return write_file_atomically(path, [&config](const std::string& temp_path) -> Ret {
        if (config.binary) {
            switch (config.pattern_type) {
                case PatternType::Sequential: return generate_sequential_input_file_binary(temp_path, config);
                case PatternType::Detailed:   return generate_detailed_input_file_binary(temp_path, config);
            }

            return Ret("unsupported binary pattern type");
        }

        switch (config.pattern_type) {
            case PatternType::Sequential: return generate_sequential_input_file(temp_path, config);
            case PatternType::Detailed:   return generate_detailed_input_file(temp_path, config);
        }

        return Ret("generate_input_file: invalid pattern type");
    });
}

template <typename T>
static inline void print_float_line(FILE* f, uint64_t id, const T* value, size_t dim, bool is_array) {
    fprintf(f, "%" PRIu64 " : [ ", id);
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
    fprintf(f, "%" PRIu64 " : [ ", id);
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

static inline void print_deleted_line(FILE* f, uint64_t id) {
    fprintf(f, "%" PRIu64 " : []\n", id);
}

template <typename T>
static Ret write_binary_record(FILE* f, uint64_t id, const T* value, size_t dim, bool is_array) {
    if (fwrite(&id, sizeof(id), 1, f) != 1) {
        return Ret("Failed to write binary id");
    }

    if (is_array) {
        if (fwrite(value, sizeof(T), dim, f) != dim) {
            return Ret("Failed to write binary vector data");
        }
        return Ret(0);
    }

    for (size_t d = 0; d < dim; ++d) {
        const T component = value[0];
        if (fwrite(&component, sizeof(component), 1, f) != 1) {
            return Ret("Failed to write binary vector data");
        }
    }

    return Ret(0);
}

// Writes predictable test input where each id maps to a repeated scalar value,
// with optional tombstones inserted at a fixed cadence.
static Ret generate_sequential_input_file(const std::string& path, const GeneratorConfig& config) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        return Ret("Failed to open file for writing: " + path);
    }
    std::experimental::scope_exit file_guard([f]() { fclose(f); });

    fprintf(f, "%s,%zu\n", data_type_to_string(config.type), config.dim);
    
    for (size_t i = 0; i < config.count; ++i) {
        uint64_t id= config.min_id + i;

        if (i > 0 && config.every_n_deleted > 0 && i % config.every_n_deleted == 0) {
            print_deleted_line(f, id);
            continue;
        }

        if (config.type == DataType::f32) {
            float value = static_cast<float>(id) + 0.1;
            print_float_line(f, id, &value, config.dim, false);
        } else if (config.type == DataType::f16) {
            const float value_f32 = static_cast<float>(id) + 0.1f;
            const float16 value = static_cast<float16>(value_f32);
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

// Writes test input with per-dimension variation by advancing an InputVector
// generator after each emitted record, again allowing periodic deleted entries.
static Ret generate_detailed_input_file(const std::string& path, const GeneratorConfig& config) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        return Ret("Failed to open file for writing: " + path);
    }
    std::experimental::scope_exit file_guard([f]() { fclose(f); });

    fprintf(f, "%s,%zu\n", data_type_to_string(config.type), config.dim);

    if (config.type == DataType::f32) {
        InputVector<float> v(config.dim, static_cast<float>(config.max_val));
        for (size_t i = 0; i < config.count; ++i) {
            uint64_t id= config.min_id + i;
            if (i > 0 && config.every_n_deleted > 0 && i % config.every_n_deleted == 0) {
                print_deleted_line(f, id);
            } else {
                print_float_line(f, id, v.data(), config.dim, true);
                v.next();
            }
        }
    } else if (config.type == DataType::f16) {
        InputVector<float16> v(config.dim, static_cast<float16>(config.max_val));
        for (size_t i = 0; i < config.count; ++i) {
            uint64_t id= config.min_id + i;
            if (i > 0 && config.every_n_deleted > 0 && i % config.every_n_deleted == 0) {
                print_deleted_line(f, id);
            } else {
                print_float_line(f, id, v.data(), config.dim, true);
                v.next();
            }
        }
    } else if (config.type == DataType::i16) {
        InputVector<int16_t> v(config.dim, static_cast<int16_t>(config.max_val));
        for (size_t i = 0; i < config.count; ++i) {
            uint64_t id= config.min_id + i;
            if (i > 0 && config.every_n_deleted > 0 && i % config.every_n_deleted == 0) {
                print_deleted_line(f, id);
            } else {
                print_int_line(f, id, v.data(), config.dim, true);
                v.next();
            }
        }
    } else {
        return Ret("generate_detailed_input_file: invalid data type");
    }


    return Ret(0);
}

// Writes a text header followed by binary records made of uint64_t ids and
// packed vector payloads with a repeated scalar value per dimension.
static Ret generate_sequential_input_file_binary(const std::string& path, const GeneratorConfig& config) {
    return generate_sequential_input_file_binary_mmap(path, config);
}

// Writes a text header followed by binary records that use the InputVector
// pattern to vary individual dimensions across successive items.
static Ret generate_detailed_input_file_binary(const std::string& path, const GeneratorConfig& config) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        return Ret("Failed to open file for writing: " + path);
    }
    std::experimental::scope_exit file_guard([f]() { fclose(f); });

    fprintf(f, "%s,%zu,bin\n", data_type_to_string(config.type), config.dim);

    if (config.type == DataType::f32) {
        InputVector<float> v(config.dim, static_cast<float>(config.max_val));
        for (size_t i = 0; i < config.count; ++i) {
            const uint64_t id = config.min_id + i;
            CHECK(write_binary_record(f, id, v.data(), config.dim, true));
            v.next();
        }
    } else if (config.type == DataType::f16) {
        InputVector<float16> v(config.dim, static_cast<float16>(config.max_val));
        for (size_t i = 0; i < config.count; ++i) {
            const uint64_t id = config.min_id + i;
            CHECK(write_binary_record(f, id, v.data(), config.dim, true));
            v.next();
        }
    } else if (config.type == DataType::i16) {
        InputVector<int16_t> v(config.dim, static_cast<int16_t>(config.max_val));
        for (size_t i = 0; i < config.count; ++i) {
            const uint64_t id = config.min_id + i;
            CHECK(write_binary_record(f, id, v.data(), config.dim, true));
            v.next();
        }
    } else {
        return Ret("generate_detailed_input_file_binary: invalid data type");
    }

    return Ret(0);
}

// Writes manually specified ids and deletion markers using the same textual
// input format consumed by InputReader.
static Ret generate_manual_input_file(const std::string& path, const ManualInputGenerator& gen) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        return Ret("Failed to open file for writing: " + path);
    }
    std::experimental::scope_exit file_guard([f]() { fclose(f); });

    fprintf(f, "%s,%zu\n", data_type_to_string(gen.type), gen.dim);

    for (const auto& [id, val] : gen.items) {
        if (!val) {
            print_deleted_line(f, id);
            continue;
        }

        if (gen.type == DataType::f32) {
            float value = static_cast<float>(id) + 0.1;
            print_float_line(f, id, &value, gen.dim, false);
        } else if (gen.type == DataType::f16) {
            const float value_f32 = static_cast<float>(id) + 0.1f;
            const float16 value = static_cast<float16>(value_f32);
            print_float_line(f, id, &value, gen.dim, false);
        } else if (gen.type == DataType::i16) {
            int16_t value = static_cast<int16_t>(id);
            print_int_line(f, id, &value, gen.dim, false);
        } else {
            return Ret("Unsupported data type");
        }
    }

    return Ret(0);
}

Ret generate_input_file(const std::string& path, const ManualInputGenerator& gen) {
    if (gen.dim < kMinDimension || gen.dim > kMaxDimension) {
        return Ret("dim must be in range [" + std::to_string(kMinDimension) +
            ", " + std::to_string(kMaxDimension) + "]");
    }
    try {
        validate_type(gen.type);
    } catch (const std::exception& e) {
        return Ret(e.what());
    }

    return write_file_atomically(path, [&gen](const std::string& temp_path) -> Ret {
        return generate_manual_input_file(temp_path, gen);
    });
}

} // namespace sketch2
