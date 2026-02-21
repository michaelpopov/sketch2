#include "input_reader.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <cstdlib>
#include <stdexcept>

namespace sketch2 {

InputReader::~InputReader() {
    if (map_) {
        munmap(const_cast<uint8_t*>(map_), map_len_);
    }
}

/*
Instructions:
map file into memory,
parse the first line to get the data type and dimension,
parse the rest of the lines to get the vector id and byte offset,
*/
Ret InputReader::init(const std::string& path) {
    // Map file into memory
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return Ret("Failed to open file: " + path);
    }
    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return Ret("Failed to stat file: " + path);
    }
    map_len_ = static_cast<size_t>(st.st_size);
    if (map_len_ == 0) {
        close(fd);
        return Ret("File is empty: " + path);
    }
    void* m = mmap(nullptr, map_len_, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) {
        return Ret("Failed to mmap file: " + path);
    }
    map_ = static_cast<const uint8_t*>(m);

    const char* p   = reinterpret_cast<const char*>(map_);
    const char* end = p + map_len_;

    // Parse the first line: "{type},{dim}\n"
    const char* comma = static_cast<const char*>(memchr(p, ',', static_cast<size_t>(end - p)));
    if (!comma) {
        return Ret("Invalid header: missing comma");
    }
    std::string type_str(p, static_cast<size_t>(comma - p));
    if      (type_str == "f32") type_ = DataType::f32;
    else if (type_str == "f16") type_ = DataType::f16;
    else if (type_str == "i32") type_ = DataType::i32;
    else return Ret("Unknown data type: " + type_str);

    char* dim_end;
    dim_ = static_cast<size_t>(strtoull(comma + 1, &dim_end, 10));
    if (dim_end == comma + 1) {
        return Ret("Invalid header: missing dimension");
    }

    if (size() < sizeof(uint64_t)) {
        return Ret("Invalid header: vector data size is too small");
    }

    buf_.resize(size()); // pre-allocate buffer for one vector, used for parsing

    // Advance past the header newline
    const char* line = dim_end;
    while (line < end && *line != '\n') ++line;
    if (line < end) ++line;

    // Parse each vector line: "{id} : [ {data...} ]\n"
    while (line < end) {
        char* id_end;
        uint64_t id = strtoull(line, &id_end, 10);
        if (id_end == line) break; // blank or trailing content

        const char* bracket = static_cast<const char*>(
            memchr(id_end, '[', static_cast<size_t>(end - id_end)));
        if (!bracket) {
            return Ret("Invalid line: missing '['");
        }

        // offset points to the character after "[" (first number)
        uint64_t offset = static_cast<uint64_t>(bracket + 1 - p);

        lines_.push_back({id, offset});

        const char* nl = static_cast<const char*>(
            memchr(id_end, '\n', static_cast<size_t>(end - id_end)));
        if (!nl) break;
        line = nl + 1;
    }

    return Ret(0);
}

size_t InputReader::count() const {
    return lines_.size();
}

DataType InputReader::type() const {
    return type_;
}

size_t InputReader::dim() const {
    return dim_;
}

size_t InputReader::size() const {
    return dim_ * to_size(type_);
}

uint64_t InputReader::id(size_t index) const {
    if (index >= lines_.size()) {
        throw std::out_of_range("InputReader::id: index out of range");
    }
    return lines_[index].id;
}

static uint16_t float_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint16_t sign     = static_cast<uint16_t>((x >> 16) & 0x8000);
    int      exp      = static_cast<int>((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = x & 0x7FFFFF;
    if (exp <= 0)  return sign;
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00);
    return static_cast<uint16_t>(sign | (exp << 10) | (mantissa >> 13));
}

/*
Instructions:
Given an index, get a pointer to the text of vector data for that index.
Parse the vector data into binary format and store it in buf_ based on the data type and dimension,
then return a pointer to the parsed data in buf_.
*/
const uint8_t* InputReader::data(size_t index) const {
    if (index >= lines_.size()) {
        throw std::out_of_range("InputReader::data: index out of range");
    }
    const char* p = reinterpret_cast<const char*>(map_) + lines_[index].offset;

    if (type_ == DataType::f32) {
        float* out = reinterpret_cast<float*>(buf_.data());
        for (size_t d = 0; d < dim_; ++d) {
            char* next;
            out[d] = strtof(p, &next);
            p = next;
            while (*p == ',' || *p == ' ') ++p;
        }
    } else if (type_ == DataType::i32) {
        uint32_t* out = reinterpret_cast<uint32_t*>(buf_.data());
        for (size_t d = 0; d < dim_; ++d) {
            char* next;
            out[d] = static_cast<uint32_t>(strtoull(p, &next, 10));
            p = next;
            while (*p == ',' || *p == ' ') ++p;
        }
    } else if (type_ == DataType::f16) {
        uint16_t* out = reinterpret_cast<uint16_t*>(buf_.data());
        for (size_t d = 0; d < dim_; ++d) {
            char* next;
            float f = strtof(p, &next);
            out[d] = float_to_f16(f);
            p = next;
            while (*p == ',' || *p == ' ') ++p;
        }
    }

    return buf_.data();
}

/*
Instructions:
If the line for the given index contains only no characters or whitespace between [], return true.
Otherwise return false.
Add check of index bounds. Throw exception if index is out of bounds.
*/
bool InputReader::is_no_data(size_t index) const {
    if (index >= lines_.size()) {
        throw std::out_of_range("InputReader::is_no_data: index out of range");
    }
    const char* p   = reinterpret_cast<const char*>(map_) + lines_[index].offset;
    const char* end = reinterpret_cast<const char*>(map_) + map_len_;
    while (p < end && *p != ']' && *p != '\n') {
        if (*p != ' ' && *p != '\t') {
            return false;
        }
        ++p;
    }
    return true;
}

} // namespace sketch2
