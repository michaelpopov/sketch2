// Implements parsing and formatting helpers for textual vector values.

#include "string_utils.h"
#include "../../third-party/fast_float/fast_float.h"
#include <cmath>
#include <cstdio>
#include <limits>
#include <string.h>

namespace sketch2 {

namespace {

constexpr fast_float::chars_format kFastFloatFormat =
    fast_float::chars_format::general |
    fast_float::chars_format::allow_leading_plus |
    fast_float::chars_format::skip_white_space;

bool parse_float_token(const char* line, const char* end, float& value, const char*& next) {
    const auto result = fast_float::from_chars(line, end, value, kFastFloatFormat);
    next = result.ptr;
    return result.ec == std::errc() && next != line && next <= end;
}

} // namespace

// Parses a textual vector payload into the typed binary buffer expected by the
// storage and compute layers. The function validates buffer size, token count,
// numeric ranges, and non-finite inputs so callers can treat success as fully parsed data.
Ret parse_vector(uint8_t* buf, size_t size, DataType type, uint16_t dim, const char* line, const char* end) {
    if (buf == nullptr || line == nullptr) {
        return Ret("parse_vector: invalid arguments");
    }

    if (end == nullptr) {
        end = line + strlen(line);
    }

    if (type == DataType::f32) {
        float* out = reinterpret_cast<float*>(buf);
        if (size < dim * sizeof(*out)) {
            return Ret("InputReader::data: invalid input buffer size");
        }
        for (size_t d = 0; d < dim; ++d) {
            if (line >= end) {
                return Ret("InputReader::data: truncated vector payload");
            }
            const char* next;
            if (!parse_float_token(line, end, out[d], next)) {
                return Ret("InputReader::data: invalid f32 token");
            }
            if (!std::isfinite(out[d])) {
                return Ret("InputReader::data: non-finite f32 token");
            }
            line = next;
            while (line < end && (*line == ',' || *line == ' ')) ++line;
        }
    } else if (type == DataType::i16) {
        int16_t* out = reinterpret_cast<int16_t*>(buf);
        if (size < dim * sizeof(*out)) {
            return Ret("InputReader::data: invalid input buffer size");
        }
        for (size_t d = 0; d < dim; ++d) {
            if (line >= end) {
                return Ret("InputReader::data: truncated vector payload");
            }
            char* next;
            const long value = strtol(line, &next, 10);
            if (next == line || next > end) {
                return Ret("InputReader::data: invalid i16 token");
            }
            if (value < static_cast<long>(std::numeric_limits<int16_t>::min()) ||
                value > static_cast<long>(std::numeric_limits<int16_t>::max())) {
                return Ret("InputReader::data: i16 token out of range");
            }
            out[d] = static_cast<int16_t>(value);
            line = next;
            while (line < end && (*line == ',' || *line == ' ')) ++line;
        }
    } else if (type == DataType::f16) {
        float16* out = reinterpret_cast<float16*>(buf);
        if (size < dim * sizeof(*out)) {
            return Ret("InputReader::data: invalid input buffer size");
        }
        for (size_t d = 0; d < dim; ++d) {
            if (line >= end) {
                return Ret("InputReader::data: truncated vector payload");
            }
            const char* next;
            float f;
            if (!parse_float_token(line, end, f, next)) {
                return Ret("InputReader::data: invalid f16 token");
            }
            if (!std::isfinite(f)) {
                return Ret("InputReader::data: non-finite f16 token");
            }
            out[d] = static_cast<float16>(f);
            if (!std::isfinite(static_cast<double>(out[d]))) {
                return Ret("InputReader::data: non-finite f16 token");
            }
            line = next;
            while (line < end && (*line == ',' || *line == ' ')) ++line;
        }
    }

    // Any non-separator payload after parsing dim values is malformed.
    while (line < end && (*line == ',' || *line == ' ')) ++line;
    if (line != end) {
        return Ret("InputReader::data: extra tokens in vector payload");
    }

    return Ret(0);
}

// Same as parse_vector() but expects space-separated values instead of
// comma-separated ones. Commas are treated as invalid token characters.
Ret parse_vector_spaces(uint8_t* buf, size_t size, DataType type, uint16_t dim, const char* line, const char* end) {
    if (buf == nullptr || line == nullptr) {
        return Ret("parse_vector_spaces: invalid arguments");
    }

    if (end == nullptr) {
        end = line + strlen(line);
    }

    if (type == DataType::f32) {
        float* out = reinterpret_cast<float*>(buf);
        if (size < dim * sizeof(*out)) {
            return Ret("InputReader::data: invalid input buffer size");
        }
        for (size_t d = 0; d < dim; ++d) {
            if (line >= end) {
                return Ret("InputReader::data: truncated vector payload");
            }
            const char* next;
            if (!parse_float_token(line, end, out[d], next)) {
                return Ret("InputReader::data: invalid f32 token");
            }
            if (!std::isfinite(out[d])) {
                return Ret("InputReader::data: non-finite f32 token");
            }
            line = next;
            while (line < end && *line == ' ') ++line;
        }
    } else if (type == DataType::i16) {
        int16_t* out = reinterpret_cast<int16_t*>(buf);
        if (size < dim * sizeof(*out)) {
            return Ret("InputReader::data: invalid input buffer size");
        }
        for (size_t d = 0; d < dim; ++d) {
            if (line >= end) {
                return Ret("InputReader::data: truncated vector payload");
            }
            char* next;
            const long value = strtol(line, &next, 10);
            if (next == line || next > end) {
                return Ret("InputReader::data: invalid i16 token");
            }
            if (value < static_cast<long>(std::numeric_limits<int16_t>::min()) ||
                value > static_cast<long>(std::numeric_limits<int16_t>::max())) {
                return Ret("InputReader::data: i16 token out of range");
            }
            out[d] = static_cast<int16_t>(value);
            line = next;
            while (line < end && *line == ' ') ++line;
        }
    } else if (type == DataType::f16) {
        float16* out = reinterpret_cast<float16*>(buf);
        if (size < dim * sizeof(*out)) {
            return Ret("InputReader::data: invalid input buffer size");
        }
        for (size_t d = 0; d < dim; ++d) {
            if (line >= end) {
                return Ret("InputReader::data: truncated vector payload");
            }
            const char* next;
            float f;
            if (!parse_float_token(line, end, f, next)) {
                return Ret("InputReader::data: invalid f16 token");
            }
            if (!std::isfinite(f)) {
                return Ret("InputReader::data: non-finite f16 token");
            }
            out[d] = static_cast<float16>(f);
            if (!std::isfinite(static_cast<double>(out[d]))) {
                return Ret("InputReader::data: non-finite f16 token");
            }
            line = next;
            while (line < end && *line == ' ') ++line;
        }
    }

    // Any non-separator payload after parsing dim values is malformed.
    while (line < end && *line == ' ') ++line;
    if (line != end) {
        return Ret("InputReader::data: extra tokens in vector payload");
    }

    return Ret(0);
}

// Returns true if the range [line, end) contains at least one comma character.
bool check_comma_format(const char* line, const char* end) {
    if (line == nullptr) {
        return false;
    }
    if (end == nullptr) {
        end = line + strlen(line);
    }
    for (; line < end; ++line) {
        if (*line == ',') {
            return true;
        }
    }
    return false;
}

// Reads the entire contents of file_path into vec, replacing any existing
// content. Trailing newlines are stripped so callers can pass vec directly
// to parse_vector / parse_vector_spaces.
Ret load_vector(const char* file_path, std::string& vec) {
    if (file_path == nullptr) {
        return Ret("load_vector: invalid arguments");
    }

    FILE* f = fopen(file_path, "rb");
    if (f == nullptr) {
        return Ret("load_vector: failed to open file");
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return Ret("load_vector: failed to seek file");
    }
    const long file_size = ftell(f);
    if (file_size < 0) {
        fclose(f);
        return Ret("load_vector: failed to get file size");
    }
    rewind(f);

    vec.resize(static_cast<size_t>(file_size));
    const size_t read = fread(&vec[0], 1, static_cast<size_t>(file_size), f);
    fclose(f);

    if (read != static_cast<size_t>(file_size)) {
        return Ret("load_vector: failed to read file");
    }

    while (!vec.empty() && (vec.back() == '\n' || vec.back() == '\r')) {
        vec.pop_back();
    }

    return Ret(0);
}

// Formats a binary vector back into "[ ... ]" text while checking buffer
// capacity after every append so callers can grow the output buffer and retry.
Ret print_vector(uint8_t* vec_data, DataType type, uint16_t dim, char* buf, size_t buf_size, size_t digits) {
    if (vec_data == nullptr || buf == nullptr || buf_size == 0 || digits > 6) {
        return Ret("print_vector: invalid arguments");
    }
    int n = snprintf(buf, buf_size, "[ ");
    if (n < 0) {
        return Ret("print_vector: failed to format output");
    }
    if (static_cast<size_t>(n) >= buf_size) {
        return Ret("print_vector: output buffer is too small");
    }

    size_t used = static_cast<size_t>(n);

    char fmt_digits[16] = {};
    if (digits != 0) {
        // Build format like "%s%.3f" where digits controls precision.
        n = snprintf(fmt_digits, sizeof(fmt_digits), "%%s%%.%zuf", digits);
        if (n <= 0 || static_cast<size_t>(n) >= sizeof(fmt_digits)) {
            return Ret("print_vector: failed to build format string");
        }
    }

    const char* fmt = digits == 0 ? "%s%.9g" : fmt_digits;

    for (size_t i = 0; i < dim; ++i) {
        const char* sep = (i == 0) ? "" : ", ";
        n = 0;

        if (type == DataType::f32) {
            const float* p = reinterpret_cast<const float*>(vec_data);
            n = snprintf(buf + used, buf_size - used, fmt, sep, p[i]);
        } else if (type == DataType::f16) {
            const float16* p = reinterpret_cast<const float16*>(vec_data);
            n = snprintf(buf + used, buf_size - used, fmt, sep, static_cast<float>(p[i]));
        } else if (type == DataType::i16) {
            const int16_t* p = reinterpret_cast<const int16_t*>(vec_data);
            n = snprintf(buf + used, buf_size - used, "%s%d", sep, static_cast<int>(p[i]));
        } else {
            return Ret("print_vector: unsupported data type");
        }

        if (n < 0) {
            return Ret("print_vector: failed to format output");
        }

        const size_t written = static_cast<size_t>(n);
        if (written >= (buf_size - used)) {
            return Ret("print_vector: output buffer is too small");
        }
        used += written;
    }

    n = snprintf(buf + used, buf_size - used, " ]");
    if (n < 0) {
        return Ret("print_vector: failed to format output");
    }
    if (static_cast<size_t>(n) >= (buf_size - used)) {
        return Ret("print_vector: output buffer is too small");
    }

    return Ret(0);
}

namespace {

constexpr uint32_t crc32_byte(uint32_t byte) {
    uint32_t crc = byte;
    for (int bit = 0; bit < 8; ++bit) {
        const uint32_t mask = -(crc & 1u);
        crc = (crc >> 1) ^ (0xEDB88320u & mask);
    }
    return crc;
}

struct Crc32Table {
    uint32_t entries[256];
    constexpr Crc32Table() : entries{} {
        for (uint32_t i = 0; i < 256; ++i) {
            entries[i] = crc32_byte(i);
        }
    }
};

constexpr Crc32Table kCrc32Table{};

} // namespace

uint32_t crc32_update(uint32_t crc, const uint8_t* data, size_t size) {
    crc = ~crc;
    for (size_t i = 0; i < size; ++i) {
        crc = kCrc32Table.entries[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
    }
    return ~crc;
}

} // namespace sketch2
