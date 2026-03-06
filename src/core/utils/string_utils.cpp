#include "string_utils.h"
#include <cstdio>
#include <string.h>

namespace sketch2 {

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
            char* next;
            out[d] = strtof(line, &next);
            if (next == line || next > end) {
                return Ret("InputReader::data: invalid f32 token");
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
            out[d] = static_cast<int16_t>(strtol(line, &next, 10));
            if (next == line || next > end) {
                return Ret("InputReader::data: invalid i16 token");
            }
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
            char* next;
            float f = strtof(line, &next);
            if (next == line || next > end) {
                return Ret("InputReader::data: invalid f16 token");
            }
            out[d] = static_cast<float16>(f);
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

Ret print_vector(uint8_t* vec_data, DataType type, uint16_t dim, char* buf, size_t buf_size) {
    if (vec_data == nullptr || buf == nullptr || buf_size == 0) {
        return Ret("print_vector: invalid arguments");
    }
    if (type == DataType::f16 && !supports_f16()) {
        return Ret("print_vector: f16 is not supported");
    }

    int n = snprintf(buf, buf_size, "[ ");
    if (n < 0) {
        return Ret("print_vector: failed to format output");
    }
    if (static_cast<size_t>(n) >= buf_size) {
        return Ret("print_vector: output buffer is too small");
    }

    size_t used = static_cast<size_t>(n);

    for (size_t i = 0; i < dim; ++i) {
        const char* sep = (i == 0) ? "" : ", ";
        n = 0;

        if (type == DataType::f32) {
            const float* p = reinterpret_cast<const float*>(vec_data);
            n = snprintf(buf + used, buf_size - used, "%s%.9g", sep, p[i]);
        } else if (type == DataType::f16) {
            const float16* p = reinterpret_cast<const float16*>(vec_data);
            n = snprintf(buf + used, buf_size - used, "%s%.9g", sep, static_cast<float>(p[i]));
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


} // namespace sketch2
