#include "string_utils.h"
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


} // namespace sketch2
