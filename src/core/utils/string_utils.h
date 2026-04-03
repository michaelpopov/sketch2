// Declares parsing and formatting helpers for textual vectors.

#pragma once
#include "utils/shared_types.h"

namespace sketch2 {

Ret parse_vector(uint8_t* buf, size_t size, DataType type, uint16_t dim, const char* line, const char* end = nullptr);
Ret parse_vector_spaces(uint8_t* buf, size_t size, DataType type, uint16_t dim, const char* line, const char* end = nullptr);
bool check_comma_format(const char* line, const char* end = nullptr);
Ret load_vector(const char* file_path, std::string& vec);
Ret print_vector(uint8_t* vec_data, DataType type, uint16_t dim, char* buf, size_t buf_size, size_t digits = 2);
uint32_t crc32_update(uint32_t crc, const uint8_t* data, size_t size);

} // namespace sketch2
