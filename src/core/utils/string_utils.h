// Declares parsing and formatting helpers for textual vectors.

#pragma once
#include "utils/shared_types.h"

namespace sketch2 {

Ret parse_vector(uint8_t* buf, size_t size, DataType type, uint16_t dim, const char* line, const char* end = nullptr);
Ret print_vector(uint8_t* vec_data, DataType type, uint16_t dim, char* buf, size_t buf_size);

} // namespace sketch2
