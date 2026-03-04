#pragma once
#include "utils/shared_types.h"

namespace sketch2 {

Ret parse_vector(uint8_t* buf, size_t size, DataType type, uint16_t dim, const char* line, const char* end = nullptr);

} // namespace sketch2
