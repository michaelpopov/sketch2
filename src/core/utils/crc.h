// Declares checksum helpers shared by storage formats.

#pragma once

#include <cstddef>
#include <cstdint>

namespace sketch2 {

// Incremental CRC32C checksum update. The historical name is retained so existing
// storage call sites do not need to distinguish the checksum engine.
uint32_t crc32_update(uint32_t crc, const uint8_t* data, size_t size);

} // namespace sketch2
