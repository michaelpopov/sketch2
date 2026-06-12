// Implements CRC32C checksum helpers.

#include "crc.h"

#include <array>
#include <cstring>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <nmmintrin.h>
#endif

#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#endif

namespace sketch2 {

namespace {

constexpr uint32_t crc32c_byte(uint32_t byte) {
    constexpr uint32_t kCrc32cPolynomial = 0x82F63B78u;
    uint32_t crc = byte;
    for (int bit = 0; bit < 8; ++bit) {
        const uint32_t mask = -(crc & 1u);
        crc = (crc >> 1) ^ (kCrc32cPolynomial & mask);
    }
    return crc;
}

struct Crc32cTable {
    std::array<std::array<uint32_t, 256>, 16> entries;
    constexpr Crc32cTable() : entries{} {
        for (uint32_t i = 0; i < 256; ++i) {
            entries[0][i] = crc32c_byte(i);
        }
        for (size_t table = 1; table < entries.size(); ++table) {
            for (uint32_t i = 0; i < 256; ++i) {
                const uint32_t crc = entries[table - 1][i];
                entries[table][i] = (crc >> 8) ^ entries[0][crc & 0xFFu];
            }
        }
    }
};

constexpr Crc32cTable kCrc32cTable{};

uint32_t load_le32(const uint8_t* data) {
    uint32_t value = 0;
    std::memcpy(&value, data, sizeof(value));
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    value = __builtin_bswap32(value);
#endif
    return value;
}

uint64_t load_le64(const uint8_t* data) {
    uint64_t value = 0;
    std::memcpy(&value, data, sizeof(value));
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    value = __builtin_bswap64(value);
#endif
    return value;
}

uint32_t crc32c_update_slicing16(uint32_t crc, const uint8_t* data, size_t size) {
    crc = ~crc;
    while (size >= 16) {
        const uint32_t a = load_le32(data) ^ crc;
        const uint32_t b = load_le32(data + 4);
        const uint32_t c = load_le32(data + 8);
        const uint32_t d = load_le32(data + 12);
        crc =
            kCrc32cTable.entries[15][a & 0xFFu] ^
            kCrc32cTable.entries[14][(a >> 8) & 0xFFu] ^
            kCrc32cTable.entries[13][(a >> 16) & 0xFFu] ^
            kCrc32cTable.entries[12][a >> 24] ^
            kCrc32cTable.entries[11][b & 0xFFu] ^
            kCrc32cTable.entries[10][(b >> 8) & 0xFFu] ^
            kCrc32cTable.entries[9][(b >> 16) & 0xFFu] ^
            kCrc32cTable.entries[8][b >> 24] ^
            kCrc32cTable.entries[7][c & 0xFFu] ^
            kCrc32cTable.entries[6][(c >> 8) & 0xFFu] ^
            kCrc32cTable.entries[5][(c >> 16) & 0xFFu] ^
            kCrc32cTable.entries[4][c >> 24] ^
            kCrc32cTable.entries[3][d & 0xFFu] ^
            kCrc32cTable.entries[2][(d >> 8) & 0xFFu] ^
            kCrc32cTable.entries[1][(d >> 16) & 0xFFu] ^
            kCrc32cTable.entries[0][d >> 24];
        data += 16;
        size -= 16;
    }
    while (size > 0) {
        crc = kCrc32cTable.entries[0][(crc ^ *data++) & 0xFFu] ^ (crc >> 8);
        --size;
    }
    return ~crc;
}

using Crc32cUpdateFn = uint32_t (*)(uint32_t, const uint8_t*, size_t);

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("sse4.2")))
uint32_t crc32c_update_sse42(uint32_t crc, const uint8_t* data, size_t size) {
    uint64_t crc64 = ~static_cast<uint64_t>(crc) & 0xFFFFFFFFu;
    while (size >= 8) {
        crc64 = _mm_crc32_u64(crc64, load_le64(data));
        data += 8;
        size -= 8;
    }
    uint32_t crc32 = static_cast<uint32_t>(crc64);
    while (size > 0) {
        crc32 = _mm_crc32_u8(crc32, *data++);
        --size;
    }
    return ~crc32;
}

bool cpu_supports_sse42_crc32c() {
    return __builtin_cpu_supports("sse4.2");
}
#endif

#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
uint32_t crc32c_update_arm_crc(uint32_t crc, const uint8_t* data, size_t size) {
    crc = ~crc;
    while (size >= 8) {
        crc = __crc32cd(crc, load_le64(data));
        data += 8;
        size -= 8;
    }
    while (size > 0) {
        crc = __crc32cb(crc, *data++);
        --size;
    }
    return ~crc;
}
#endif

Crc32cUpdateFn resolve_crc32c_update() {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (cpu_supports_sse42_crc32c()) {
        return crc32c_update_sse42;
    }
#endif
#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
    return crc32c_update_arm_crc;
#else
    return crc32c_update_slicing16;
#endif
}

} // namespace

uint32_t crc32_update(uint32_t crc, const uint8_t* data, size_t size) {
    static const Crc32cUpdateFn update = resolve_crc32c_update();
    return update(crc, data, size);
}

} // namespace sketch2
