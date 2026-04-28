#pragma once

#include "chunked_bits.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <string>
#include <utility>

#include <unistd.h>

namespace sketch2::test {

inline std::string unique_filter_name(const char* base) {
    static size_t counter = 0;
    return std::string(base) + "_" + std::to_string(getpid()) + "_" +
        std::to_string(++counter);
}

struct ScopedPathCleanup {
    std::filesystem::path path;

    ~ScopedPathCleanup() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

struct AlignedBlob {
    void* data = nullptr;
    size_t size = 0;
    size_t allocation_size = 0;

    AlignedBlob() = default;
    AlignedBlob(const AlignedBlob&) = delete;
    AlignedBlob& operator=(const AlignedBlob&) = delete;
    AlignedBlob(AlignedBlob&& other) noexcept
        : data(std::exchange(other.data, nullptr)),
          size(std::exchange(other.size, 0)),
          allocation_size(std::exchange(other.allocation_size, 0)) {}
    AlignedBlob& operator=(AlignedBlob&& other) noexcept {
        if (this != &other) {
            std::free(data);
            data = std::exchange(other.data, nullptr);
            size = std::exchange(other.size, 0);
            allocation_size = std::exchange(other.allocation_size, 0);
        }
        return *this;
    }
    ~AlignedBlob() {
        std::free(data);
    }

    uint8_t* bytes() {
        return static_cast<uint8_t*>(data);
    }
    const uint8_t* bytes() const {
        return static_cast<const uint8_t*>(data);
    }
};

inline size_t align_up(size_t value, size_t alignment) {
    const size_t mask = alignment - 1u;
    return (value + mask) & ~mask;
}

inline AlignedBlob make_aligned_blob(size_t size) {
    AlignedBlob blob;
    blob.size = size;
    blob.allocation_size = align_up(size, kChunkedBitsBlobAlignment);
    blob.data = std::aligned_alloc(kChunkedBitsBlobAlignment, blob.allocation_size);
    EXPECT_NE(nullptr, blob.data);
    return blob;
}

inline AlignedBlob read_aligned_blob(const std::filesystem::path& path) {
    AlignedBlob blob = make_aligned_blob(std::filesystem::file_size(path));
    if (blob.data == nullptr) {
        return blob;
    }

    FILE* file = std::fopen(path.c_str(), "rb");
    EXPECT_NE(nullptr, file);
    if (file == nullptr) {
        return blob;
    }
    EXPECT_EQ(blob.size, std::fread(blob.data, 1, blob.size, file));
    std::fclose(file);
    return blob;
}

inline void expect_persisted_filter_contains(
        const std::filesystem::path& path, std::initializer_list<uint64_t> ids) {
    AlignedBlob blob = read_aligned_blob(path);
    ASSERT_NE(nullptr, blob.data);

    ChunkedBitsView view;
    ASSERT_EQ(0, view.init_blob(blob.data, blob.size).code());

    auto it = view.begin();
    for (uint64_t id : ids) {
        EXPECT_TRUE(it.consume_if_equal(id)) << id;
    }
}

} // namespace sketch2::test
