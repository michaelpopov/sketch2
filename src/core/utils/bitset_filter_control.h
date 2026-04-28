#pragma once

#include "utils/chunked_bits.h"
#include "utils/mapped_region.h"

#include <cstddef>
#include <filesystem>
#include <memory>

namespace sketch2 {

enum class BitsetFilterStorageKind {
    Heap,
    MappedFile,
    MappedFileTemporary
};

constexpr const char* kBitsetFilterNamedFileSuffix = ".bitset";
constexpr const char* kBitsetFilterNamedTempFileTemplateSuffix = ".bitset.tmp.XXXXXX";

struct BitsetFilterStorage {
    BitsetFilterStorage() = default;
    BitsetFilterStorage(const BitsetFilterStorage&) = delete;
    BitsetFilterStorage& operator=(const BitsetFilterStorage&) = delete;
    BitsetFilterStorage(BitsetFilterStorage&&) = delete;
    BitsetFilterStorage& operator=(BitsetFilterStorage&&) = delete;
    ~BitsetFilterStorage();

    void reset();

    BitsetFilterStorageKind kind = BitsetFilterStorageKind::Heap;
    void* data = nullptr;
    size_t size = 0;
    int fd = -1;
    MappedRegion region;
};

struct BitsetFilterControl {
    static Ret create(ChunkedBits& bits, std::unique_ptr<BitsetFilterControl>* out);
    static Ret create_empty(std::unique_ptr<BitsetFilterControl>* out);

    ~BitsetFilterControl() = default;

    void reset();

    // storage must be declared before view so the borrowed view is destroyed
    // before the backing bytes are released.
    BitsetFilterStorage storage;
    ChunkedBitsView view;

private:
    BitsetFilterControl() = default;

    Ret init_from_builder_(ChunkedBits& bits);
    Ret init_heap_from_bits_(const ChunkedBits& bits, size_t blob_size);
    Ret init_named_mapped_from_bits_(
        const ChunkedBits& bits, size_t blob_size,
        const std::filesystem::path& spill_dir);
    Ret init_temp_mapped_from_bits_(const ChunkedBits& bits, size_t blob_size,
        const std::filesystem::path& spill_dir);
};

BitsetFilterStorageKind bitset_filter_storage_kind_for_testing(const BitsetFilterControl* control);

} // namespace sketch2
