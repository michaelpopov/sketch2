#pragma once

#include "core/utils/chunked_bits.h"
#include "core/utils/mapped_region.h"

#include <cstddef>
#include <filesystem>

namespace sketch2api::detail {

enum class BitsetFilterStorageKind {
    Heap,
    MappedFile,
};

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
    std::filesystem::path path;
    sketch2::MappedRegion region;
};

struct BitsetFilterControl {
    void reset();

    // storage must be declared before view so the borrowed view is destroyed
    // before the backing bytes are released.
    BitsetFilterStorage storage;
    sketch2::ChunkedBitsView view;
};

sketch2::Ret init_heap_bitset_filter(
    const sketch2::ChunkedBits& bits, size_t blob_size, BitsetFilterControl* control);
sketch2::Ret init_mapped_bitset_filter(
    const sketch2::ChunkedBits& bits, size_t blob_size,
    const std::filesystem::path& spill_dir, BitsetFilterControl* control);

BitsetFilterStorageKind bitset_filter_storage_kind_for_testing(const BitsetFilterControl* control);
const std::filesystem::path& bitset_filter_temp_path_for_testing(const BitsetFilterControl* control);

} // namespace sketch2api::detail
