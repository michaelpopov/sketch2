#pragma once

#include "core/utils/chunked_bits.h"
#include "core/utils/mapped_region.h"

#include <cstddef>
#include <filesystem>

namespace sketch2api::detail {

enum class AllowlistStorageKind {
    Heap,
    MappedFile,
};

struct AllowlistStorage {
    AllowlistStorage() = default;
    AllowlistStorage(const AllowlistStorage&) = delete;
    AllowlistStorage& operator=(const AllowlistStorage&) = delete;
    AllowlistStorage(AllowlistStorage&&) = delete;
    AllowlistStorage& operator=(AllowlistStorage&&) = delete;
    ~AllowlistStorage();

    void reset();

    AllowlistStorageKind kind = AllowlistStorageKind::Heap;
    void* data = nullptr;
    size_t size = 0;
    std::filesystem::path path;
    sketch2::MappedRegion region;
};

struct AllowlistControl {
    void reset();

    // storage must be declared before view so the borrowed view is destroyed
    // before the backing bytes are released.
    AllowlistStorage storage;
    sketch2::ChunkedBitsView view;
};

sketch2::Ret init_heap_allowlist(
    const sketch2::ChunkedBits& bits, size_t blob_size, AllowlistControl* control);
sketch2::Ret init_mapped_allowlist(
    const sketch2::ChunkedBits& bits, size_t blob_size,
    const std::filesystem::path& spill_dir, AllowlistControl* control);

AllowlistStorageKind allowlist_storage_kind_for_testing(const AllowlistControl* control);
const std::filesystem::path& allowlist_temp_path_for_testing(const AllowlistControl* control);

} // namespace sketch2api::detail
