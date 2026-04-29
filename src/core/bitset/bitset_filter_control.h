#pragma once

#include "chunked_bits.h"
#include "utils/mapped_region.h"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>

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

    const void* data() const;
    void* writable_data();
    size_t size() const;

    BitsetFilterStorageKind kind = BitsetFilterStorageKind::Heap;

    // Heap mode only. Aligned-allocated blob; nullptr in mapped modes.
    void* heap_blob = nullptr;
    size_t heap_size = 0;

    // Mapped modes only. Region owns the mmap; fd is paired with it for
    // close on reset. Empty in heap mode.
    int fd = -1;
    MappedRegion region;
};

struct BitsetFilterControl;

using BitsetFilterControlPtr = std::unique_ptr<BitsetFilterControl>;

struct BitsetFilterControl {
    static Ret create(ChunkedBits& bits, BitsetFilterControlPtr* out);
    static Ret create_empty(BitsetFilterControlPtr* out);
    static Ret load_named(const char* name, BitsetFilterControlPtr* out);

    ~BitsetFilterControl() = default;

    // storage must be declared before view so the borrowed view is destroyed
    // before the unique_ptr drops the backing bytes.
    std::unique_ptr<BitsetFilterStorage> storage;
    ChunkedBitsView view;

private:
    BitsetFilterControl() = default;

    void reset();

    Ret init_from_builder_(ChunkedBits& bits);
    Ret init_heap_from_bits_(const ChunkedBits& bits, size_t blob_size);
    Ret init_mapped_storage_from_fd_(
        const ChunkedBits& bits, size_t blob_size, int* fd,
        BitsetFilterStorageKind kind);
    Ret init_named_mapped_from_bits_(
        const ChunkedBits& bits, size_t blob_size,
        const std::filesystem::path& spill_dir);
    Ret init_temp_mapped_from_bits_(const ChunkedBits& bits, size_t blob_size,
        const std::filesystem::path& spill_dir);
    Ret init_named_mapped_from_file_(const char* name);
    Ret swap_to_readonly_storage_for_cache_(
        const std::string& name,
        const std::filesystem::path& final_path);
};

BitsetFilterStorageKind bitset_filter_storage_kind_for_testing(const BitsetFilterControl* control);
std::filesystem::path named_bitset_filter_path(const std::string& name);
Ret load_named_bitset_filter(const char* name, BitsetFilterControlPtr* out);
Ret drop_named_bitset_filter(const char* name, bool* removed_out);

} // namespace sketch2
