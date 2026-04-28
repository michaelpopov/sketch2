#pragma once

#include "roaring_ids.h"
#include "shared_types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace sketch2 {

// ChunkedBits builds the bitset filter representation serialized by SQLite
// bitset_agg(). A single RoaringIdsBuilder can only cover a uint32_t offset
// span from its base, so ids are split into fixed 2^20-sized chunks. The chunk
// size is independent of dataset ranges: bitset_agg() does not know which
// virtual table will consume the result, but fixed chunks still keep memory
// proportional to selected ids instead of max(id) - min(id).
constexpr uint64_t kChunkBits = 20; // 1,048,576 ids
constexpr uint64_t kChunkSize = 1ull << kChunkBits;
constexpr uint64_t kChunkMask = kChunkSize - 1;
constexpr size_t kChunkedBitsBuilderBufferSize = 4096;
constexpr size_t kChunkedBitsMaxChunks = 100000;
constexpr size_t kChunkedBitsBlobHeaderBytes = 16;
constexpr size_t kChunkedBitsBlobDirectoryEntryBytes = 24;
constexpr size_t kChunkedBitsBlobAlignment = 32;

inline uint64_t get_chunk_id(uint64_t id) {
    return id >> kChunkBits;
}

class ChunkedBits {
public:
    ChunkedBits() = default;
    ChunkedBits(const ChunkedBits&) = delete;
    ChunkedBits& operator=(const ChunkedBits&) = delete;
    ChunkedBits(ChunkedBits&&) noexcept = default;
    ChunkedBits& operator=(ChunkedBits&&) noexcept = default;

    Ret add(uint64_t id);
    Ret finish();
    size_t serialized_size_bytes() const;
    Ret serialize(void* out, size_t size) const;

    Ret set_name(const char* name);

private:
    using BuildersMap = std::unordered_map<uint64_t, RoaringIdsBuilder>;

    // Finalized chunks are sorted by chunk_id so membership checks can find
    // the right RoaringIds with binary search while scan threads share this
    // object read-only.
    struct Chunk {
        uint64_t chunk_id = 0;
        RoaringIds ids;
    };

    Ret compute_serialized_size_bytes(size_t* out) const;

    // Builders are used only during aggregation, where incoming ids may be
    // unsorted and sparse. finish() freezes them into chunks_ and clears this
    // map before the object is passed to scanner code.
    BuildersMap builders_;
    uint64_t last_chunk_id_ = 0;
    RoaringIdsBuilder* last_builder_ = nullptr;
    std::vector<Chunk> chunks_;
    bool finished_ = false;
    std::string name_;
};

// ChunkedBitsView reads serialized ChunkedBits blobs without copying payloads:
// each RoaringIds chunk keeps frozen-view pointers into the blob bytes.
// init_blob() borrows the caller's buffer, so that buffer must outlive the
// view. init_owned_blob() takes ownership of a malloc-compatible buffer and
// keeps it alive for the view.
class ChunkedBitsView {
public:
    class Iterator {
    public:
        Iterator() = default;

        bool eof() const;
        uint64_t id() const;
        void next();
        bool seek_at_least(uint64_t id);
        bool consume_if_equal(uint64_t id);

    private:
        friend class ChunkedBitsView;
        explicit Iterator(const ChunkedBitsView* view);

        bool load_current_chunk_();

        const ChunkedBitsView* view_ = nullptr;
        size_t chunk_index_ = 0;
        RoaringIds::SeekCursor ids_;
    };

    ChunkedBitsView() = default;
    ChunkedBitsView(const ChunkedBitsView&) = delete;
    ChunkedBitsView& operator=(const ChunkedBitsView&) = delete;
    ChunkedBitsView(ChunkedBitsView&&) noexcept = default;
    ChunkedBitsView& operator=(ChunkedBitsView&&) noexcept = default;

    Ret init_blob(const void* data, size_t size);
    Ret init_owned_blob(void* data, size_t size);
    Iterator begin() const;

private:
    struct FreeDeleter {
        void operator()(void* ptr) const;
    };

    struct Chunk {
        uint64_t chunk_id = 0;
        RoaringIds ids;
    };

    std::unique_ptr<void, FreeDeleter> owned_blob_;
    std::vector<Chunk> chunks_;
};

} // namespace sketch2
