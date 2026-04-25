#pragma once

#include "roaring_ids.h"
#include "shared_types.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace sketch2 {

// ChunkedBits is the process-local allowlist representation produced by
// SQLite bitset_agg(). A single RoaringIdsBuilder can only cover a uint32_t
// offset span from its base, so ids are split into fixed 2^20-sized chunks.
// The chunk size is independent of dataset ranges: bitset_agg() does not know
// which virtual table will consume the result, but fixed chunks still keep
// memory proportional to selected ids instead of max(id) - min(id).
constexpr uint64_t kChunkBits = 20; // 1,048,576 ids
constexpr uint64_t kChunkSize = 1ull << kChunkBits;
constexpr uint64_t kChunkMask = kChunkSize - 1;
constexpr size_t kChunkedBitsBuilderBufferSize = 4096;
constexpr size_t kChunkedBitsMaxChunks = 100000;

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
    bool contains(uint64_t id) const;
    bool empty() const;

private:
    using BuildersMap = std::unordered_map<uint64_t, RoaringIdsBuilder>;

    // Finalized chunks are sorted by chunk_id so membership checks can find
    // the right RoaringIds with binary search while scan threads share this
    // object read-only.
    struct Chunk {
        uint64_t chunk_id = 0;
        RoaringIds ids;
    };

    // Builders are used only during aggregation, where incoming ids may be
    // unsorted and sparse. finish() freezes them into chunks_ and clears this
    // map before the object is passed to scanner code.
    BuildersMap builders_;
    uint64_t last_chunk_id_ = 0;
    RoaringIdsBuilder* last_builder_ = nullptr;
    std::vector<Chunk> chunks_;
    bool finished_ = false;
};

} // namespace sketch2
