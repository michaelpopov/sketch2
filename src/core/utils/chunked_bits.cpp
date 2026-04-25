#include "chunked_bits.h"

#include <algorithm>

namespace sketch2 {

Ret ChunkedBits::add(uint64_t id) {
    if (finished_) {
        return Ret("ChunkedBits::add: cannot add after finish");
    }

    const uint64_t chunk_id = get_chunk_id(id);
    if (last_builder_ != nullptr && last_chunk_id_ == chunk_id) {
        return last_builder_->add(id);
    }

    auto it = builders_.find(chunk_id);
    if (it == builders_.end()) {
        if (builders_.size() >= kChunkedBitsMaxChunks) {
            return Ret("ChunkedBits::add: too many chunks");
        }
        auto [inserted_it, inserted] = builders_.try_emplace(chunk_id);
        it = inserted_it;
        // Buffered mode lets SQLite feed ids in arbitrary order; each flush
        // sorts within the chunk before handing offsets to CRoaring.
        if (inserted) {
            CHECK(it->second.init_buffered(
                chunk_id << kChunkBits, kChunkedBitsBuilderBufferSize, true));
        }
    }

    last_chunk_id_ = chunk_id;
    last_builder_ = &it->second;
    return last_builder_->add(id);
}

Ret ChunkedBits::finish() {
    if (finished_) {
        return Ret(0);
    }

    chunks_.reserve(builders_.size());
    for (auto& item : builders_) {
        // build() flushes any pending buffered ids, run-optimizes the bitmap,
        // and moves the read-only RoaringIds into the scan-time chunk table.
        chunks_.push_back(Chunk{
            .chunk_id = item.first,
            .ids = std::move(item.second).build(),
        });
    }
    last_chunk_id_ = 0;
    last_builder_ = nullptr;
    builders_.clear();
    std::sort(chunks_.begin(), chunks_.end(),
        [](const Chunk& lhs, const Chunk& rhs) {
            return lhs.chunk_id < rhs.chunk_id;
        });
    finished_ = true;
    return Ret(0);
}

bool ChunkedBits::contains(uint64_t id) const {
    const uint64_t chunk_id = get_chunk_id(id);
    const auto it = std::lower_bound(chunks_.begin(), chunks_.end(), chunk_id,
        [](const Chunk& chunk, uint64_t value) {
            return chunk.chunk_id < value;
        });
    return it != chunks_.end() && it->chunk_id == chunk_id && it->ids.contains(id);
}

bool ChunkedBits::empty() const {
    if (!finished_) {
        return builders_.empty();
    }
    for (const Chunk& chunk : chunks_) {
        if (!chunk.ids.empty()) {
            return false;
        }
    }
    return true;
}

} // namespace sketch2
