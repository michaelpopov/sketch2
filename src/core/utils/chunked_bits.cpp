#include "chunked_bits.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace sketch2 {

namespace {

constexpr uint32_t kChunkedBitsBlobMagic =
    (static_cast<uint32_t>('S')) |
    (static_cast<uint32_t>('K') << 8u) |
    (static_cast<uint32_t>('C') << 16u) |
    (static_cast<uint32_t>('B') << 24u);
constexpr uint16_t kChunkedBitsBlobVersion = 1;

bool is_aligned(const void* ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

bool add_overflows(size_t lhs, size_t rhs, size_t* out) {
    if (lhs > std::numeric_limits<size_t>::max() - rhs) {
        return true;
    }
    *out = lhs + rhs;
    return false;
}

bool multiply_overflows(size_t lhs, size_t rhs, size_t* out) {
    if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs) {
        return true;
    }
    *out = lhs * rhs;
    return false;
}

bool align_up(size_t value, size_t alignment, size_t* out) {
    const size_t mask = alignment - 1u;
    if (value > std::numeric_limits<size_t>::max() - mask) {
        return false;
    }
    *out = (value + mask) & ~mask;
    return true;
}

void write_u16_le(uint8_t* out, uint16_t value) {
    out[0] = static_cast<uint8_t>(value);
    out[1] = static_cast<uint8_t>(value >> 8u);
}

void write_u32_le(uint8_t* out, uint32_t value) {
    for (size_t i = 0; i < sizeof(value); ++i) {
        out[i] = static_cast<uint8_t>(value >> (i * 8u));
    }
}

void write_u64_le(uint8_t* out, uint64_t value) {
    for (size_t i = 0; i < sizeof(value); ++i) {
        out[i] = static_cast<uint8_t>(value >> (i * 8u));
    }
}

uint16_t read_u16_le(const uint8_t* data) {
    return static_cast<uint16_t>(data[0]) |
        (static_cast<uint16_t>(data[1]) << 8u);
}

uint32_t read_u32_le(const uint8_t* data) {
    uint32_t value = 0;
    for (size_t i = 0; i < sizeof(value); ++i) {
        value |= static_cast<uint32_t>(data[i]) << (i * 8u);
    }
    return value;
}

uint64_t read_u64_le(const uint8_t* data) {
    uint64_t value = 0;
    for (size_t i = 0; i < sizeof(value); ++i) {
        value |= static_cast<uint64_t>(data[i]) << (i * 8u);
    }
    return value;
}

} // namespace

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

Ret ChunkedBits::compute_serialized_size_bytes(size_t* out) const {
    if (out == nullptr) {
        return Ret("ChunkedBits::serialized_size_bytes: output pointer is null");
    }
    *out = 0;

    if (!finished_) {
        return Ret("ChunkedBits::serialized_size_bytes: finish must be called first");
    }

    size_t directory_bytes = 0;
    if (multiply_overflows(
            chunks_.size(), kChunkedBitsBlobDirectoryEntryBytes, &directory_bytes)) {
        return Ret("ChunkedBits::serialized_size_bytes: directory size overflow");
    }

    size_t size = 0;
    if (add_overflows(kChunkedBitsBlobHeaderBytes, directory_bytes, &size)) {
        return Ret("ChunkedBits::serialized_size_bytes: header and directory size overflow");
    }

    for (const Chunk& chunk : chunks_) {
        if (!align_up(size, kChunkedBitsBlobAlignment, &size)) {
            return Ret("ChunkedBits::serialized_size_bytes: payload alignment overflow");
        }
        if (add_overflows(size, chunk.ids.serialized_size_bytes(), &size)) {
            return Ret("ChunkedBits::serialized_size_bytes: payload size overflow");
        }
    }
    *out = size;
    return Ret(0);
}

size_t ChunkedBits::serialized_size_bytes() const {
    size_t size = 0;
    if (compute_serialized_size_bytes(&size).code() != 0) {
        return 0;
    }
    return size;
}

Ret ChunkedBits::serialize(void* out, size_t size) const {
    if (!finished_) {
        return Ret("ChunkedBits::serialize: finish must be called before serialization");
    }
    if (out == nullptr) {
        return Ret("ChunkedBits::serialize: output pointer is null");
    }
    if (!is_aligned(out, kChunkedBitsBlobAlignment)) {
        return Ret("ChunkedBits::serialize: output buffer must be 32-byte aligned");
    }

    size_t expected_size = 0;
    Ret ret = compute_serialized_size_bytes(&expected_size);
    if (ret.code() != 0) {
        return ret;
    }
    if (size != expected_size) {
        return Ret("ChunkedBits::serialize: output buffer size mismatch");
    }

    auto* bytes = static_cast<uint8_t*>(out);
    std::memset(bytes, 0, size);
    write_u32_le(bytes, kChunkedBitsBlobMagic);
    write_u16_le(bytes + 4, kChunkedBitsBlobVersion);
    write_u16_le(bytes + 6, static_cast<uint16_t>(kChunkBits));
    write_u64_le(bytes + 8, static_cast<uint64_t>(chunks_.size()));

    size_t payload_offset = kChunkedBitsBlobHeaderBytes +
        chunks_.size() * kChunkedBitsBlobDirectoryEntryBytes;
    for (size_t i = 0; i < chunks_.size(); ++i) {
        const Chunk& chunk = chunks_[i];
        if (!align_up(payload_offset, kChunkedBitsBlobAlignment, &payload_offset)) {
            return Ret("ChunkedBits::serialize: payload offset overflow");
        }
        const size_t payload_size = chunk.ids.serialized_size_bytes();
        if (payload_offset > size || payload_size > size - payload_offset) {
            return Ret("ChunkedBits::serialize: payload exceeds output buffer");
        }

        uint8_t* dir = bytes + kChunkedBitsBlobHeaderBytes +
            i * kChunkedBitsBlobDirectoryEntryBytes;
        write_u64_le(dir, chunk.chunk_id);
        write_u64_le(dir + 8, static_cast<uint64_t>(payload_offset));
        write_u64_le(dir + 16, static_cast<uint64_t>(payload_size));
        CHECK(chunk.ids.serialize(reinterpret_cast<char*>(bytes + payload_offset)));
        payload_offset += payload_size;
    }

    return Ret(0);
}

Ret ChunkedBitsView::init_blob(const void* data, size_t size) {
    owned_blob_.reset();
    chunks_.clear();

    if (data == nullptr) {
        return Ret("ChunkedBitsView::init_blob: data pointer is null");
    }
    if (!is_aligned(data, kChunkedBitsBlobAlignment)) {
        return Ret("ChunkedBitsView::init_blob: blob buffer must be 32-byte aligned");
    }
    if (size < kChunkedBitsBlobHeaderBytes) {
        return Ret("ChunkedBitsView::init_blob: blob is too small");
    }

    const auto* bytes = static_cast<const uint8_t*>(data);
    if (read_u32_le(bytes) != kChunkedBitsBlobMagic) {
        return Ret("ChunkedBitsView::init_blob: invalid blob magic");
    }
    if (read_u16_le(bytes + 4) != kChunkedBitsBlobVersion) {
        return Ret("ChunkedBitsView::init_blob: unsupported blob version");
    }
    if (read_u16_le(bytes + 6) != kChunkBits) {
        return Ret("ChunkedBitsView::init_blob: invalid chunk_bits");
    }

    const uint64_t chunk_count_u64 = read_u64_le(bytes + 8);
    if (chunk_count_u64 > kChunkedBitsMaxChunks ||
            chunk_count_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return Ret("ChunkedBitsView::init_blob: too many chunks");
    }
    const size_t chunk_count = static_cast<size_t>(chunk_count_u64);
    if (chunk_count == 0) {
        if (size != kChunkedBitsBlobHeaderBytes) {
            return Ret("ChunkedBitsView::init_blob: empty filter blob has trailing data");
        }
        return Ret(0);
    }

    size_t directory_bytes = 0;
    if (multiply_overflows(
            chunk_count, kChunkedBitsBlobDirectoryEntryBytes, &directory_bytes)) {
        return Ret("ChunkedBitsView::init_blob: directory size overflow");
    }
    size_t directory_end = 0;
    if (add_overflows(kChunkedBitsBlobHeaderBytes, directory_bytes, &directory_end) ||
            directory_end > size) {
        return Ret("ChunkedBitsView::init_blob: directory exceeds blob size");
    }

    chunks_.reserve(chunk_count);
    uint64_t previous_chunk_id = 0;
    for (size_t i = 0; i < chunk_count; ++i) {
        const uint8_t* dir = bytes + kChunkedBitsBlobHeaderBytes +
            i * kChunkedBitsBlobDirectoryEntryBytes;
        const uint64_t chunk_id = read_u64_le(dir);
        const uint64_t payload_offset_u64 = read_u64_le(dir + 8);
        const uint64_t payload_size_u64 = read_u64_le(dir + 16);
        if (i > 0 && chunk_id <= previous_chunk_id) {
            return Ret("ChunkedBitsView::init_blob: chunk directory is not strictly sorted");
        }
        previous_chunk_id = chunk_id;
        if (payload_offset_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
                payload_size_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            return Ret("ChunkedBitsView::init_blob: payload range is too large");
        }
        const size_t payload_offset = static_cast<size_t>(payload_offset_u64);
        const size_t payload_size = static_cast<size_t>(payload_size_u64);
        if ((payload_offset % kChunkedBitsBlobAlignment) != 0u) {
            return Ret("ChunkedBitsView::init_blob: payload offset is not 32-byte aligned");
        }
        if (payload_offset > size || payload_size > size - payload_offset) {
            return Ret("ChunkedBitsView::init_blob: payload exceeds blob size");
        }

        Chunk chunk;
        chunk.chunk_id = chunk_id;
        CHECK(chunk.ids.init_frozen_view(
            bytes + payload_offset, payload_size, chunk_id << kChunkBits));
        chunks_.push_back(std::move(chunk));
    }

    return Ret(0);
}

void ChunkedBitsView::FreeDeleter::operator()(void* ptr) const {
    std::free(ptr);
}

Ret ChunkedBitsView::init_owned_blob(void* data, size_t size) {
    std::unique_ptr<void, FreeDeleter> owned(data);
    const Ret ret = init_blob(data, size);
    if (ret.code() != 0) {
        return ret;
    }
    owned_blob_ = std::move(owned);
    return Ret(0);
}

ChunkedBitsView::Iterator::Iterator(const ChunkedBitsView* view)
    : view_(view) {
    load_current_chunk_();
}

bool ChunkedBitsView::Iterator::load_current_chunk_() {
    ids_ = RoaringIds::SeekCursor();
    while (view_ != nullptr && chunk_index_ < view_->chunks_.size()) {
        ids_ = view_->chunks_[chunk_index_].ids.seek_begin();
        if (!ids_.eof()) {
            return true;
        }
        ++chunk_index_;
    }
    return false;
}

bool ChunkedBitsView::Iterator::eof() const {
    return view_ == nullptr || chunk_index_ >= view_->chunks_.size() || ids_.eof();
}

uint64_t ChunkedBitsView::Iterator::id() const {
    if (eof()) {
        throw std::out_of_range("ChunkedBitsView::Iterator::id: iterator is at end");
    }
    return ids_.id();
}

void ChunkedBitsView::Iterator::next() {
    if (eof()) {
        return;
    }
    ids_.next();
    if (!ids_.eof()) {
        return;
    }
    ++chunk_index_;
    load_current_chunk_();
}

bool ChunkedBitsView::Iterator::seek_at_least(uint64_t id) {
    if (view_ == nullptr) {
        return false;
    }
    if (!eof() && this->id() >= id) {
        return true;
    }

    const uint64_t target_chunk_id = get_chunk_id(id);
    const size_t previous_chunk_index = chunk_index_;
    while (chunk_index_ < view_->chunks_.size() &&
            view_->chunks_[chunk_index_].chunk_id < target_chunk_id) {
        ++chunk_index_;
    }
    if ((chunk_index_ != previous_chunk_index || ids_.eof()) && !load_current_chunk_()) {
        return false;
    }
    if (chunk_index_ >= view_->chunks_.size()) {
        return false;
    }

    const Chunk& chunk = view_->chunks_[chunk_index_];
    if (chunk.chunk_id != target_chunk_id) {
        return true;
    }
    if (ids_.seek_at_least(id)) {
        return true;
    }

    ++chunk_index_;
    return load_current_chunk_();
}

bool ChunkedBitsView::Iterator::consume_if_equal(uint64_t id) {
    if (!seek_at_least(id) || this->id() != id) {
        return false;
    }
    next();
    return true;
}

ChunkedBitsView::Iterator ChunkedBitsView::begin() const {
    return Iterator(this);
}

} // namespace sketch2
