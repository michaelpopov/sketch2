#include "chunked_bits.h"

#include "utils/checked_arithmetic.h"

#include <algorithm>
#include <cctype>
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

struct ChunkedBitsBlobHeader {
    uint32_t magic = 0;
    uint16_t version = 0;
    uint16_t chunk_bits = 0;
    uint64_t chunk_count = 0;
};

struct ChunkedBitsBlobDirectoryEntry {
    uint64_t chunk_id = 0;
    uint64_t payload_offset = 0;
    uint64_t payload_size = 0;
};

static_assert(sizeof(ChunkedBitsBlobHeader) == kChunkedBitsBlobHeaderBytes);
static_assert(sizeof(ChunkedBitsBlobDirectoryEntry) ==
    kChunkedBitsBlobDirectoryEntryBytes);

} // namespace

Ret validate_chunked_bits_name(const char* name) {
    if (name == nullptr || name[0] == '\0') {
        return Ret("sketch2: invalid bitset filter name");
    }

    for (const char* p = name; *p != '\0'; ++p) {
        const auto c = static_cast<unsigned char>(*p);
        if (!std::isalnum(c) && c != '_') {
            return Ret("sketch2: invalid bitset filter name");
        }
    }

    return Ret(0);
}

Ret ChunkedBits::set_name(const char* name) {
    if (name == nullptr) {
        return Ret(0);
    }

    CHECK(validate_chunked_bits_name(name));
    name_ = name;

    return Ret(0);
}

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
        // Buffered mode lets SQLite feed ids in arbitrary order; each flush
        // sorts within the chunk before handing offsets to CRoaring.
        it = builders_.try_emplace(chunk_id).first;
        CHECK(it->second.init_buffered(
            chunk_id << kChunkBits, kChunkedBitsBuilderBufferSize, true));
    }

    last_chunk_id_ = chunk_id;
    last_builder_ = &it->second;
    return last_builder_->add(id);
}

Ret ChunkedBits::add(const uint64_t* ids, size_t size) {
    if (finished_) {
        return Ret("ChunkedBits::add: cannot add after finish");
    }
    if (size == 0) {
        return Ret(0);
    }
    if (ids == nullptr) {
        return Ret("ChunkedBits::add: ids pointer is null");
    }

    size_t i = 0;
    while (i < size) {
        const uint64_t chunk_id = get_chunk_id(ids[i]);
        size_t j = i + 1;
        while (j < size) {
            if (ids[j] < ids[j - 1]) {
                return Ret("ChunkedBits::add: ids must be sorted in non-decreasing order");
            }
            if (get_chunk_id(ids[j]) != chunk_id) {
                break;
            }
            ++j;
        }

        RoaringIdsBuilder* builder = nullptr;
        if (last_builder_ != nullptr && last_chunk_id_ == chunk_id) {
            builder = last_builder_;
        } else {
            auto it = builders_.find(chunk_id);
            if (it == builders_.end()) {
                if (builders_.size() >= kChunkedBitsMaxChunks) {
                    return Ret("ChunkedBits::add: too many chunks");
                }
                it = builders_.try_emplace(chunk_id).first;
                CHECK(it->second.init_buffered(
                    chunk_id << kChunkBits, kChunkedBitsBuilderBufferSize, true));
            }
            builder = &it->second;
            last_chunk_id_ = chunk_id;
            last_builder_ = builder;
        }

        CHECK(builder->load(ids + i, j - i));
        i = j;
    }
    return Ret(0);
}

Ret ChunkedBits::finish() {
    if (finished_) {
        return finish_ret_;
    }

    chunks_.reserve(builders_.size());
    for (auto& item : builders_) {
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

    size_t size = 0;
    finish_ret_ = compute_serialized_size_bytes(&size);
    if (finish_ret_.code() != 0) {
        cached_serialized_size_ = 0;
    } else {
        cached_serialized_size_ = size;
    }
    return finish_ret_;
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
    if (!finished_) {
        return 0;
    }
    return cached_serialized_size_;
}

Ret ChunkedBits::serialize(void* out, size_t size) const {
    if (!finished_) {
        return Ret("ChunkedBits::serialize: finish must be called before serialization");
    }
    if (finish_ret_.code() != 0) {
        return finish_ret_;
    }
    if (out == nullptr) {
        return Ret("ChunkedBits::serialize: output pointer is null");
    }
    if (!is_aligned(out, kChunkedBitsBlobAlignment)) {
        return Ret("ChunkedBits::serialize: output buffer must be 32-byte aligned");
    }

    if (size != cached_serialized_size_) {
        return Ret("ChunkedBits::serialize: output buffer size mismatch");
    }

    auto* bytes = static_cast<uint8_t*>(out);
    std::memset(bytes, 0, size);
    const ChunkedBitsBlobHeader header = {
        .magic = kChunkedBitsBlobMagic,
        .version = kChunkedBitsBlobVersion,
        .chunk_bits = static_cast<uint16_t>(kChunkBits),
        .chunk_count = static_cast<uint64_t>(chunks_.size()),
    };
    std::memcpy(bytes, &header, sizeof(header));

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

        const ChunkedBitsBlobDirectoryEntry entry = {
            .chunk_id = chunk.chunk_id,
            .payload_offset = static_cast<uint64_t>(payload_offset),
            .payload_size = static_cast<uint64_t>(payload_size),
        };
        std::memcpy(bytes + kChunkedBitsBlobHeaderBytes +
                i * kChunkedBitsBlobDirectoryEntryBytes,
            &entry, sizeof(entry));
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
    ChunkedBitsBlobHeader header;
    std::memcpy(&header, bytes, sizeof(header));
    if (header.magic != kChunkedBitsBlobMagic) {
        return Ret("ChunkedBitsView::init_blob: invalid blob magic");
    }
    if (header.version != kChunkedBitsBlobVersion) {
        return Ret("ChunkedBitsView::init_blob: unsupported blob version");
    }
    if (header.chunk_bits != kChunkBits) {
        return Ret("ChunkedBitsView::init_blob: invalid chunk_bits");
    }

    const uint64_t chunk_count_u64 = header.chunk_count;
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
        ChunkedBitsBlobDirectoryEntry entry;
        std::memcpy(&entry, bytes + kChunkedBitsBlobHeaderBytes +
                i * kChunkedBitsBlobDirectoryEntryBytes,
            sizeof(entry));
        const uint64_t chunk_id = entry.chunk_id;
        const uint64_t payload_offset_u64 = entry.payload_offset;
        const uint64_t payload_size_u64 = entry.payload_size;
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
