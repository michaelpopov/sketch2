// Shared scanner hot scan loops.

#pragma once

#include "core/compute/compute_engine.h"
#include "core/compute/cosine_distance.h"
#include "core/compute/scanner_heap_utils.h"
#include "core/compute/scanner_query_context.h"
#include "core/storage/data_reader.h"
#include "core/utils/bitset_filter.h"

#include <cassert>

namespace sketch2 {

constexpr size_t kPrefetchCacheLineBytes = 64;

inline void prefetch_vector_record(const uint8_t* data, size_t record_stride_bytes) {
    if (data == nullptr) {
        return;
    }

    for (size_t offset = 0; offset < record_stride_bytes; offset += kPrefetchCacheLineBytes) {
        __builtin_prefetch(data + offset, 0, 1);
    }
}

inline bool bitset_contains_id(const BitsetFilter* bitset, uint64_t id) {
    assert(bitset != nullptr);
    assert(bitset->data != nullptr || bitset->size == 0);
    if (id < bitset->base_id) {
        return false;
    }

    const uint64_t relative_id = id - bitset->base_id;
    const uint64_t byte_index = relative_id >> 3;
    if (byte_index >= bitset->size) {
        return false;
    }

    const uint8_t mask = static_cast<uint8_t>(1u << (relative_id & 7u));
    return (bitset->data[byte_index] & mask) != 0u;
}

template <bool HasBitset, typename PushFn>
inline void scan_ordered_iterator(DataReader::OrderedIterator it, size_t record_stride_bytes,
        const BitsetFilter* bitset, const PushFn& push_fn) {
    while (!it.eof()) {
        const uint64_t id = it.id();
        DataReader::OrderedIterator next_it = it;
        next_it.next();
        if (!next_it.eof()) {
            prefetch_vector_record(next_it.data(), record_stride_bytes);
        }
        if constexpr (HasBitset) {
            if (!bitset_contains_id(bitset, id)) {
                it = next_it;
                continue;
            }
        }
        push_fn(id, it);
        it = next_it;
    }
}

template <bool HasBitset, typename PushFn>
inline void scan_data_reader(const DataReader& reader, size_t record_stride_bytes,
        const BitsetFilter* bitset, const PushFn& push_fn) {
    scan_ordered_iterator<HasBitset>(
        reader.base_begin(), record_stride_bytes, bitset, push_fn);
    scan_ordered_iterator<HasBitset>(
        reader.delta_begin(), record_stride_bytes, bitset, push_fn);
}

template <typename PushFn>
inline void scan_data_reader_with_optional_bitset(const DataReader& reader,
        size_t record_stride_bytes, const BitsetFilter* bitset, const PushFn& push_fn) {
    if (bitset == nullptr) {
        scan_data_reader<false>(reader, record_stride_bytes, nullptr, push_fn);
        return;
    }

    scan_data_reader<true>(reader, record_stride_bytes, bitset, push_fn);
}

inline void scan_data_reader_with_dist(const DataReader& reader, size_t count, DistHeap* heap,
        ComputeDistFn dist_fn, const QueryDistContext& query, const BitsetFilter* bitset = nullptr) {
    const size_t record_stride_bytes = reader.stride();
    scan_data_reader_with_optional_bitset(reader, record_stride_bytes, bitset,
        [heap, count, dist_fn, query](uint64_t id, DataReader::OrderedIterator curr_it) {
            push_result(heap, count, id, dist_fn(curr_it.data(), query.vec, query.dim));
        });
}

inline void scan_data_reader_with_dot(const DataReader& reader, size_t count, DistHeap* heap,
        ComputeDotFn dot_fn, const QueryDotContext& query, const BitsetFilter* bitset = nullptr) {
    const size_t record_stride_bytes = reader.stride();
    scan_data_reader_with_optional_bitset(reader, record_stride_bytes, bitset,
        [heap, count, dot_fn, query](uint64_t id, DataReader::OrderedIterator curr_it) {
            push_result(heap, count, id, dot_fn(curr_it.data(), query.vec, query.dim));
        });
}

inline void scan_data_reader_with_query_norm(const DataReader& reader, size_t count, DistHeap* heap,
        ComputeDistWithQueryNormFn dist_fn, const QueryCosContext& query,
        const BitsetFilter* bitset = nullptr) {
    const size_t record_stride_bytes = reader.stride();
    scan_data_reader_with_optional_bitset(reader, record_stride_bytes, bitset,
        [heap, count, dist_fn, query](uint64_t id, DataReader::OrderedIterator curr_it) {
            push_result(heap, count, id,
                dist_fn(curr_it.data(), query.vec, query.dim, query.norm_sq));
        });
}

inline void scan_data_reader_with_cos_stored_norms(const DataReader& reader, size_t count,
        DistHeap* heap, ComputeDotFn dot_fn, const QueryCosContext& query,
        const BitsetFilter* bitset = nullptr) {
    const size_t record_stride_bytes = reader.stride();
    scan_data_reader_with_optional_bitset(reader, record_stride_bytes, bitset,
        [heap, count, dot_fn, query, &reader](uint64_t id, DataReader::OrderedIterator curr_it) {
            const uint8_t* record = curr_it.data();
            const double dot = dot_fn(record, query.vec, query.dim);
            push_result(heap, count, id, finalize_cosine_distance_from_inverse_norms(
                dot, static_cast<double>(reader.get_norm_from_record_unchecked(record)), query.inv_norm));
        });
}

inline void scan_data_reader_with_l2_stored_norms(const DataReader& reader, size_t count,
        DistHeap* heap, ComputeDotFn dot_fn, const QueryL2Context& query,
        const BitsetFilter* bitset = nullptr) {
    const size_t record_stride_bytes = reader.stride();
    scan_data_reader_with_optional_bitset(reader, record_stride_bytes, bitset,
        [heap, count, dot_fn, query, &reader](uint64_t id, DataReader::OrderedIterator curr_it) {
            const uint8_t* record = curr_it.data();
            const double dot = dot_fn(record, query.vec, query.dim);
            push_result(heap, count, id, finalize_squared_l2_distance_from_squared_norms(
                dot, static_cast<double>(reader.get_norm_from_record_unchecked(record)), query.norm_sq));
        });
}

} // namespace sketch2
