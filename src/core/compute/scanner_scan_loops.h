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
constexpr size_t kPrefetchDistance = 1;

inline void prefetch_vector_record(const uint8_t* data, size_t vector_size_bytes) {
    if (data == nullptr) {
        return;
    }

    for (size_t offset = 0; offset < vector_size_bytes; offset += kPrefetchCacheLineBytes) {
        __builtin_prefetch(data + offset, 0, 1);
    }
}

inline void prefetch_ordered_iterator_lookahead(DataReader::OrderedIterator it,
        size_t lookahead_distance, size_t vector_size_bytes) {
    for (size_t step = 0; step < lookahead_distance && !it.eof(); ++step) {
        it.next();
    }
    if (!it.eof()) {
        prefetch_vector_record(it.data(), vector_size_bytes);
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
inline void scan_ordered_iterator(DataReader::OrderedIterator it, size_t vector_size_bytes,
        const BitsetFilter* bitset, const PushFn& push_fn) {
    while (!it.eof()) {
        const uint64_t id = it.id();
        DataReader::OrderedIterator next_it = it;
        next_it.next();
        prefetch_ordered_iterator_lookahead(it, kPrefetchDistance, vector_size_bytes);
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
inline void scan_data_reader(const DataReader& reader, size_t vector_size_bytes,
        const BitsetFilter* bitset, const PushFn& push_fn) {
    scan_ordered_iterator<HasBitset>(
        reader.base_begin(), vector_size_bytes, bitset, push_fn);
    scan_ordered_iterator<HasBitset>(
        reader.delta_begin(), vector_size_bytes, bitset, push_fn);
}

template <typename PushFn>
inline void scan_data_reader_with_optional_bitset(const DataReader& reader,
        size_t vector_size_bytes, const BitsetFilter* bitset, const PushFn& push_fn) {
    if (bitset == nullptr) {
        scan_data_reader<false>(reader, vector_size_bytes, nullptr, push_fn);
        return;
    }

    scan_data_reader<true>(reader, vector_size_bytes, bitset, push_fn);
}

inline void scan_data_reader_with_dist(const DataReader& reader, size_t count, DistHeap* heap,
        CalcDistFn dist_fn, const QueryDistContext& query, const BitsetFilter* bitset = nullptr) {
    const size_t vector_size_bytes = reader.dim() * data_type_size(reader.type());
    scan_data_reader_with_optional_bitset(reader, vector_size_bytes, bitset,
        [heap, count, dist_fn, query](uint64_t id, DataReader::OrderedIterator curr_it) {
            push_result(heap, count, id, dist_fn(curr_it.data(), query.vec, query.dim));
        });
}

inline void scan_data_reader_with_query_norm(const DataReader& reader, size_t count, DistHeap* heap,
        CalcDistWithQueryNormFn dist_fn, const QueryCosContext& query,
        const BitsetFilter* bitset = nullptr) {
    const size_t vector_size_bytes = reader.dim() * data_type_size(reader.type());
    scan_data_reader_with_optional_bitset(reader, vector_size_bytes, bitset,
        [heap, count, dist_fn, query](uint64_t id, DataReader::OrderedIterator curr_it) {
            push_result(heap, count, id,
                dist_fn(curr_it.data(), query.vec, query.dim, query.norm_sq));
        });
}

inline void scan_data_reader_with_cos_stored_norms(const DataReader& reader, size_t count,
        DistHeap* heap, CalcDotFn dot_fn, const QueryCosContext& query,
        const BitsetFilter* bitset = nullptr) {
    const size_t vector_size_bytes = reader.dim() * data_type_size(reader.type());
    scan_data_reader_with_optional_bitset(reader, vector_size_bytes, bitset,
        [heap, count, dot_fn, query](uint64_t id, DataReader::OrderedIterator curr_it) {
            const double dot = dot_fn(curr_it.data(), query.vec, query.dim);
            push_result(heap, count, id, finalize_cosine_distance_from_inverse_norms(
                dot, static_cast<double>(curr_it.get_norm()), query.inv_norm));
        });
}

inline void scan_data_reader_with_l2_stored_norms(const DataReader& reader, size_t count,
        DistHeap* heap, CalcDotFn dot_fn, const QueryL2Context& query,
        const BitsetFilter* bitset = nullptr) {
    const size_t vector_size_bytes = reader.dim() * data_type_size(reader.type());
    scan_data_reader_with_optional_bitset(reader, vector_size_bytes, bitset,
        [heap, count, dot_fn, query](uint64_t id, DataReader::OrderedIterator curr_it) {
            const double dot = dot_fn(curr_it.data(), query.vec, query.dim);
            push_result(heap, count, id, finalize_squared_l2_distance_from_squared_norms(
                dot, static_cast<double>(curr_it.get_norm()), query.norm_sq));
        });
}

} // namespace sketch2
