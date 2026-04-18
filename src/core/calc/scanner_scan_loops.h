// Shared scanner hot scan loops.

#pragma once

#include "core/calc/calc_engine.h"
#include "core/calc/cosine_distance.h"
#include "core/calc/scanner_heap_utils.h"
#include "core/calc/scanner_query_context.h"
#include "core/storage/data_reader.h"
#include "core/utils/bitset_filter.h"

#include <cassert>

namespace sketch2 {

constexpr size_t kPrefetchCacheLineBytes = 64;

inline void prefetch_next_vector_record(const DataReader::OrderedIterator& it,
        size_t vector_size_bytes) {
    DataReader::OrderedIterator next_it = it;
    next_it.next();
    if (next_it.eof()) {
        return;
    }

    const uint8_t* const next_data = next_it.data();
    __builtin_prefetch(next_data, 0, 1);
    if (vector_size_bytes > kPrefetchCacheLineBytes) {
        __builtin_prefetch(next_data + kPrefetchCacheLineBytes, 0, 1);
    }
}

inline bool bitset_allows_id(const BitsetFilter* bitset, uint64_t id) {
    if (bitset == nullptr) {
        return true;
    }

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

inline void scan_ordered_iterator_with_dist(DataReader::OrderedIterator it, size_t count,
        DistHeap* heap, CalcDistFn dist_fn, const QueryDistContext& query,
        size_t vector_size_bytes, const BitsetFilter* bitset = nullptr) {
    for (; !it.eof(); it.next()) {
        const uint64_t id = it.id();
        if (!bitset_allows_id(bitset, id)) {
            continue;
        }
        prefetch_next_vector_record(it, vector_size_bytes);
        push_result(heap, count, id, dist_fn(it.data(), query.vec, query.dim));
    }
}

inline void scan_ordered_iterator_with_query_norm(DataReader::OrderedIterator it, size_t count,
        DistHeap* heap, CalcDistWithQueryNormFn dist_fn, const QueryCosContext& query,
        size_t vector_size_bytes, const BitsetFilter* bitset = nullptr) {
    for (; !it.eof(); it.next()) {
        const uint64_t id = it.id();
        if (!bitset_allows_id(bitset, id)) {
            continue;
        }
        prefetch_next_vector_record(it, vector_size_bytes);
        push_result(heap, count, id, dist_fn(it.data(), query.vec, query.dim, query.norm_sq));
    }
}

inline void scan_ordered_iterator_with_cos_stored_norms(DataReader::OrderedIterator it, size_t count,
        DistHeap* heap, CalcDotFn dot_fn, const QueryCosContext& query,
        size_t vector_size_bytes, const BitsetFilter* bitset = nullptr) {
    for (; !it.eof(); it.next()) {
        const uint64_t id = it.id();
        if (!bitset_allows_id(bitset, id)) {
            continue;
        }
        prefetch_next_vector_record(it, vector_size_bytes);
        const double dot = dot_fn(it.data(), query.vec, query.dim);
        push_result(heap, count, id, finalize_cosine_distance_from_inverse_norms(
            dot, static_cast<double>(it.get_norm()), query.inv_norm));
    }
}

inline void scan_ordered_iterator_with_l2_stored_norms(DataReader::OrderedIterator it, size_t count,
        DistHeap* heap, CalcDotFn dot_fn, const QueryL2Context& query,
        size_t vector_size_bytes, const BitsetFilter* bitset = nullptr) {
    for (; !it.eof(); it.next()) {
        const uint64_t id = it.id();
        if (!bitset_allows_id(bitset, id)) {
            continue;
        }
        prefetch_next_vector_record(it, vector_size_bytes);
        const double dot = dot_fn(it.data(), query.vec, query.dim);
        push_result(heap, count, id, finalize_squared_l2_distance_from_squared_norms(
            dot, static_cast<double>(it.get_norm()), query.norm_sq));
    }
}

inline void scan_data_reader_with_dist(const DataReader& reader, size_t count, DistHeap* heap,
        CalcDistFn dist_fn, const QueryDistContext& query, const BitsetFilter* bitset = nullptr) {
    const size_t vector_size_bytes = reader.dim() * data_type_size(reader.type());
    scan_ordered_iterator_with_dist(
        reader.base_begin(), count, heap, dist_fn, query, vector_size_bytes, bitset);
    scan_ordered_iterator_with_dist(
        reader.delta_begin(), count, heap, dist_fn, query, vector_size_bytes, bitset);
}

inline void scan_data_reader_with_query_norm(const DataReader& reader, size_t count, DistHeap* heap,
        CalcDistWithQueryNormFn dist_fn, const QueryCosContext& query,
        const BitsetFilter* bitset = nullptr) {
    const size_t vector_size_bytes = reader.dim() * data_type_size(reader.type());
    scan_ordered_iterator_with_query_norm(
        reader.base_begin(), count, heap, dist_fn, query, vector_size_bytes, bitset);
    scan_ordered_iterator_with_query_norm(
        reader.delta_begin(), count, heap, dist_fn, query, vector_size_bytes, bitset);
}

inline void scan_data_reader_with_cos_stored_norms(const DataReader& reader, size_t count,
        DistHeap* heap, CalcDotFn dot_fn, const QueryCosContext& query,
        const BitsetFilter* bitset = nullptr) {
    const size_t vector_size_bytes = reader.dim() * data_type_size(reader.type());
    scan_ordered_iterator_with_cos_stored_norms(
        reader.base_begin(), count, heap, dot_fn, query, vector_size_bytes, bitset);
    scan_ordered_iterator_with_cos_stored_norms(
        reader.delta_begin(), count, heap, dot_fn, query, vector_size_bytes, bitset);
}

inline void scan_data_reader_with_l2_stored_norms(const DataReader& reader, size_t count,
        DistHeap* heap, CalcDotFn dot_fn, const QueryL2Context& query,
        const BitsetFilter* bitset = nullptr) {
    const size_t vector_size_bytes = reader.dim() * data_type_size(reader.type());
    scan_ordered_iterator_with_l2_stored_norms(
        reader.base_begin(), count, heap, dot_fn, query, vector_size_bytes, bitset);
    scan_ordered_iterator_with_l2_stored_norms(
        reader.delta_begin(), count, heap, dot_fn, query, vector_size_bytes, bitset);
}

} // namespace sketch2
