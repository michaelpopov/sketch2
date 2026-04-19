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
inline void scan_reader_records(const DataReader& reader, size_t start_index, size_t end_index,
        size_t record_stride_bytes, const BitsetFilter* bitset, const PushFn& push_fn) {
    size_t index = start_index;
    while (index < end_index) {
        const uint64_t id = reader.id_unchecked(index);
        const size_t next_index = index + 1;
        if (next_index < end_index) {
            prefetch_vector_record(reader.record_unchecked(next_index), record_stride_bytes);
        }
        if constexpr (HasBitset) {
            if (!bitset_contains_id(bitset, id)) {
                index = next_index;
                continue;
            }
        }
        push_fn(id, reader.record_unchecked(index));
        index = next_index;
    }
}

template <bool HasBitset, typename PushFn>
inline void scan_visible_base_records(const DataReader& reader, size_t record_stride_bytes,
        const BitsetFilter* bitset, const PushFn& push_fn) {
    const size_t end_index = reader.count_unchecked();
    if (end_index == 0) {
        return;
    }

    if (!reader.has_hidden_rows_unchecked()) {
        scan_reader_records<HasBitset>(reader, 0, end_index, record_stride_bytes, bitset, push_fn);
        return;
    }

    size_t index = reader.first_visible_base_index_unchecked();
    while (index < end_index) {
        const uint64_t id = reader.id_unchecked(index);
        const size_t next_index = reader.next_visible_base_index_unchecked(index + 1);
        if (next_index < end_index) {
            prefetch_vector_record(reader.record_unchecked(next_index), record_stride_bytes);
        }
        if constexpr (HasBitset) {
            if (!bitset_contains_id(bitset, id)) {
                index = next_index;
                continue;
            }
        }
        push_fn(id, reader.record_unchecked(index));
        index = next_index;
    }
}

template <bool HasBitset, typename PushFn>
inline void scan_data_reader(const DataReader& reader, size_t record_stride_bytes,
        const BitsetFilter* bitset, const PushFn& push_fn) {
    scan_visible_base_records<HasBitset>(
        reader, record_stride_bytes, bitset, push_fn);

    const DataReader* delta_reader = reader.delta_reader_unchecked();
    if (delta_reader == nullptr) {
        return;
    }

    scan_reader_records<HasBitset>(
        *delta_reader, 0, delta_reader->count_unchecked(), record_stride_bytes, bitset, push_fn);
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
        [heap, count, dist_fn, query](uint64_t id, const uint8_t* record) {
            push_result_smaller_better(heap, count, id, dist_fn(record, query.vec, query.dim));
        });
}

inline void scan_data_reader_with_dot(const DataReader& reader, size_t count, DistHeap* heap,
        ComputeDotFn dot_fn, const QueryDotContext& query, const BitsetFilter* bitset = nullptr) {
    const size_t record_stride_bytes = reader.stride();
    scan_data_reader_with_optional_bitset(reader, record_stride_bytes, bitset,
        [heap, count, dot_fn, query](uint64_t id, const uint8_t* record) {
            push_result_larger_better(heap, count, id, dot_fn(record, query.vec, query.dim));
        });
}

inline void scan_data_reader_with_query_norm(const DataReader& reader, size_t count, DistHeap* heap,
        ComputeDistWithQueryNormFn dist_fn, const QueryCosContext& query,
        const BitsetFilter* bitset = nullptr) {
    const size_t record_stride_bytes = reader.stride();
    scan_data_reader_with_optional_bitset(reader, record_stride_bytes, bitset,
        [heap, count, dist_fn, query](uint64_t id, const uint8_t* record) {
            push_result_smaller_better(heap, count, id,
                dist_fn(record, query.vec, query.dim, query.norm_sq));
        });
}

inline void scan_data_reader_with_cos_stored_norms(const DataReader& reader, size_t count,
        DistHeap* heap, ComputeDotFn dot_fn, const QueryCosContext& query,
        const BitsetFilter* bitset = nullptr) {
    const size_t record_stride_bytes = reader.stride();
    const size_t norm_offset_in_record = reader.norm_offset_in_record_unchecked();
    scan_data_reader_with_optional_bitset(reader, record_stride_bytes, bitset,
        [heap, count, dot_fn, query, norm_offset_in_record](uint64_t id, const uint8_t* record) {
            const double dot = dot_fn(record, query.vec, query.dim);
            float norm = 0.0f;
            std::memcpy(&norm, record + norm_offset_in_record, sizeof(norm));
            push_result_smaller_better(heap, count, id, finalize_cosine_distance_from_inverse_norms(
                dot, static_cast<double>(norm), query.inv_norm));
        });
}

struct L2StoredNormLowerBoundState {
    double query_norm = 0.0;
    double lo_stored_norm_sq = 0.0;
    double hi_stored_norm_sq = 0.0;
    bool heap_is_full = false;
    bool thresholds_valid = false;
};

inline void refresh_l2_stored_norm_lower_bound(const DistHeap& heap, size_t count,
        L2StoredNormLowerBoundState* state) {
    if (state->thresholds_valid) {
        return;
    }

    if (!state->heap_is_full) {
        if (heap.size() < count) {
            return;
        }
        state->heap_is_full = true;
    }

    const double heap_worst_score = std::max(0.0, heap.top().score);
    const double sqrt_heap_worst_score = std::sqrt(heap_worst_score);
    const double lo_stored_norm = std::max(0.0, state->query_norm - sqrt_heap_worst_score);
    const double hi_stored_norm = state->query_norm + sqrt_heap_worst_score;
    state->lo_stored_norm_sq = lo_stored_norm * lo_stored_norm;
    state->hi_stored_norm_sq = hi_stored_norm * hi_stored_norm;
    state->thresholds_valid = true;
}

inline void update_l2_stored_norm_lower_bound_after_push(const DistHeap& heap, size_t count,
        bool heap_changed, L2StoredNormLowerBoundState* state) {
    if (!state->heap_is_full) {
        if (heap.size() < count) {
            return;
        }
        state->heap_is_full = true;
        state->thresholds_valid = false;
        return;
    }

    if (heap_changed) {
        state->thresholds_valid = false;
    }
}

inline bool l2_stored_norm_lower_bound_exceeds_heap(const DistHeap& heap, size_t count,
        double stored_norm_sq, L2StoredNormLowerBoundState* state) {
    refresh_l2_stored_norm_lower_bound(heap, count, state);
    if (!state->thresholds_valid) {
        return false;
    }

    const double clamped_stored_norm_sq = std::max(0.0, stored_norm_sq);

    // Keep ties exact: an equal score can still win on smaller id, so only reject
    // when the lower bound is strictly worse than the current heap worst score.
    return clamped_stored_norm_sq < state->lo_stored_norm_sq
        || clamped_stored_norm_sq > state->hi_stored_norm_sq;
}

inline void scan_data_reader_with_l2_stored_norms(const DataReader& reader, size_t count,
        DistHeap* heap, ComputeDotFn dot_fn, const QueryL2Context& query,
        const BitsetFilter* bitset = nullptr) {
    const size_t record_stride_bytes = reader.stride();
    const size_t norm_offset_in_record = reader.norm_offset_in_record_unchecked();
    L2StoredNormLowerBoundState lower_bound_state;
    lower_bound_state.query_norm = std::sqrt(std::max(0.0, query.norm_sq));
    scan_data_reader_with_optional_bitset(reader, record_stride_bytes, bitset,
        [heap, count, dot_fn, query, norm_offset_in_record, lower_bound_state_ptr = &lower_bound_state](
                uint64_t id, const uint8_t* record) {
            float stored_norm_sq_f32 = 0.0f;
            std::memcpy(&stored_norm_sq_f32, record + norm_offset_in_record, sizeof(stored_norm_sq_f32));
            const double stored_norm_sq = static_cast<double>(stored_norm_sq_f32);

            // Read the inline squared norm first so the cheap lower bound can skip
            // the expensive dot kernel for candidates that cannot beat the full heap.
            if (l2_stored_norm_lower_bound_exceeds_heap(
                    *heap, count, stored_norm_sq, lower_bound_state_ptr)) {
                return;
            }

            const double dot = dot_fn(record, query.vec, query.dim);
            const bool heap_changed = push_result_smaller_better_changed(
                heap, count, id, finalize_squared_l2_distance_from_squared_norms(
                    dot, stored_norm_sq, query.norm_sq));
            update_l2_stored_norm_lower_bound_after_push(
                *heap, count, heap_changed, lower_bound_state_ptr);
        });
}

} // namespace sketch2
