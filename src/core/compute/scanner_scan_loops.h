// Shared scanner hot scan loops.

#pragma once

#include "core/compute/compute_kernels.h"
#include "core/compute/metric_finalizers.h"
#include "core/compute/scanner_heap_utils.h"
#include "core/compute/scanner_query_context.h"
#include "core/storage/data_reader.h"
#include "core/utils/bitset_filter.h"

#include <cassert>

namespace sketch2 {

inline uint64_t reader_heap_base_id(const DataReader& reader) {
    return reader.min_range_id_unchecked();
}

inline uint32_t normalize_reader_heap_id(uint64_t id, uint64_t heap_base_id) {
    return static_cast<uint32_t>(id - heap_base_id);
}

// Hint the hardware prefetcher with the head of the next record. Records sit
// contiguously in the mapped region, so one prefetch lets the HW stream
// prefetcher pull subsequent cache lines on its own.
inline void prefetch_vector_record(const uint8_t* data) {
    __builtin_prefetch(data, 0, 1);
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
        const BitsetFilter* bitset, const PushFn& push_fn) {
    size_t index = start_index;
    while (index < end_index) {
        const uint64_t id = reader.id_unchecked(index);
        const size_t next_index = index + 1;
        if (next_index < end_index) {
            prefetch_vector_record(reader.record_unchecked(next_index));
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
inline void scan_visible_base_records(const DataReader& reader,
        const BitsetFilter* bitset, const PushFn& push_fn) {
    const size_t end_index = reader.count_unchecked();
    if (end_index == 0) {
        return;
    }

    if (!reader.has_hidden_rows_unchecked()) {
        scan_reader_records<HasBitset>(reader, 0, end_index, bitset, push_fn);
        return;
    }

    size_t index = reader.first_visible_base_index_unchecked();
    while (index < end_index) {
        const uint64_t id = reader.id_unchecked(index);
        const size_t next_index = reader.next_visible_base_index_unchecked(index + 1);
        if (next_index < end_index) {
            prefetch_vector_record(reader.record_unchecked(next_index));
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
inline void scan_data_reader(const DataReader& reader,
        const BitsetFilter* bitset, const PushFn& push_fn) {
    scan_visible_base_records<HasBitset>(reader, bitset, push_fn);

    const DataReader* delta_reader = reader.delta_reader_unchecked();
    if (delta_reader == nullptr) {
        return;
    }

    scan_reader_records<HasBitset>(
        *delta_reader, 0, delta_reader->count_unchecked(), bitset, push_fn);
}

template <typename PushFn>
inline void scan_with_optional_bitset(const DataReader& reader,
        const BitsetFilter* bitset, const PushFn& push_fn) {
    if (bitset == nullptr) {
        scan_data_reader<false>(reader, nullptr, push_fn);
        return;
    }

    scan_data_reader<true>(reader, bitset, push_fn);
}

// Each leaf template takes the kernel as a non-type template parameter rather
// than a runtime function pointer, so GCC inlines the kernel into the record
// loop instead of emitting an indirect call per candidate.

// L2 path used when the reader has no stored norms: invokes a generic squared
// L2 kernel directly per record. No norms are needed because L2 = Σ(a-b)².
template <ComputeDistFn DistFn>
inline void scan_l2_fallback(const DataReader& reader, size_t count, LocalDistHeap* heap,
        const QueryDistContext& query, const BitsetFilter* bitset = nullptr) {
    assert(heap->comparator().smaller_score_better);
    const uint64_t heap_base_id = reader_heap_base_id(reader);
    scan_with_optional_bitset(reader, bitset,
        [heap, count, query, heap_base_id](uint64_t id, const uint8_t* record) {
            push_result_local(
                heap, count, normalize_reader_heap_id(id, heap_base_id),
                DistFn(record, query.vec, query.dim));
        });
}

// DOT metric: invokes a dot-product kernel per record.
template <ComputeDotFn DotFn>
inline void scan_dot(const DataReader& reader, size_t count, LocalDistHeap* heap,
        const QueryDotContext& query, const BitsetFilter* bitset = nullptr) {
    assert(!heap->comparator().smaller_score_better);
    const uint64_t heap_base_id = reader_heap_base_id(reader);
    scan_with_optional_bitset(reader, bitset,
        [heap, count, query, heap_base_id](uint64_t id, const uint8_t* record) {
            push_result_local(
                heap, count, normalize_reader_heap_id(id, heap_base_id),
                DotFn(record, query.vec, query.dim));
        });
}

// COS path used when the reader has no stored norms: invokes a kernel that
// receives the precomputed query norm and computes the record norm internally.
template <ComputeDistWithQueryNormFn DistFn>
inline void scan_cos_fallback(const DataReader& reader, size_t count, LocalDistHeap* heap,
        const QueryCosContext& query, const BitsetFilter* bitset = nullptr) {
    assert(heap->comparator().smaller_score_better);
    const uint64_t heap_base_id = reader_heap_base_id(reader);
    scan_with_optional_bitset(reader, bitset,
        [heap, count, query, heap_base_id](uint64_t id, const uint8_t* record) {
            push_result_local(
                heap, count, normalize_reader_heap_id(id, heap_base_id),
                DistFn(record, query.vec, query.dim, query.norm_sq));
        });
}

// COS path used when the reader has stored inline norms: reads each stored
// norm, calls a dot kernel, then combines them into the cosine distance.
template <ComputeDotFn DotFn>
inline void scan_cos_stored_norms(const DataReader& reader, size_t count,
        LocalDistHeap* heap, const QueryCosContext& query, const BitsetFilter* bitset = nullptr) {
    assert(heap->comparator().smaller_score_better);
    const size_t norm_offset_in_record = reader.norm_offset_in_record_unchecked();
    const uint64_t heap_base_id = reader_heap_base_id(reader);
    scan_with_optional_bitset(reader, bitset,
        [heap, count, query, norm_offset_in_record, heap_base_id](
                uint64_t id, const uint8_t* record) {
            float norm = 0.0f;
            std::memcpy(&norm, record + norm_offset_in_record, sizeof(norm));
            if (norm == 0.0f) {
                push_result_local(
                    heap, count, normalize_reader_heap_id(id, heap_base_id),
                    query.inv_norm == 0.0 ? 0.0 : 1.0);
                return;
            }
            const double dot = DotFn(record, query.vec, query.dim);
            push_result_local(
                heap, count, normalize_reader_heap_id(id, heap_base_id),
                cos_dist_from_inv_norms(
                    dot, static_cast<double>(norm), query.inv_norm));
        });
}

struct L2LowerBoundState {
    double query_norm = 0.0;
    double lo_stored_norm_sq = 0.0;
    double hi_stored_norm_sq = 0.0;
    bool heap_is_full = false;
    bool thresholds_valid = false;
};

inline void refresh_l2_lower_bound(const LocalDistHeap& heap, size_t count,
        L2LowerBoundState* state) {
    assert(heap.comparator().smaller_score_better);
    if (state->thresholds_valid) {
        return;
    }

    if (!state->heap_is_full) {
        if (heap.size() < count) {
            return;
        }
        state->heap_is_full = true;
    }

    const double heap_worst_score = std::max(0.0, static_cast<double>(heap.top().score));
    const double sqrt_heap_worst_score = std::sqrt(heap_worst_score);
    const double lo_stored_norm = std::max(0.0, state->query_norm - sqrt_heap_worst_score);
    const double hi_stored_norm = state->query_norm + sqrt_heap_worst_score;
    state->lo_stored_norm_sq = lo_stored_norm * lo_stored_norm;
    state->hi_stored_norm_sq = hi_stored_norm * hi_stored_norm;
    state->thresholds_valid = true;
}

inline void update_l2_lower_bound_after_push(const LocalDistHeap& heap, size_t count,
        bool heap_changed, L2LowerBoundState* state) {
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

inline bool l2_lower_bound_skips_record(const LocalDistHeap& heap, size_t count,
        double stored_norm_sq, L2LowerBoundState* state) {
    refresh_l2_lower_bound(heap, count, state);
    if (!state->thresholds_valid) {
        return false;
    }

    const double clamped_stored_norm_sq = std::max(0.0, stored_norm_sq);

    // Keep ties exact: an equal score can still win on smaller id, so only reject
    // when the lower bound is strictly worse than the current heap worst score.
    return clamped_stored_norm_sq < state->lo_stored_norm_sq
        || clamped_stored_norm_sq > state->hi_stored_norm_sq;
}

// L2 path used when the reader has stored inline norms: reads each stored
// norm, applies a triangle-inequality lower bound to skip unviable records,
// then calls a dot kernel and combines into the squared L2 distance.
template <ComputeDotFn DotFn>
inline void scan_l2_stored_norms(const DataReader& reader, size_t count,
        LocalDistHeap* heap, const QueryL2Context& query, const BitsetFilter* bitset = nullptr) {
    assert(heap->comparator().smaller_score_better);
    const size_t norm_offset_in_record = reader.norm_offset_in_record_unchecked();
    const uint64_t heap_base_id = reader_heap_base_id(reader);
    L2LowerBoundState lower_bound_state;
    lower_bound_state.query_norm = std::sqrt(std::max(0.0, query.norm_sq));
    scan_with_optional_bitset(reader, bitset,
        [heap, count, query, norm_offset_in_record, heap_base_id,
                lower_bound_state_ptr = &lower_bound_state](
                uint64_t id, const uint8_t* record) {
            float stored_norm_sq_f32 = 0.0f;
            std::memcpy(&stored_norm_sq_f32, record + norm_offset_in_record, sizeof(stored_norm_sq_f32));
            const double stored_norm_sq = static_cast<double>(stored_norm_sq_f32);

            if (l2_lower_bound_skips_record(
                    *heap, count, stored_norm_sq, lower_bound_state_ptr)) {
                return;
            }

            const double dot = DotFn(record, query.vec, query.dim);
            const bool heap_changed = push_result_local_changed(
                heap, count, normalize_reader_heap_id(id, heap_base_id),
                l2_dist_from_dot_and_norms(
                    dot, stored_norm_sq, query.norm_sq));
            update_l2_lower_bound_after_push(
                *heap, count, heap_changed, lower_bound_state_ptr);
        });
}

} // namespace sketch2
