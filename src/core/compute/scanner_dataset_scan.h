// Shared dataset-level scanner traversal helpers.

#pragma once

#include "core/compute/scanner_heap_utils.h"
#include "core/compute/scanner_log_utils.h"
#include "core/compute/scanner_scan_loops.h"
#include "core/storage/dataset_reader.h"
#include "core/utils/singleton.h"
#include "core/utils/thread_pool.h"

#include <algorithm>
#include <exception>
#include <future>
#include <vector>

namespace sketch2 {

inline void merge_local_heap(
        LocalDistHeap* local_heap, uint64_t heap_base_id, size_t count, DistHeap* final_heap) {
    // Insertion order into the bounded final heap doesn't matter -- it maintains
    // its own top-k invariant -- so iterate the backing array directly rather
    // than draining via pop_top(), which would sift down O(log n) per item. This
    // mirrors how the single-reader fast path reads data() in
    // sort_local_heap_into_result().
    for (const LocalDistItem& item : local_heap->data()) {
        push_result(final_heap, count, item.id + heap_base_id, item.score);
    }
}

// Total rows a reader can contribute to a scan: its base records plus any
// attached delta records.
inline size_t reader_total_rows(const DataReader& reader) {
    size_t rows = reader.count_unchecked();
    if (const DataReader* delta = reader.delta_reader_unchecked()) {
        rows += delta->count_unchecked();
    }
    return rows;
}

// Upper bound on items a reader can place in a top-k heap of size `count`. A
// reader never yields more than its own row total, so clamping here keeps the
// bounded top-k policy identical while preventing a large k against a small
// reader from pre-allocating capacity for rows that cannot exist.
inline size_t reader_heap_capacity_hint(const DataReader& reader, size_t count) {
    return std::min(count, reader_total_rows(reader));
}

inline Ret collect_dataset_readers(const DatasetReader& dataset, uint64_t query_id,
        std::vector<DataReaderPtr>* readers) {
    readers->clear();
    auto drs = dataset.reader();
    while (true) {
        auto [reader, ret] = drs->next();
        CHECK(ret);
        if (!reader) {
            break;
        }
        readers->push_back(std::move(reader));
    }
    log_collected_dataset_readers(query_id, dataset.name(), readers->size());
    return Ret(0);
}

template <typename ReaderScanFn>
inline Ret scan_dataset_readers(uint64_t query_id, const std::vector<DataReaderPtr>& readers, size_t count,
        std::vector<DistItem>* result, const ReaderScanFn& scan_reader, DistFunc func,
        const BitsetFilter* bitset = nullptr) {
    if (count == 0) {
        return Ret(0);
    }

    const auto& pool = get_singleton().thread_pool();
    const bool has_thread_pool = static_cast<bool>(pool);
    const bool uses_parallel = has_thread_pool && readers.size() >= 2;
    log_dataset_scan_mode(query_id, readers.size(), has_thread_pool, uses_parallel, bitset != nullptr);

    if (readers.size() == 1) {
        // Single-reader fast path: scan into a local heap and convert directly
        // into the result vector, skipping the final-heap construction.
        LocalDistHeap local_heap(LocalDistItemCompare{func});
        local_heap.reserve(reader_heap_capacity_hint(*readers[0], count));
        scan_reader(*readers[0], count, &local_heap, bitset);
        sort_local_heap_into_result(
            &local_heap, reader_heap_base_id(*readers[0]), func, result);
        return Ret(0);
    }

    // The merged heap holds the global top-k, so it can never exceed the total
    // rows across all readers. Clamp the same way as the per-reader heaps.
    size_t total_reader_rows = 0;
    for (const auto& reader : readers) {
        total_reader_rows += reader_total_rows(*reader);
    }
    DistHeap final_heap(DistItemCompare{func});
    final_heap.reserve(std::min(count, total_reader_rows));

    if (!uses_parallel) {
        for (size_t i = 0; i < readers.size(); ++i) {
            LocalDistHeap local_heap(LocalDistItemCompare{func});
            local_heap.reserve(reader_heap_capacity_hint(*readers[i], count));
            scan_reader(*readers[i], count, &local_heap, bitset);
            merge_local_heap(
                &local_heap, reader_heap_base_id(*readers[i]), count, &final_heap);
        }
        extract_items(&final_heap, result);
        return Ret(0);
    }

    std::vector<std::future<LocalDistHeap>> futures;
    futures.reserve(readers.size());

    // If submit() throws partway through (pool shutting down, or OOM), the tasks
    // already enqueued keep running on worker threads and still dereference
    // caller-owned memory (the query vector and bitset). Capture the error but
    // fall through to the drain loop so every submitted task is waited on before
    // this scope exits.
    std::exception_ptr first_error;
    try {
        for (const auto& reader : readers) {
            futures.push_back(pool->submit([scan_reader, count, reader, func, bitset]() {
                LocalDistHeap local_heap(LocalDistItemCompare{func});
                local_heap.reserve(reader_heap_capacity_hint(*reader, count));
                scan_reader(*reader, count, &local_heap, bitset);
                return local_heap;
            }));
        }
    } catch (...) {
        first_error = std::current_exception();
    }

    size_t merged_candidates = 0;
    for (size_t i = 0; i < futures.size(); ++i) {
        try {
            LocalDistHeap local_heap = futures[i].get();
            if (first_error != nullptr) {
                continue;
            }
            merged_candidates += local_heap.size();
            merge_local_heap(
                &local_heap, reader_heap_base_id(*readers[i]), count, &final_heap);
        } catch (...) {
            if (first_error == nullptr) {
                first_error = std::current_exception();
            }
        }
    }

    if (first_error != nullptr) {
        std::rethrow_exception(first_error);
    }
    log_parallel_reader_merge(query_id, readers.size(), merged_candidates, final_heap.size());
    extract_items(&final_heap, result);
    return Ret(0);
}

} // namespace sketch2
