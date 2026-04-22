// Shared dataset-level scanner traversal helpers.

#pragma once

#include "core/compute/scanner_heap_utils.h"
#include "core/compute/scanner_log_utils.h"
#include "core/compute/scanner_query_context.h"
#include "core/compute/scanner_scan_loops.h"
#include "core/storage/dataset_reader.h"
#include "core/utils/singleton.h"
#include "core/utils/thread_pool.h"

#include <exception>
#include <future>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sketch2 {

inline void merge_reader_local_heap_into_final_heap(
        DistHeapEx* local_heap, uint64_t heap_base_id, size_t count, DistHeap* final_heap) {
    while (!local_heap->empty()) {
        const DistItemEx item = local_heap->pop_top();
        push_result(final_heap, count, item.id + heap_base_id, item.score);
    }
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
        DistHeap* heap, const ReaderScanFn& scan_reader, DistFunc func,
        const BitsetFilter* bitset = nullptr) {
    if (count == 0) {
        return Ret(0);
    }

    const auto& pool = get_singleton().thread_pool();
    const bool has_thread_pool = static_cast<bool>(pool);
    const bool uses_parallel = has_thread_pool && readers.size() >= 2;
    log_dataset_scan_mode(query_id, readers.size(), has_thread_pool, uses_parallel, bitset != nullptr);

    if (!uses_parallel) {
        if (readers.size() == 1) {
            DistHeapEx local_heap(DistItemExCompare{func});
            local_heap.reserve(count);
            scan_reader(*readers[0], count, &local_heap, bitset);
            rebuild_final_heap_from_local_heap(&local_heap, reader_heap_base_id(*readers[0]), heap);
            return Ret(0);
        }

        for (size_t i = 0; i < readers.size(); ++i) {
            DistHeapEx local_heap(DistItemExCompare{func});
            local_heap.reserve(count);
            scan_reader(*readers[i], count, &local_heap, bitset);
            merge_reader_local_heap_into_final_heap(
                &local_heap, reader_heap_base_id(*readers[i]), count, heap);
        }
        return Ret(0);
    }

    std::vector<std::future<DistHeapEx>> futures;
    futures.reserve(readers.size());
    for (const auto& reader : readers) {
        futures.push_back(pool->submit([scan_reader, count, reader, func, bitset]() {
            DistHeapEx local_heap(DistItemExCompare{func});
            local_heap.reserve(count);
            scan_reader(*reader, count, &local_heap, bitset);
            return local_heap;
        }));
    }

    std::exception_ptr first_error;
    size_t merged_candidates = 0;
    for (size_t i = 0; i < futures.size(); ++i) {
        try {
            DistHeapEx local_heap = futures[i].get();
            if (first_error != nullptr) {
                continue;
            }
            merged_candidates += local_heap.size();
            merge_reader_local_heap_into_final_heap(
                &local_heap, reader_heap_base_id(*readers[i]), count, heap);
        } catch (...) {
            if (first_error == nullptr) {
                first_error = std::current_exception();
            }
        }
    }

    if (first_error != nullptr) {
        std::rethrow_exception(first_error);
    }
    log_parallel_reader_merge(query_id, readers.size(), merged_candidates, heap->size());
    return Ret(0);
}

inline Ret scan_dataset_heap_with_dist_rt(uint64_t query_id, const DatasetReader& dataset, size_t count,
        DistHeap* heap, const QueryDistContext& query, DistFunc func, ComputeDistFn dist_fn,
        const BitsetFilter* bitset = nullptr) {
    std::vector<DataReaderPtr> readers;
    CHECK(collect_dataset_readers(dataset, query_id, &readers));
    return scan_dataset_readers(
        query_id, readers, count, heap,
        [query_id, query, dist_fn](const DataReader& reader, size_t local_count, DistHeapEx* local_heap,
                const BitsetFilter* bitset_filter) {
            log_reader_scan_plan(query_id, reader, "raw_dist", false, bitset_filter != nullptr);
            scan_data_reader_with_dist_rt(reader, local_count, local_heap, query, dist_fn, bitset_filter);
        },
        func,
        bitset);
}

template <ComputeDistFn DistFn>
inline Ret scan_dataset_heap_with_dist(uint64_t query_id, const DatasetReader& dataset, size_t count,
        DistHeap* heap, const QueryDistContext& query, DistFunc func,
        const BitsetFilter* bitset = nullptr) {
    return scan_dataset_heap_with_dist_rt(query_id, dataset, count, heap, query, func, DistFn, bitset);
}

inline Ret scan_dataset_heap_with_dot_rt(uint64_t query_id, const DatasetReader& dataset, size_t count,
        DistHeap* heap, const QueryDotContext& query, DistFunc func, ComputeDotFn dot_fn,
        const BitsetFilter* bitset = nullptr) {
    assert(func == DistFunc::DOT);
    std::vector<DataReaderPtr> readers;
    CHECK(collect_dataset_readers(dataset, query_id, &readers));
    return scan_dataset_readers(
        query_id, readers, count, heap,
        [query_id, query, dot_fn](const DataReader& reader, size_t local_count, DistHeapEx* local_heap,
                const BitsetFilter* bitset_filter) {
            log_reader_scan_plan(query_id, reader, "dot", false, bitset_filter != nullptr);
            scan_data_reader_with_dot_rt(reader, local_count, local_heap, query, dot_fn, bitset_filter);
        },
        func,
        bitset);
}

template <ComputeDotFn DotFn>
inline Ret scan_dataset_heap_with_dot(uint64_t query_id, const DatasetReader& dataset, size_t count,
        DistHeap* heap, const QueryDotContext& query, DistFunc func,
        const BitsetFilter* bitset = nullptr) {
    return scan_dataset_heap_with_dot_rt(query_id, dataset, count, heap, query, func, DotFn, bitset);
}

inline Ret scan_dataset_heap_with_optional_cosine_norms_rt(uint64_t query_id,
        const DatasetReader& dataset, size_t count, DistHeap* heap, const QueryCosContext& query,
        DistFunc func, ComputeDotFn dot_fn, ComputeDistWithQueryNormFn fallback_dist_fn,
        const BitsetFilter* bitset = nullptr) {
    assert(func == DistFunc::COS);
    std::vector<DataReaderPtr> readers;
    CHECK(collect_dataset_readers(dataset, query_id, &readers));
    return scan_dataset_readers(
        query_id, readers, count, heap,
        [query, func, query_id, dot_fn, fallback_dist_fn](
                const DataReader& reader, size_t local_count, DistHeapEx* local_heap,
                const BitsetFilter* bitset_filter) {
            const bool uses_stored_norms = reader.has_matching_stored_norms(func);
            log_reader_scan_plan(query_id, reader,
                uses_stored_norms ? "cos_stored_norms" : "cos_query_norm_fallback",
                uses_stored_norms, bitset_filter != nullptr);
            if (uses_stored_norms) {
                scan_data_reader_with_cos_stored_norms_rt(
                    reader, local_count, local_heap, query, dot_fn, bitset_filter);
            } else {
                scan_data_reader_with_query_norm_rt(
                    reader, local_count, local_heap, query, fallback_dist_fn, bitset_filter);
            }
        },
        func,
        bitset);
}

template <ComputeDotFn DotFn, ComputeDistWithQueryNormFn FallbackDistFn>
inline Ret scan_dataset_heap_with_optional_cosine_norms(uint64_t query_id,
        const DatasetReader& dataset, size_t count, DistHeap* heap, const QueryCosContext& query,
        DistFunc func, const BitsetFilter* bitset = nullptr) {
    return scan_dataset_heap_with_optional_cosine_norms_rt(
        query_id, dataset, count, heap, query, func, DotFn, FallbackDistFn, bitset);
}

inline Ret scan_dataset_heap_with_optional_l2_norms_rt(uint64_t query_id,
        const DatasetReader& dataset, size_t count, DistHeap* heap, const QueryL2Context& query,
        DistFunc func, ComputeDotFn dot_fn, ComputeDistFn fallback_dist_fn,
        const BitsetFilter* bitset = nullptr) {
    assert(func == DistFunc::L2);
    std::vector<DataReaderPtr> readers;
    CHECK(collect_dataset_readers(dataset, query_id, &readers));
    return scan_dataset_readers(
        query_id, readers, count, heap,
        [query, func, query_id, dot_fn, fallback_dist_fn](
                const DataReader& reader, size_t local_count, DistHeapEx* local_heap,
                const BitsetFilter* bitset_filter) {
            const bool uses_stored_norms = reader.has_matching_stored_norms(func);
            log_reader_scan_plan(query_id, reader,
                uses_stored_norms ? "l2_stored_norms" : "l2_dist_fallback",
                uses_stored_norms, bitset_filter != nullptr);
            if (uses_stored_norms) {
                scan_data_reader_with_l2_stored_norms_rt(
                    reader, local_count, local_heap, query, dot_fn, bitset_filter);
            } else {
                scan_data_reader_with_dist_rt(
                    reader, local_count, local_heap,
                    QueryDistContext{query.vec, query.dim}, fallback_dist_fn, bitset_filter);
            }
        },
        func,
        bitset);
}

template <ComputeDotFn DotFn, ComputeDistFn FallbackDistFn>
inline Ret scan_dataset_heap_with_optional_l2_norms(uint64_t query_id,
        const DatasetReader& dataset, size_t count, DistHeap* heap, const QueryL2Context& query,
        DistFunc func, const BitsetFilter* bitset = nullptr) {
    return scan_dataset_heap_with_optional_l2_norms_rt(
        query_id, dataset, count, heap, query, func, DotFn, FallbackDistFn, bitset);
}

} // namespace sketch2
