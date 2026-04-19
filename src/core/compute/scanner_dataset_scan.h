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
        for (const auto& reader : readers) {
            scan_reader(*reader, count, heap, bitset);
        }
        return Ret(0);
    }

    std::vector<std::future<DistHeap>> futures;
    futures.reserve(readers.size());
    for (const auto& reader : readers) {
        futures.push_back(pool->submit([scan_reader, count, reader, func, bitset]() {
            DistHeap local_heap(DistItemCompare{func});
            local_heap.reserve(count);
            scan_reader(*reader, count, &local_heap, bitset);
            return local_heap;
        }));
    }

    std::exception_ptr first_error;
    std::vector<DistItem> merged_candidates;
    if (readers.size() <= (std::numeric_limits<size_t>::max() / count)) {
        merged_candidates.reserve(readers.size() * count);
    }
    for (auto& fut : futures) {
        try {
            DistHeap local_heap = fut.get();
            if (first_error != nullptr) {
                continue;
            }
            while (!local_heap.empty()) {
                merged_candidates.push_back(local_heap.top());
                local_heap.pop();
            }
        } catch (...) {
            if (first_error == nullptr) {
                first_error = std::current_exception();
            }
        }
    }

    if (first_error != nullptr) {
        std::rethrow_exception(first_error);
    }

    if (merged_candidates.size() > count) {
        const DistItemCompare better(func);
        std::nth_element(
            merged_candidates.begin(),
            merged_candidates.begin() + count,
            merged_candidates.end(),
            better);
        merged_candidates.resize(count);
    }

    for (const auto& item : merged_candidates) {
        push_result(heap, count, item.id, item.score);
    }
    log_parallel_reader_merge(query_id, readers.size(), merged_candidates.size(), heap->size());
    return Ret(0);
}

inline Ret scan_dataset_heap_with_dist(uint64_t query_id, const DatasetReader& dataset, size_t count,
        DistHeap* heap, ComputeDistFn dist_fn, const QueryDistContext& query, DistFunc func,
        const BitsetFilter* bitset = nullptr) {
    std::vector<DataReaderPtr> readers;
    CHECK(collect_dataset_readers(dataset, query_id, &readers));
    return scan_dataset_readers(
        query_id, readers, count, heap,
        [dist_fn, query, query_id](const DataReader& reader, size_t local_count, DistHeap* local_heap,
                const BitsetFilter* bitset_filter) {
            log_reader_scan_plan(query_id, reader, "raw_dist", false, bitset_filter != nullptr);
            scan_data_reader_with_dist(reader, local_count, local_heap, dist_fn, query, bitset_filter);
        },
        func,
        bitset);
}

inline Ret scan_dataset_heap_with_dot(uint64_t query_id, const DatasetReader& dataset, size_t count,
        DistHeap* heap, ComputeDotFn dot_fn, const QueryDotContext& query, DistFunc func,
        const BitsetFilter* bitset = nullptr) {
    assert(func == DistFunc::DOT);
    std::vector<DataReaderPtr> readers;
    CHECK(collect_dataset_readers(dataset, query_id, &readers));
    return scan_dataset_readers(
        query_id, readers, count, heap,
        [dot_fn, query, query_id](const DataReader& reader, size_t local_count, DistHeap* local_heap,
                const BitsetFilter* bitset_filter) {
            log_reader_scan_plan(query_id, reader, "dot", false, bitset_filter != nullptr);
            scan_data_reader_with_dot(reader, local_count, local_heap, dot_fn, query, bitset_filter);
        },
        func,
        bitset);
}

inline Ret scan_dataset_heap_with_query_norm(uint64_t query_id, const DatasetReader& dataset, size_t count,
        DistHeap* heap, ComputeDistWithQueryNormFn dist_fn, const QueryCosContext& query,
        DistFunc func, const BitsetFilter* bitset = nullptr) {
    std::vector<DataReaderPtr> readers;
    CHECK(collect_dataset_readers(dataset, query_id, &readers));
    return scan_dataset_readers(
        query_id, readers, count, heap,
        [dist_fn, query, query_id](const DataReader& reader, size_t local_count,
                DistHeap* local_heap, const BitsetFilter* bitset_filter) {
            log_reader_scan_plan(query_id, reader, "cos_query_norm", false, bitset_filter != nullptr);
            scan_data_reader_with_query_norm(
                reader, local_count, local_heap, dist_fn, query, bitset_filter);
        },
        func,
        bitset);
}

inline Ret scan_dataset_heap_with_optional_cosine_norms(uint64_t query_id,
        const DatasetReader& dataset, size_t count, DistHeap* heap, ComputeDotFn dot_fn,
        ComputeDistWithQueryNormFn fallback_dist_fn, const QueryCosContext& query, DistFunc func,
        const BitsetFilter* bitset = nullptr) {
    assert(func == DistFunc::COS);
    std::vector<DataReaderPtr> readers;
    CHECK(collect_dataset_readers(dataset, query_id, &readers));
    return scan_dataset_readers(
        query_id, readers, count, heap,
        [dot_fn, fallback_dist_fn, query, func, query_id](
                const DataReader& reader, size_t local_count, DistHeap* local_heap,
                const BitsetFilter* bitset_filter) {
            const bool uses_stored_norms = reader.has_matching_stored_norms(func);
            log_reader_scan_plan(query_id, reader,
                uses_stored_norms ? "cos_stored_norms" : "cos_query_norm_fallback",
                uses_stored_norms, bitset_filter != nullptr);
            if (uses_stored_norms) {
                scan_data_reader_with_cos_stored_norms(
                    reader, local_count, local_heap, dot_fn, query, bitset_filter);
            } else {
                scan_data_reader_with_query_norm(
                    reader, local_count, local_heap, fallback_dist_fn, query, bitset_filter);
            }
        },
        func,
        bitset);
}

inline Ret scan_dataset_heap_with_optional_l2_norms(uint64_t query_id,
        const DatasetReader& dataset, size_t count, DistHeap* heap, ComputeDotFn dot_fn,
        ComputeDistFn fallback_dist_fn, const QueryL2Context& query, DistFunc func,
        const BitsetFilter* bitset = nullptr) {
    assert(func == DistFunc::L2);
    std::vector<DataReaderPtr> readers;
    CHECK(collect_dataset_readers(dataset, query_id, &readers));
    return scan_dataset_readers(
        query_id, readers, count, heap,
        [dot_fn, fallback_dist_fn, query, func, query_id](
                const DataReader& reader, size_t local_count, DistHeap* local_heap,
                const BitsetFilter* bitset_filter) {
            const bool uses_stored_norms = reader.has_matching_stored_norms(func);
            log_reader_scan_plan(query_id, reader,
                uses_stored_norms ? "l2_stored_norms" : "l2_dist_fallback",
                uses_stored_norms, bitset_filter != nullptr);
            if (uses_stored_norms) {
                scan_data_reader_with_l2_stored_norms(
                    reader, local_count, local_heap, dot_fn, query, bitset_filter);
            } else {
                scan_data_reader_with_dist(
                    reader, local_count, local_heap, fallback_dist_fn,
                    QueryDistContext{query.vec, query.dim}, bitset_filter);
            }
        },
        func,
        bitset);
}

} // namespace sketch2
