// Shared dataset-level scanner traversal helpers.

#pragma once

#include "core/calc/scanner_heap_utils.h"
#include "core/calc/scanner_query_context.h"
#include "core/calc/scanner_scan_loops.h"
#include "core/storage/dataset_reader.h"
#include "core/utils/singleton.h"
#include "core/utils/thread_pool.h"

#include <exception>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace sketch2 {

inline Ret collect_dataset_readers(const DatasetReader& dataset, std::vector<DataReaderPtr>* readers) {
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
    return Ret(0);
}

template <typename ReaderScanFn>
inline Ret scan_dataset_readers(const std::vector<DataReaderPtr>& readers, size_t count, DistHeap* heap,
        const ReaderScanFn& scan_reader, DistFunc func, const BitsetFilter* bitset = nullptr) {
    if (count == 0) {
        return Ret(0);
    }

    const auto& pool = get_singleton().thread_pool();
    if (!pool || readers.size() < 2) {
        for (const auto& reader : readers) {
            scan_reader(*reader, count, heap, bitset);
        }
        return Ret(0);
    }

    const auto scan_reader_shared = std::make_shared<ReaderScanFn>(scan_reader);
    std::vector<std::future<DistHeap>> futures;
    futures.reserve(readers.size());
    for (const auto& reader : readers) {
        futures.push_back(pool->submit([scan_reader_shared, count, reader, func, bitset]() {
            DistHeap local_heap(DistItemCompare{func});
            local_heap.reserve(count);
            (*scan_reader_shared)(*reader, count, &local_heap, bitset);
            return local_heap;
        }));
    }

    std::exception_ptr first_error;
    std::vector<DistItem> merged_candidates;
    if (count > 0 && readers.size() <= (std::numeric_limits<size_t>::max() / count)) {
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
    return Ret(0);
}

inline Ret scan_dataset_heap_with_dist(const DatasetReader& dataset, size_t count, DistHeap* heap,
        CalcDistFn dist_fn, const QueryDistContext& query, DistFunc func,
        const BitsetFilter* bitset = nullptr) {
    std::vector<DataReaderPtr> readers;
    CHECK(collect_dataset_readers(dataset, &readers));
    return scan_dataset_readers(
        readers, count, heap,
        [dist_fn, query](const DataReader& reader, size_t local_count, DistHeap* local_heap,
                const BitsetFilter* bitset_filter) {
            scan_data_reader_with_dist(reader, local_count, local_heap, dist_fn, query, bitset_filter);
        },
        func,
        bitset);
}

inline Ret scan_dataset_heap_with_query_norm(const DatasetReader& dataset, size_t count, DistHeap* heap,
        CalcDistWithQueryNormFn dist_fn, const QueryCosContext& query,
        DistFunc func, const BitsetFilter* bitset = nullptr) {
    std::vector<DataReaderPtr> readers;
    CHECK(collect_dataset_readers(dataset, &readers));
    return scan_dataset_readers(
        readers, count, heap,
        [dist_fn, query](const DataReader& reader, size_t local_count,
                DistHeap* local_heap, const BitsetFilter* bitset_filter) {
            scan_data_reader_with_query_norm(
                reader, local_count, local_heap, dist_fn, query, bitset_filter);
        },
        func,
        bitset);
}

inline Ret scan_dataset_heap_with_optional_cosine_norms(const DatasetReader& dataset, size_t count,
        DistHeap* heap, CalcDotFn dot_fn, CalcDistWithQueryNormFn fallback_dist_fn,
        const QueryCosContext& query, DistFunc func, const BitsetFilter* bitset = nullptr) {
    assert(func == DistFunc::COS);
    std::vector<DataReaderPtr> readers;
    CHECK(collect_dataset_readers(dataset, &readers));
    return scan_dataset_readers(
        readers, count, heap,
        [dot_fn, fallback_dist_fn, query, func](
                const DataReader& reader, size_t local_count, DistHeap* local_heap,
                const BitsetFilter* bitset_filter) {
            if (reader.has_matching_stored_norms(func)) {
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

inline Ret scan_dataset_heap_with_optional_l2_norms(const DatasetReader& dataset, size_t count,
        DistHeap* heap, CalcDotFn dot_fn, CalcDistFn fallback_dist_fn,
        const QueryL2Context& query, DistFunc func, const BitsetFilter* bitset = nullptr) {
    assert(func == DistFunc::L2);
    std::vector<DataReaderPtr> readers;
    CHECK(collect_dataset_readers(dataset, &readers));
    return scan_dataset_readers(
        readers, count, heap,
        [dot_fn, fallback_dist_fn, query, func](
                const DataReader& reader, size_t local_count, DistHeap* local_heap,
                const BitsetFilter* bitset_filter) {
            if (reader.has_matching_stored_norms(func)) {
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
