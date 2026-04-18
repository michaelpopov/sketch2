// Shared scanner building blocks reused by the Highway and NumKong scanners.

#pragma once

#include "core/calc/calc_engine.h"
#include "core/calc/cosine_distance.h"
#include "core/calc/dist_item.h"
#include "core/storage/data_reader.h"
#include "core/storage/dataset_reader.h"
#include "core/utils/bitset_filter.h"
#include "core/utils/log.h"
#include "core/utils/singleton.h"
#include "core/utils/thread_pool.h"
#include "core/utils/timer.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <future>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <vector>

namespace sketch2 {

const char* calc_engine_name(CalcEngine engine);

class DistHeap : public std::priority_queue<DistItem, std::vector<DistItem>, DistItemCompare> {
public:
    using std::priority_queue<DistItem, std::vector<DistItem>, DistItemCompare>::priority_queue;

    const DistItemCompare& comparator() const {
        return this->comp;
    }

    void reserve(size_t capacity) {
        this->c.reserve(capacity);
    }
};

inline const char* scanner_dist_func_name(DistFunc func) {
    switch (func) {
        case DistFunc::DOT: return "DOT";
        case DistFunc::L2: return "L2";
        case DistFunc::COS: return "COS";
        default: return "unknown";
    }
}

inline void log_query(const std::string& source, DistFunc func, DataType type, size_t dim,
        size_t count, CalcEngine engine, int64_t elapsed_ms) {
    LOG_TRACE << "ScannerEx query: source=" << source
              << " engine=" << calc_engine_name(engine)
              << " metric=" << scanner_dist_func_name(func)
              << " type=" << data_type_to_string(type)
              << " dim=" << dim
              << " k=" << count
              << " time=" << elapsed_ms << " ms";
}

inline void push_result(DistHeap* heap, size_t count, uint64_t id, double score) {
    const DistItem item{id, score};
    if (heap->size() < count) {
        heap->push(item);
    } else if (heap->comparator()(item, heap->top())) {
        heap->pop();
        heap->push(item);
    }
}

inline void extract_items(DistHeap* heap, std::vector<DistItem>* result) {
    result->resize(heap->size());
    for (size_t i = heap->size(); i-- > 0;) {
        (*result)[i] = heap->top();
        heap->pop();
    }
}

inline void extract_ids_from_items(const std::vector<DistItem>& items, std::vector<uint64_t>* result) {
    result->clear();
    result->reserve(items.size());
    for (const auto& item : items) {
        result->push_back(item.id);
    }
}

inline double query_inverse_norm(double query_norm_sq) {
    if (query_norm_sq == 0.0) {
        return 0.0;
    }
    return 1.0 / std::sqrt(query_norm_sq);
}

inline double finalize_squared_l2_distance_from_squared_norms(
        double dot, double norm_a_sq, double norm_b_sq) {
    return std::max(0.0, norm_a_sq + norm_b_sq - (2.0 * dot));
}

struct QueryDistContext {
    const uint8_t* vec = nullptr;
    size_t dim = 0;
};

struct QueryCosContext {
    const uint8_t* vec = nullptr;
    size_t dim = 0;
    double norm_sq = 0.0;
    double inv_norm = 0.0;
};

struct QueryL2Context {
    const uint8_t* vec = nullptr;
    size_t dim = 0;
    double norm_sq = 0.0;
};

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
