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

constexpr size_t kPrefetchCacheLineBytes = 64;

template <typename Iterator>
inline void prefetch_next_vector_record(const Iterator& it, size_t vector_size_bytes) {
    Iterator next_it = it;
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

struct FnPtrDistScore {
    CalcDistFn fn;
    const uint8_t* vec;
    size_t dim;

    template <typename Iterator>
    double operator()(const Iterator& it) const {
        return fn(it.data(), vec, dim);
    }
};

struct FnPtrQueryNormScore {
    CalcDistWithQueryNormFn fn;
    const uint8_t* vec;
    size_t dim;
    double query_norm_sq;

    template <typename Iterator>
    double operator()(const Iterator& it) const {
        return fn(it.data(), vec, dim, query_norm_sq);
    }
};

struct FnPtrInvNormScore {
    CalcDotFn fn;
    const uint8_t* vec;
    size_t dim;
    double query_inv_norm;

    template <typename Iterator>
    double operator()(const Iterator& it) const {
        assert(query_inv_norm >= 0.0);
        const double dot = fn(it.data(), vec, dim);
        return finalize_cosine_distance_from_inverse_norms(
            dot, static_cast<double>(it.get_norm()), query_inv_norm);
    }
};

struct FnPtrStoredSquaredNormL2Score {
    CalcDotFn fn;
    const uint8_t* vec;
    size_t dim;
    double query_norm_sq;

    template <typename Iterator>
    double operator()(const Iterator& it) const {
        const double dot = fn(it.data(), vec, dim);
        return finalize_squared_l2_distance_from_squared_norms(
            dot, static_cast<double>(it.get_norm()), query_norm_sq);
    }
};

template <typename Iterator, typename ScoreFn>
inline void scan_iterator_scored(Iterator it, size_t count, DistHeap* heap, const ScoreFn& score,
        size_t vector_size_bytes, const BitsetFilter* bitset = nullptr) {
    for (; !it.eof(); it.next()) {
        const uint64_t id = it.id();
        if (!bitset_allows_id(bitset, id)) {
            continue;
        }
#ifndef DUMMY_CALC
        prefetch_next_vector_record(it, vector_size_bytes);
        push_result(heap, count, id, score(it));
#else
        (void)score;
        const volatile uint8_t* vec_data = static_cast<const volatile uint8_t*>(it.data());
        uint64_t byte_sum = 0;
        for (size_t i = 0; i < vector_size_bytes; i += 4096) {
            byte_sum ^= vec_data[i];
        }
        push_result(heap, count, id, static_cast<double>(byte_sum));
#endif
    }
}

template <typename ScoreFn>
inline void scan_data_reader_scored(const DataReader& reader, size_t count, DistHeap* heap,
        const ScoreFn& score, const BitsetFilter* bitset = nullptr) {
    const size_t vector_size_bytes = reader.dim() * data_type_size(reader.type());
    scan_iterator_scored(reader.base_begin(), count, heap, score, vector_size_bytes, bitset);
    scan_iterator_scored(reader.delta_begin(), count, heap, score, vector_size_bytes, bitset);
}

template <typename ReaderScanFn>
inline Ret scan_dataset_heap_custom(const DatasetReader& dataset, size_t count, DistHeap* heap,
        const ReaderScanFn& scan_reader, DistFunc func, const BitsetFilter* bitset = nullptr) {
    auto drs = dataset.reader();
    std::vector<DataReaderPtr> readers;
    while (true) {
        auto [reader, ret] = drs->next();
        CHECK(ret);
        if (!reader) {
            break;
        }
        readers.push_back(std::move(reader));
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

template <typename ScoreFn>
inline Ret build_dataset_heap_with_score(const DatasetReader& dataset, size_t count,
        const ScoreFn& score, DistFunc func, DistHeap* heap, const BitsetFilter* bitset = nullptr) {
    return scan_dataset_heap_custom(
        dataset, count, heap,
        [score](const DataReader& reader, size_t local_count, DistHeap* local_heap,
                const BitsetFilter* bitset_filter) {
            scan_data_reader_scored(reader, local_count, local_heap, score, bitset_filter);
        },
        func,
        bitset);
}

template <typename StoredNormScoreFn, typename FallbackScoreFn>
inline Ret build_dataset_heap_with_optional_stored_norms(const DatasetReader& dataset, size_t count,
        const StoredNormScoreFn& stored_norm_score, const FallbackScoreFn& fallback_score,
        DistFunc func, DistHeap* heap, const BitsetFilter* bitset = nullptr) {
    const bool supports_optional_stored_norms = (func == DistFunc::COS || func == DistFunc::L2);
    assert(supports_optional_stored_norms);
    if (!supports_optional_stored_norms) {
        throw std::invalid_argument(
            std::string("stored-norm scoring is only supported for COS/L2, got ") +
            scanner_dist_func_name(func));
    }

    return scan_dataset_heap_custom(
        dataset, count, heap,
        [stored_norm_score, fallback_score, func](const DataReader& reader, size_t local_count,
                DistHeap* local_heap, const BitsetFilter* bitset_filter) {
            if (reader.has_matching_stored_norms(func)) {
                scan_data_reader_scored(reader, local_count, local_heap, stored_norm_score, bitset_filter);
            } else {
                scan_data_reader_scored(reader, local_count, local_heap, fallback_score, bitset_filter);
            }
        },
        func,
        bitset);
}

} // namespace sketch2
