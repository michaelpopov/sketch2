// Implements top-k scanning over data readers and datasets using the calc
// engine layer for metric score computation.

#include "core/calc/scanner_ex.h"
#include "core/calc/calc_engine.h"
#include "core/calc/cosine_distance.h"
#include "core/storage/data_reader.h"
#include "core/storage/dataset_reader.h"
#include "core/utils/log.h"
#include "core/utils/singleton.h"
#include "core/utils/thread_pool.h"
#include "core/utils/timer.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <future>
#include <memory>
#include <queue>
#include <stdexcept>

namespace sketch2 {

namespace {

class DistHeap : public std::priority_queue<DistItem, std::vector<DistItem>, DistItemCompare> {
public:
    using std::priority_queue<DistItem, std::vector<DistItem>, DistItemCompare>::priority_queue;

    const DistItemCompare& comparator() const {
        return this->comp;
    }
};

const char* dist_func_name(DistFunc func) {
    switch (func) {
        case DistFunc::DOT: return "DOT";
        case DistFunc::L2: return "L2";
        case DistFunc::COS: return "COS";
        default: return "unknown";
    }
}

void log_query(const std::string& source, DistFunc func, DataType type, size_t dim,
        size_t count, CalcEngine engine, int64_t elapsed_ms) {
    LOG_TRACE << "ScannerEx query: source=" << source
             << " engine=" << calc_engine_name(engine)
             << " metric=" << dist_func_name(func)
             << " type=" << data_type_to_string(type)
             << " dim=" << dim
             << " k=" << count
             << " time=" << elapsed_ms << " ms";
}

void push_result(DistHeap* heap, size_t count, uint64_t id, double score) {
    const DistItem item{id, score};
    if (heap->size() < count) {
        heap->push(item);
    } else if (heap->comparator()(item, heap->top())) {
        heap->pop();
        heap->push(item);
    }
}

void extract_items(DistHeap* heap, std::vector<DistItem>* result) {
    result->resize(heap->size());
    for (size_t i = heap->size(); i-- > 0;) {
        (*result)[i] = heap->top();
        heap->pop();
    }
}

void extract_ids_from_items(const std::vector<DistItem>& items, std::vector<uint64_t>* result) {
    result->clear();
    result->reserve(items.size());
    for (const auto& item : items) {
        result->push_back(item.id);
    }
}

inline double query_inverse_norm(double query_norm_sq) {
    if (query_norm_sq == 0.0) return 0.0;
    return 1.0 / std::sqrt(query_norm_sq);
}

// ---------------------------------------------------------------------------
// Scorer functors using function pointers
// ---------------------------------------------------------------------------

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
            dot, static_cast<double>(it.cosine_inv_norm()), query_inv_norm);
    }
};

// ---------------------------------------------------------------------------
// Scanning infrastructure (mirrors scanner.cpp)
// ---------------------------------------------------------------------------

template <typename Iterator, typename ScoreFn>
void scan_iterator_scored(Iterator it, size_t count, DistHeap* heap, const ScoreFn& score,
        size_t vector_size_bytes,
        const BitsetFilter* bitset = nullptr) {
    for (; !it.eof(); it.next()) {
        if (bitset != nullptr) {
            assert(bitset->data != nullptr);
            const uint64_t id = it.id();
            const uint64_t byte_index = id >> 3;
            if (byte_index >= bitset->size) continue;
            const uint8_t mask = static_cast<uint8_t>(1u << (id & 7u));
            if ((bitset->data[byte_index] & mask) == 0u) continue;
        }
#ifndef DUMMY_CALC
        (void)vector_size_bytes;
        push_result(heap, count, it.id(), score(it));
#else
        (void)score;
        // Access vector's data and use it as a dummy to prevent
        // optimizer removing this code. It is required for measuring
        // I/O performance.
        const uint8_t* vec_data = it.data();
        uint64_t byte_sum = 0;
        for (size_t i = 0; i < vector_size_bytes; ++i) {
            byte_sum += vec_data[i];
        }
        push_result(heap, count, it.id(), static_cast<double>(byte_sum));
#endif
    }
}

template <typename ScoreFn>
void scan_data_reader_scored(const DataReader& reader,
        size_t count, DistHeap* heap, const ScoreFn& score,
        const BitsetFilter* bitset = nullptr) {
    const size_t vector_size_bytes = reader.dim() * data_type_size(reader.type());
    scan_iterator_scored(reader.base_begin(), count, heap, score, vector_size_bytes, bitset);
    scan_iterator_scored(reader.delta_begin(), count, heap, score, vector_size_bytes, bitset);
}

template <typename ReaderScanFn>
Ret scan_dataset_heap_custom(const DatasetReader& dataset, size_t count, DistHeap* heap,
        const ReaderScanFn& scan_reader, DistFunc func, const BitsetFilter* bitset = nullptr) {
    auto drs = dataset.reader();
    std::vector<DataReaderPtr> readers;
    while (true) {
        auto [reader, ret] = drs->next();
        CHECK(ret);
        if (!reader) break;
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
            (*scan_reader_shared)(*reader, count, &local_heap, bitset);
            return local_heap;
        }));
    }

    std::exception_ptr first_error;
    for (auto& fut : futures) {
        try {
            DistHeap local_heap = fut.get();
            if (first_error != nullptr) {
                continue;
            }
            while (!local_heap.empty()) {
                push_result(heap, count, local_heap.top().id, local_heap.top().score);
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

    return Ret(0);
}

template <typename ScoreFn>
Ret build_dataset_heap_with_score(const DatasetReader& dataset, size_t count, const ScoreFn& score,
        DistFunc func, DistHeap* heap, const BitsetFilter* bitset = nullptr) {
    return scan_dataset_heap_custom(
        dataset, count, heap,
        [score](const DataReader& reader, size_t local_count, DistHeap* local_heap,
                const BitsetFilter* bitset) {
            scan_data_reader_scored(reader, local_count, local_heap, score, bitset);
        },
        func,
        bitset);
}

template <typename InvScoreFn, typename QueryScoreFn>
Ret build_dataset_heap_with_cos_scores(const DatasetReader& dataset, size_t count,
        const InvScoreFn& inv_score, const QueryScoreFn& query_score, DistFunc func, DistHeap* heap,
        const BitsetFilter* bitset = nullptr) {
    return scan_dataset_heap_custom(
        dataset, count, heap,
        [inv_score, query_score](const DataReader& reader, size_t local_count, DistHeap* local_heap,
                const BitsetFilter* bitset) {
            if (reader.has_cosine_inv_norms()) {
                scan_data_reader_scored(reader, local_count, local_heap, inv_score, bitset);
            } else {
                scan_data_reader_scored(reader, local_count, local_heap, query_score, bitset);
            }
        },
        func,
        bitset);
}

// ---------------------------------------------------------------------------
// Build the top-k heap using resolved CalcKernels
// ---------------------------------------------------------------------------

Ret build_heap(CalcEngine engine, const DatasetReader& dataset, DistFunc func,
        size_t count, const uint8_t* vec, DistHeap* heap, const BitsetFilter* bitset) {
    const DataType type = dataset.type();
    const size_t dim = dataset.dim();
    const CalcKernels k = resolve_calc_kernels(engine, func, type);

    if (func == DistFunc::COS) {
        assert(k.squared_norm && k.dot && k.dist_with_query_norm);
        const double query_norm_sq = k.squared_norm(vec, dim);
        const double query_inv = query_inverse_norm(query_norm_sq);
        const FnPtrInvNormScore inv_score{k.dot, vec, dim, query_inv};
        const FnPtrQueryNormScore query_score{k.dist_with_query_norm, vec, dim, query_norm_sq};
        return build_dataset_heap_with_cos_scores(dataset, count, inv_score, query_score, func, heap, bitset);
    }

    assert(k.dist);
    const FnPtrDistScore score{k.dist, vec, dim};
    return build_dataset_heap_with_score(dataset, count, score, func, heap, bitset);
}

} // namespace

// ---------------------------------------------------------------------------
// ScannerEx public API
// ---------------------------------------------------------------------------

const char* calc_engine_name(CalcEngine engine) {
    switch (engine) {
        case CalcEngine::compute: return "compute";
        case CalcEngine::highway: return "highway";
        case CalcEngine::numkong: return "numkong";
        default: return "unknown";
    }
}

ScannerEx::ScannerEx(CalcEngine engine) : engine_(engine) {}

Ret ScannerEx::find(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) const {
    try {
        std::vector<DistItem> items;
        CHECK(find_items_(dataset, count, vec, items, nullptr));
        extract_ids_from_items(items, &result);
        return Ret(0);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret ScannerEx::find_items(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result, const BitsetFilter* bitset) const {
    try {
        return find_items_(dataset, count, vec, result, bitset);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret ScannerEx::find_items_(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result, const BitsetFilter* bitset) const {
    if (vec == nullptr || count == 0) {
        return Ret("ScannerEx::find: invalid arguments.");
    }
    result.clear();
    const DistFunc func = dataset.dist_func();
    DistHeap heap(DistItemCompare{func});
    Timer timer("scanner_ex::query");
    CHECK(build_heap(engine_, dataset, func, count, vec, &heap, bitset));
    log_query(dataset.name(), func, dataset.type(), dataset.dim(), count, engine_, timer.elapsed_ms());
    extract_items(&heap, &result);
    return Ret(0);
}

} // namespace sketch2
