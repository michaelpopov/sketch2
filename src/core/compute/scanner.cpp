// Implements top-k scanning over data readers and datasets using the distance kernels.

#include "scanner.h"
#include "core/compute/compute_cos.h"
#include "core/compute/compute_l1.h"
#include "core/compute/compute_l2.h"
#include "core/storage/data_reader.h"
#include "core/storage/dataset_reader.h"
#include "core/utils/log.h"
#include "core/utils/thread_pool.h"
#include "core/utils/timer.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <future>
#include <memory>
#include <queue>
#include <stdexcept>

namespace sketch2 {

namespace {

using DistHeap = std::priority_queue<DistItem, std::vector<DistItem>, DistItem::Compare>;

const char* dist_func_name(DistFunc func) {
    switch (func) {
        case DistFunc::L1: return "L1";
        case DistFunc::L2: return "L2";
        case DistFunc::COS: return "COS";
        default: return "unknown";
    }
}

void log_query(const std::string& source, DistFunc func, DataType type, size_t dim,
        size_t count, int64_t elapsed_ms) {
    LOG_INFO << "Scanner query: runtime_backend='"
             << get_singleton().compute_unit().name()
             << "' source=" << source
             << " metric=" << dist_func_name(func)
             << " type=" << data_type_to_string(type)
             << " dim=" << dim
             << " k=" << count
             << " time=" << elapsed_ms << " ms";
}

void push_result(DistHeap* heap, size_t count, uint64_t id, double dist) {
    const DistItem item{id, dist};
    DistItem::Compare is_better;
    if (heap->size() < count) {
        heap->push(item);
    } else if (is_better(item, heap->top())) {
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

// Zero-norm queries use an inverse norm of zero rather than infinity so the
// cosine finalizer can preserve the public zero-vector distance contract.
inline double query_inverse_norm(double query_norm_sq) {
    if (query_norm_sq == 0.0) {
        return 0.0;
    }
    return 1.0 / std::sqrt(query_norm_sq);
}

// Resolve the type-specific norm helper once per query so the per-row cosine
// scan can stay on a fully specialized backend/type path.
template <typename ComputeTarget>
double query_squared_norm(DataType type, const uint8_t* vec, size_t dim) {
    switch (type) {
        case DataType::f32:
            return ComputeTarget::squared_norm_f32(vec, dim);
        case DataType::f16:
            return ComputeTarget::squared_norm_f16(vec, dim);
        case DataType::i16:
            return ComputeTarget::squared_norm_i16(vec, dim);
        default:
            assert(false);
            throw std::runtime_error("Scanner::find: unsupported data type.");
    }
}

template <auto DistanceFn>
struct DistanceScore {
    const uint8_t* vec = nullptr;
    size_t dim = 0;

    template <typename Iterator>
    double operator()(const Iterator& it) const {
        return DistanceFn(it.data(), vec, dim);
    }
};

template <auto DistanceFn>
struct QueryNormScore {
    const uint8_t* vec = nullptr;
    size_t dim = 0;
    double query_norm_sq = 0.0;

    template <typename Iterator>
    double operator()(const Iterator& it) const {
        return DistanceFn(it.data(), vec, dim, query_norm_sq);
    }
};

template <auto DotFn>
struct InvNormScore {
    const uint8_t* vec = nullptr;
    size_t dim = 0;
    double query_inv_norm = 0.0;

    template <typename Iterator>
    double operator()(const Iterator& it) const {
        assert(query_inv_norm >= 0.0);
        const double dot = DotFn(it.data(), vec, dim);
        return finalize_cosine_distance_from_inverse_norms(
            dot, static_cast<double>(it.cosine_inv_norm()), query_inv_norm);
    }
};

template <typename Iterator, typename ScoreFn>
void scan_iterator_scored(Iterator it, size_t count, DistHeap* heap, const ScoreFn& score,
        const BitsetFilter* bitset = nullptr) {
    for (; !it.eof(); it.next()) {
        if (bitset != nullptr) {
            assert(bitset->data != nullptr);
            const uint64_t id = it.id();
            const uint64_t byte_index = id >> 3;
            if (byte_index >= bitset->size) {
                continue;
            }
            const uint8_t mask = static_cast<uint8_t>(1u << (id & 7u));
            if ((bitset->data[byte_index] & mask) == 0u) {
                continue;
            }
        }
        push_result(heap, count, it.id(), score(it));
    }
}

// Scans both the base and delta iterators of a single DataReader file pair.
template <typename ScoreFn>
void scan_data_reader_scored(const DataReader& reader,
        size_t count, DistHeap* heap, const ScoreFn& score, const BitsetFilter* bitset = nullptr) {
    scan_iterator_scored(reader.base_begin(), count, heap, score, bitset);
    scan_iterator_scored(reader.delta_begin(), count, heap, score, bitset);
}

// Builds a top-k heap across all persisted dataset files. Each file is scanned
// independently; the accumulator is not consulted (it is a write-side concern).
//
// When the singleton thread pool is available and there are at least two
// readers, each reader is scanned on a worker thread using a private heap.
// Private heaps are merged into the shared heap once all futures complete.
template <typename ReaderScanFn>
Ret scan_dataset_heap_custom(const DatasetReader& dataset, size_t count, DistHeap* heap,
        const ReaderScanFn& scan_reader, const BitsetFilter* bitset = nullptr) {
    // Collect all readers up front so the iterator is not shared across threads.
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

    // Sequential fallback: pool unavailable or too few readers to parallelize.
    if (!pool || readers.size() < 2) {
        for (const auto& reader : readers) {
            scan_reader(*reader, count, heap, bitset);
        }
        return Ret(0);
    }

    // Submit one task per reader. Each task builds a private heap so no
    // synchronization is needed during scanning.
    //
    // Keep scan_reader on shared ownership so worker tasks do not reference
    // stack state if submit/get throws and this function exits before every
    // queued task has finished.
    const auto scan_reader_shared = std::make_shared<ReaderScanFn>(scan_reader);
    std::vector<std::future<DistHeap>> futures;
    futures.reserve(readers.size());
    for (const auto& reader : readers) {
        futures.push_back(pool->submit([scan_reader_shared, count, reader, bitset]() {
            DistHeap local_heap;
            (*scan_reader_shared)(*reader, count, &local_heap, bitset);
            return local_heap;
        }));
    }

    // Collect per-reader heaps and merge into the shared heap.
    for (auto& fut : futures) {
        DistHeap local_heap = fut.get();
        while (!local_heap.empty()) {
            push_result(heap, count, local_heap.top().id, local_heap.top().dist);
            local_heap.pop();
        }
    }

    return Ret(0);
}

// Dataset score adapters bind a fixed scorer into the shared dataset heap builder.
template <typename ScoreFn>
Ret build_dataset_heap_with_score(const DatasetReader& dataset, size_t count, const ScoreFn& score,
        DistHeap* heap, const BitsetFilter* bitset = nullptr) {
    return scan_dataset_heap_custom(
        dataset, count, heap,
        [&](const DataReader& reader, size_t local_count, DistHeap* local_heap, const BitsetFilter* bitset) {
            scan_data_reader_scored(reader, local_count, local_heap, score, bitset);
        },
        bitset);
}

// Cosine dataset scans choose per reader whether to use stored inverse norms or
// the full query-norm path, but otherwise reuse the same heap-building logic.
template <typename InvScoreFn, typename QueryScoreFn>
Ret build_dataset_heap_with_cos_scores(const DatasetReader& dataset, size_t count,
        const InvScoreFn& inv_score, const QueryScoreFn& query_score, DistHeap* heap,
        const BitsetFilter* bitset = nullptr) {
    return scan_dataset_heap_custom(
        dataset, count, heap,
        [&](const DataReader& reader, size_t local_count, DistHeap* local_heap, const BitsetFilter* bitset) {
            if (reader.has_cosine_inv_norms()) {
                scan_data_reader_scored(reader, local_count, local_heap, inv_score, bitset);
            } else {
                scan_data_reader_scored(reader, local_count, local_heap, query_score, bitset);
            }
        },
        bitset);
}

// Reader adapters scan a single DataReader without the dataset layer.
template <typename ScoreFn>
Ret build_reader_heap_with_score(const DataReader& reader, size_t count, const ScoreFn& score,
        DistHeap* heap) {
    scan_data_reader_scored(reader, count, heap, score);
    return Ret(0);
}

template <typename InvScoreFn, typename QueryScoreFn>
Ret build_reader_heap_with_cos_scores(const DataReader& reader, size_t count,
        const InvScoreFn& inv_score, const QueryScoreFn& query_score, DistHeap* heap) {
    if (reader.has_cosine_inv_norms()) {
        scan_data_reader_scored(reader, count, heap, inv_score);
    } else {
        scan_data_reader_scored(reader, count, heap, query_score);
    }
    return Ret(0);
}

// Type dispatch happens once per query. That produces a scorer with a fixed
// element width so the inner scan loop never branches on DataType.
template <typename ComputeTarget>
Ret dispatch_dataset(DataType type, const DatasetReader& dataset, size_t count, const uint8_t* vec,
        DistHeap* heap, const BitsetFilter* bitset = nullptr) {
    const size_t dim = dataset.dim();
    switch (type) {
        case DataType::f32:
            return build_dataset_heap_with_score(
                dataset, count, DistanceScore<&ComputeTarget::dist_f32>{vec, dim}, heap, bitset);
        case DataType::f16:
            return build_dataset_heap_with_score(
                dataset, count, DistanceScore<&ComputeTarget::dist_f16>{vec, dim}, heap, bitset);
        case DataType::i16:
            return build_dataset_heap_with_score(
                dataset, count, DistanceScore<&ComputeTarget::dist_i16>{vec, dim}, heap, bitset);
        default: return Ret("Scanner::find: unsupported data type.");
    }
}

// Dispatches cosine KNN over a dataset. It computes the query norm once and
// then uses either precomputed inverse norms or a full query-norm distance path
// depending on what each backing reader stores.
template <typename ComputeTarget>
Ret dispatch_dataset_cos(DataType type, const DatasetReader& dataset, size_t count, const uint8_t* vec,
        DistHeap* heap, const BitsetFilter* bitset = nullptr) {
    const size_t dim = dataset.dim();
    const double query_norm_sq = query_squared_norm<ComputeTarget>(type, vec, dim);
    const double query_inv = query_inverse_norm(query_norm_sq);
    switch (type) {
        case DataType::f32: {
            const InvNormScore<&ComputeTarget::dot_f32> inv_score{vec, dim, query_inv};
            const QueryNormScore<&ComputeTarget::dist_f32_with_query_norm> query_score{vec, dim, query_norm_sq};
            return build_dataset_heap_with_cos_scores(dataset, count, inv_score, query_score, heap, bitset);
        }
        case DataType::f16: {
            const InvNormScore<&ComputeTarget::dot_f16> inv_score{vec, dim, query_inv};
            const QueryNormScore<&ComputeTarget::dist_f16_with_query_norm> query_score{vec, dim, query_norm_sq};
            return build_dataset_heap_with_cos_scores(dataset, count, inv_score, query_score, heap, bitset);
        }
        case DataType::i16: {
            const InvNormScore<&ComputeTarget::dot_i16> inv_score{vec, dim, query_inv};
            const QueryNormScore<&ComputeTarget::dist_i16_with_query_norm> query_score{vec, dim, query_norm_sq};
            return build_dataset_heap_with_cos_scores(dataset, count, inv_score, query_score, heap, bitset);
        }
        default: return Ret("Scanner::find: unsupported data type.");
    }
}

template <typename ComputeTarget>
Ret dispatch_reader(DataType type, const DataReader& reader, size_t count, const uint8_t* vec,
        DistHeap* heap) {
    const size_t dim = reader.dim();
    switch (type) {
        case DataType::f32:
            return build_reader_heap_with_score(
                reader, count, DistanceScore<&ComputeTarget::dist_f32>{vec, dim}, heap);
        case DataType::f16:
            return build_reader_heap_with_score(
                reader, count, DistanceScore<&ComputeTarget::dist_f16>{vec, dim}, heap);
        case DataType::i16:
            return build_reader_heap_with_score(
                reader, count, DistanceScore<&ComputeTarget::dist_i16>{vec, dim}, heap);
        default: return Ret("Scanner::find: unsupported data type.");
    }
}

// Reader cosine KNN variant that computes query normalization once and then
// dispatches to the type-specific scorer.
template <typename ComputeTarget>
Ret dispatch_reader_cos(DataType type, const DataReader& reader, size_t count, const uint8_t* vec,
        DistHeap* heap) {
    const size_t dim = reader.dim();
    const double query_norm_sq = query_squared_norm<ComputeTarget>(type, vec, dim);
    const double query_inv = query_inverse_norm(query_norm_sq);
    switch (type) {
        case DataType::f32: {
            const InvNormScore<&ComputeTarget::dot_f32> inv_score{vec, dim, query_inv};
            const QueryNormScore<&ComputeTarget::dist_f32_with_query_norm> query_score{vec, dim, query_norm_sq};
            return build_reader_heap_with_cos_scores(reader, count, inv_score, query_score, heap);
        }
        case DataType::f16: {
            const InvNormScore<&ComputeTarget::dot_f16> inv_score{vec, dim, query_inv};
            const QueryNormScore<&ComputeTarget::dist_f16_with_query_norm> query_score{vec, dim, query_norm_sq};
            return build_reader_heap_with_cos_scores(reader, count, inv_score, query_score, heap);
        }
        case DataType::i16: {
            const InvNormScore<&ComputeTarget::dot_i16> inv_score{vec, dim, query_inv};
            const QueryNormScore<&ComputeTarget::dist_i16_with_query_norm> query_score{vec, dim, query_norm_sq};
            return build_reader_heap_with_cos_scores(reader, count, inv_score, query_score, heap);
        }
        default: return Ret("Scanner::find: unsupported data type.");
    }
}

// Backend dispatch stays at this outer layer so scanner hot loops execute
// entirely within one concrete L1/L2/cosine implementation family.
// Overloads on Dataset vs DataReader select the right metric dispatch path
// (dataset path handles accumulator shadowing; reader path scans files directly).
template <typename L1Target, typename L2Target, typename CosTarget>
Ret dispatch_with_backend(DataType type, DistFunc func, const DatasetReader& dataset, size_t count,
        const uint8_t* vec, DistHeap* heap, const BitsetFilter* bitset = nullptr) {
    switch (func) {
        case DistFunc::L1:
            return dispatch_dataset<L1Target>(type, dataset, count, vec, heap, bitset);
        case DistFunc::L2:
            return dispatch_dataset<L2Target>(type, dataset, count, vec, heap, bitset);
        case DistFunc::COS:
            return dispatch_dataset_cos<CosTarget>(type, dataset, count, vec, heap, bitset);
        default:
            return Ret("Scanner::find: unsupported distance function.");
    }
}

template <typename L1Target, typename L2Target, typename CosTarget>
Ret dispatch_with_backend(DataType type, DistFunc func, const DataReader& reader, size_t count,
        const uint8_t* vec, DistHeap* heap, const BitsetFilter* bitset = nullptr) {
    (void)bitset;
    switch (func) {
        case DistFunc::L1:
            return dispatch_reader<L1Target>(type, reader, count, vec, heap);
        case DistFunc::L2:
            return dispatch_reader<L2Target>(type, reader, count, vec, heap);
        case DistFunc::COS:
            return dispatch_reader_cos<CosTarget>(type, reader, count, vec, heap);
        default:
            return Ret("Scanner::find: not implemented");
    }
}

// Build the top-k heap by selecting the right SIMD backend at runtime.
// Works for both Dataset and DataReader via the dispatch_with_backend overloads.
// ID-returning and item-returning APIs both share this path.
template <typename Source>
Ret build_heap(const Source& source, DistFunc func, size_t count,
        const uint8_t* vec, DistHeap* heap, const BitsetFilter* bitset = nullptr) {
    const DataType type = source.type();
    switch (get_singleton().compute_unit().kind()) {
#if defined(SKETCH_ENABLE_AVX512VNNI) && SKETCH_ENABLE_AVX512VNNI && (defined(__x86_64__) || defined(__i386__))
        case ComputeBackendKind::avx512_vnni:
            return dispatch_with_backend<ComputeL1_AVX512_VNNI, ComputeL2_AVX512_VNNI, ComputeCos_AVX512_VNNI>(
                type, func, source, count, vec, heap, bitset);
#endif
#if defined(SKETCH_ENABLE_AVX512F) && SKETCH_ENABLE_AVX512F && (defined(__x86_64__) || defined(__i386__))
        case ComputeBackendKind::avx512f:
            return dispatch_with_backend<ComputeL1_AVX512, ComputeL2_AVX512, ComputeCos_AVX512>(
                type, func, source, count, vec, heap, bitset);
#endif
#if defined(SKETCH_ENABLE_AVX2) && SKETCH_ENABLE_AVX2 && (defined(__x86_64__) || defined(__i386__))
        case ComputeBackendKind::avx2:
            return dispatch_with_backend<ComputeL1_AVX2, ComputeL2_AVX2, ComputeCos_AVX2>(
                type, func, source, count, vec, heap, bitset);
#endif
#if defined(__aarch64__)
        case ComputeBackendKind::neon:
            return dispatch_with_backend<ComputeL1_Neon, ComputeL2_Neon, ComputeCos_Neon>(
                type, func, source, count, vec, heap, bitset);
#endif
        case ComputeBackendKind::scalar:
        default:
            return dispatch_with_backend<ComputeL1, ComputeL2, ComputeCos>(
                type, func, source, count, vec, heap, bitset);
    }
}

} // namespace

Ret Scanner::find(const DatasetReader& dataset, size_t count, const uint8_t* vec,
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

Ret Scanner::find_items(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result, const BitsetFilter* bitset) const {
    try {
        return find_items_(dataset, count, vec, result, bitset);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret Scanner::find_items_(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result, const BitsetFilter* bitset) const {
    if (vec == nullptr || count == 0) {
        return Ret("Scanner::find: invalid arguments.");
    }
    result.clear();
    const DistFunc func = dataset.dist_func();
    DistHeap heap;
    Timer timer("scanner::query");
    CHECK(build_heap(dataset, func, count, vec, &heap, bitset));
    log_query(dataset.name(), func, dataset.type(), dataset.dim(), count, timer.elapsed_ms());
    extract_items(&heap, &result);
    return Ret(0);
}

} // namespace sketch2
