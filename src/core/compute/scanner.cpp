// Implements top-k scanning over data readers and datasets using the distance kernels.

#include "scanner.h"
#include "core/compute/compute_cos.h"
#include "core/compute/compute_l1.h"
#include "core/compute/compute_l2.h"
#include "core/storage/data_reader.h"
#include "core/storage/dataset.h"
#include "core/utils/log.h"
#include <algorithm>
#include <cassert>
#include <cmath>
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

const char* query_result_name(bool items) {
    return items ? "items" : "ids";
}

void log_query_dispatch(const char* source, DistFunc func, DataType type, size_t dim,
        size_t count, bool items) {
    LOG_INFO << "Scanner query dispatch: backend='"
             << get_singleton().compute_unit().name()
             << "' source=" << source
             << " metric=" << dist_func_name(func)
             << " type=" << data_type_to_string(type)
             << " dim=" << dim
             << " k=" << count
             << " result=" << query_result_name(items);
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

void extract_ids(DistHeap* heap, std::vector<uint64_t>* result) {
    result->resize(heap->size());
    for (size_t i = heap->size(); i-- > 0;) {
        (*result)[i] = heap->top().id;
        heap->pop();
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
void scan_iterator_scored(Iterator it, size_t count, DistHeap* heap, const ScoreFn& score) {
    for (; !it.eof(); it.next()) {
        push_result(heap, count, it.id(), score(it));
    }
}

// Scans one ordered reader while skipping ids shadowed by newer state. The
// skip list is walked in lockstep with the iterator so each id is compared once.
template <typename Iterator, typename ScoreFn>
void scan_ordered_reader_scored(Iterator it, const std::vector<uint64_t>& skip_ids,
        size_t count, DistHeap* heap, const ScoreFn& score) {
    if (skip_ids.empty()) {
        scan_iterator_scored(it, count, heap, score);
        return;
    }

    auto skip_it = skip_ids.end();
    for (; !it.eof(); it.next()) {
        const uint64_t id = it.id();
        if (skip_it == skip_ids.end()) {
            skip_it = std::lower_bound(skip_ids.begin(), skip_ids.end(), id);
        }
        while (skip_it != skip_ids.end() && *skip_it < id) {
            ++skip_it;
        }
        if (skip_it != skip_ids.end() && *skip_it == id) {
            continue;
        }
        push_result(heap, count, id, score(it));
    }
}

template <typename ScoreFn>
void scan_data_reader_scored(const DataReader& reader, const std::vector<uint64_t>& skip_ids,
        size_t count, DistHeap* heap, const ScoreFn& score) {
    scan_ordered_reader_scored(reader.base_begin(), skip_ids, count, heap, score);
    scan_ordered_reader_scored(reader.delta_begin(), skip_ids, count, heap, score);
}

// Builds a top-k heap across the persisted dataset and the pending accumulator.
// Persisted readers skip ids modified in the accumulator so the heap only sees
// each logical id once, with the freshest version winning.
template <typename ReaderScanFn, typename AccumScoreFn>
Ret scan_dataset_heap_custom(const Dataset& dataset, size_t count, DistHeap* heap,
        const ReaderScanFn& scan_reader, const AccumScoreFn& accum_score,
        [[maybe_unused]] bool require_accumulator_cosine_inv_norms = false) {
    CHECK(dataset.prepare_read_state());
    assert(!require_accumulator_cosine_inv_norms ||
        !dataset.has_accumulator() || dataset.accumulator_has_cosine_inv_norms());
    const std::vector<uint64_t> modified_ids = dataset.accumulator_modified_ids();

    auto drs = dataset.reader();
    while (true) {
        auto [reader, ret] = drs->next();
        CHECK(ret);
        if (!reader) {
            break;
        }
        scan_reader(*reader, modified_ids, count, heap);
    }

    scan_iterator_scored(dataset.accumulator_begin(), count, heap, accum_score);
    return Ret(0);
}

template <typename ReaderScanFn>
Ret scan_reader_heap_custom(const DataReader& reader, size_t count, DistHeap* heap,
        const ReaderScanFn& scan_reader) {
    scan_reader(reader, {}, count, heap);
    return Ret(0);
}

// These helpers keep the "build heap then materialize sorted output" pattern in
// one place so dataset and reader scan variants do not duplicate heap plumbing.
template <typename HeapBuildFn>
Ret collect_items(const HeapBuildFn& build, std::vector<DistItem>& result) {
    DistHeap heap;
    CHECK(build(&heap));
    extract_items(&heap, &result);
    return Ret(0);
}

template <typename HeapBuildFn>
Ret collect_ids(const HeapBuildFn& build, std::vector<uint64_t>& result) {
    DistHeap heap;
    CHECK(build(&heap));
    extract_ids(&heap, &result);
    return Ret(0);
}

// Dataset score adapters bind a fixed scorer into the shared dataset heap
// builder, keeping accumulator shadowing and top-k behavior identical across
// metrics and output shapes.
template <typename ScoreFn>
Ret scan_dataset_items_with_score(const Dataset& dataset, size_t count, const ScoreFn& score,
        std::vector<DistItem>& result) {
    return collect_items([&](DistHeap* heap) {
        return scan_dataset_heap_custom(
            dataset,
            count,
            heap,
            [&](const DataReader& reader, const std::vector<uint64_t>& skip_ids, size_t local_count, DistHeap* local_heap) {
                scan_data_reader_scored(reader, skip_ids, local_count, local_heap, score);
            },
            score);
    }, result);
}

template <typename ScoreFn>
Ret scan_dataset_ids_with_score(const Dataset& dataset, size_t count, const ScoreFn& score,
        std::vector<uint64_t>& result) {
    return collect_ids([&](DistHeap* heap) {
        return scan_dataset_heap_custom(
            dataset,
            count,
            heap,
            [&](const DataReader& reader, const std::vector<uint64_t>& skip_ids, size_t local_count, DistHeap* local_heap) {
                scan_data_reader_scored(reader, skip_ids, local_count, local_heap, score);
            },
            score);
    }, result);
}

// Cosine dataset scans choose per reader whether to use stored inverse norms or
// the full query-norm path, but otherwise reuse the same heap-building logic.
template <typename InvScoreFn, typename QueryScoreFn>
Ret scan_dataset_items_with_cos_scores(const Dataset& dataset, size_t count,
        const InvScoreFn& inv_score, const QueryScoreFn& query_score,
        std::vector<DistItem>& result) {
    return collect_items([&](DistHeap* heap) {
        return scan_dataset_heap_custom(
            dataset,
            count,
            heap,
            [&](const DataReader& reader, const std::vector<uint64_t>& skip_ids, size_t local_count, DistHeap* local_heap) {
                if (reader.has_cosine_inv_norms()) {
                    scan_data_reader_scored(reader, skip_ids, local_count, local_heap, inv_score);
                } else {
                    scan_data_reader_scored(reader, skip_ids, local_count, local_heap, query_score);
                }
            },
            inv_score,
            true);
    }, result);
}

template <typename InvScoreFn, typename QueryScoreFn>
Ret scan_dataset_ids_with_cos_scores(const Dataset& dataset, size_t count,
        const InvScoreFn& inv_score, const QueryScoreFn& query_score,
        std::vector<uint64_t>& result) {
    return collect_ids([&](DistHeap* heap) {
        return scan_dataset_heap_custom(
            dataset,
            count,
            heap,
            [&](const DataReader& reader, const std::vector<uint64_t>& skip_ids, size_t local_count, DistHeap* local_heap) {
                if (reader.has_cosine_inv_norms()) {
                    scan_data_reader_scored(reader, skip_ids, local_count, local_heap, inv_score);
                } else {
                    scan_data_reader_scored(reader, skip_ids, local_count, local_heap, query_score);
                }
            },
            inv_score,
            true);
    }, result);
}

// Reader adapters mirror the dataset helpers without the accumulator layer, so
// reader-only scans still share the same scorer and heap semantics.
template <typename ScoreFn>
Ret scan_reader_items_with_score(const DataReader& reader, size_t count, const ScoreFn& score,
        std::vector<DistItem>& result) {
    return collect_items([&](DistHeap* heap) {
        return scan_reader_heap_custom(
            reader,
            count,
            heap,
            [&](const DataReader& local_reader, const std::vector<uint64_t>& skip_ids, size_t local_count, DistHeap* local_heap) {
                scan_data_reader_scored(local_reader, skip_ids, local_count, local_heap, score);
            });
    }, result);
}

template <typename ScoreFn>
Ret scan_reader_ids_with_score(const DataReader& reader, size_t count, const ScoreFn& score,
        std::vector<uint64_t>& result) {
    return collect_ids([&](DistHeap* heap) {
        return scan_reader_heap_custom(
            reader,
            count,
            heap,
            [&](const DataReader& local_reader, const std::vector<uint64_t>& skip_ids, size_t local_count, DistHeap* local_heap) {
                scan_data_reader_scored(local_reader, skip_ids, local_count, local_heap, score);
            });
    }, result);
}

template <typename InvScoreFn, typename QueryScoreFn>
Ret scan_reader_items_with_cos_scores(const DataReader& reader, size_t count,
        const InvScoreFn& inv_score, const QueryScoreFn& query_score,
        std::vector<DistItem>& result) {
    return collect_items([&](DistHeap* heap) {
        return scan_reader_heap_custom(
            reader,
            count,
            heap,
            [&](const DataReader& local_reader, const std::vector<uint64_t>& skip_ids, size_t local_count, DistHeap* local_heap) {
                if (local_reader.has_cosine_inv_norms()) {
                    scan_data_reader_scored(local_reader, skip_ids, local_count, local_heap, inv_score);
                } else {
                    scan_data_reader_scored(local_reader, skip_ids, local_count, local_heap, query_score);
                }
            });
    }, result);
}

template <typename InvScoreFn, typename QueryScoreFn>
Ret scan_reader_ids_with_cos_scores(const DataReader& reader, size_t count,
        const InvScoreFn& inv_score, const QueryScoreFn& query_score,
        std::vector<uint64_t>& result) {
    return collect_ids([&](DistHeap* heap) {
        return scan_reader_heap_custom(
            reader,
            count,
            heap,
            [&](const DataReader& local_reader, const std::vector<uint64_t>& skip_ids, size_t local_count, DistHeap* local_heap) {
                if (local_reader.has_cosine_inv_norms()) {
                    scan_data_reader_scored(local_reader, skip_ids, local_count, local_heap, inv_score);
                } else {
                    scan_data_reader_scored(local_reader, skip_ids, local_count, local_heap, query_score);
                }
            });
    }, result);
}

// Type dispatch happens once per query. That produces a scorer with a fixed
// element width so the inner scan loop never branches on DataType.
template <typename ComputeTarget>
Ret dispatch_reader_ids(DataType type, const DataReader& reader, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) {
    const size_t dim = reader.dim();
    switch (type) {
        case DataType::f32:
            return scan_reader_ids_with_score(
                reader, count, DistanceScore<&ComputeTarget::dist_f32>{vec, dim}, result);
        case DataType::f16:
            return scan_reader_ids_with_score(
                reader, count, DistanceScore<&ComputeTarget::dist_f16>{vec, dim}, result);
        case DataType::i16:
            return scan_reader_ids_with_score(
                reader, count, DistanceScore<&ComputeTarget::dist_i16>{vec, dim}, result);
        default: return Ret("Scanner::find: unsupported data type.");
    }
}

template <typename ComputeTarget>
Ret dispatch_dataset_ids(DataType type, const Dataset& dataset, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) {
    const size_t dim = dataset.dim();
    switch (type) {
        case DataType::f32:
            return scan_dataset_ids_with_score(
                dataset, count, DistanceScore<&ComputeTarget::dist_f32>{vec, dim}, result);
        case DataType::f16:
            return scan_dataset_ids_with_score(
                dataset, count, DistanceScore<&ComputeTarget::dist_f16>{vec, dim}, result);
        case DataType::i16:
            return scan_dataset_ids_with_score(
                dataset, count, DistanceScore<&ComputeTarget::dist_i16>{vec, dim}, result);
        default: return Ret("Scanner::find: unsupported data type.");
    }
}

template <typename ComputeTarget>
Ret dispatch_dataset_items(DataType type, const Dataset& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result) {
    const size_t dim = dataset.dim();
    switch (type) {
        case DataType::f32:
            return scan_dataset_items_with_score(
                dataset, count, DistanceScore<&ComputeTarget::dist_f32>{vec, dim}, result);
        case DataType::f16:
            return scan_dataset_items_with_score(
                dataset, count, DistanceScore<&ComputeTarget::dist_f16>{vec, dim}, result);
        case DataType::i16:
            return scan_dataset_items_with_score(
                dataset, count, DistanceScore<&ComputeTarget::dist_i16>{vec, dim}, result);
        default: return Ret("Scanner::find: unsupported data type.");
    }
}

template <typename ComputeTarget>
Ret dispatch_reader_items(DataType type, const DataReader& reader, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result) {
    const size_t dim = reader.dim();
    switch (type) {
        case DataType::f32:
            return scan_reader_items_with_score(
                reader, count, DistanceScore<&ComputeTarget::dist_f32>{vec, dim}, result);
        case DataType::f16:
            return scan_reader_items_with_score(
                reader, count, DistanceScore<&ComputeTarget::dist_f16>{vec, dim}, result);
        case DataType::i16:
            return scan_reader_items_with_score(
                reader, count, DistanceScore<&ComputeTarget::dist_i16>{vec, dim}, result);
        default: return Ret("Scanner::find: unsupported data type.");
    }
}

// Dispatches cosine KNN over a dataset. It computes the query norm once and
// then uses either precomputed inverse norms or a full query-norm distance path
// depending on what each backing reader stores.
template <typename ComputeTarget>
Ret dispatch_dataset_cos_ids(DataType type, const Dataset& dataset, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) {
    const size_t dim = dataset.dim();
    const double query_norm_sq = query_squared_norm<ComputeTarget>(type, vec, dim);
    const double query_inv = query_inverse_norm(query_norm_sq);
    switch (type) {
        case DataType::f32: {
            const InvNormScore<&ComputeTarget::dot_f32> inv_score{vec, dim, query_inv};
            const QueryNormScore<&ComputeTarget::dist_f32_with_query_norm> query_score{vec, dim, query_norm_sq};
            return scan_dataset_ids_with_cos_scores(dataset, count, inv_score, query_score, result);
        }
        case DataType::f16: {
            const InvNormScore<&ComputeTarget::dot_f16> inv_score{vec, dim, query_inv};
            const QueryNormScore<&ComputeTarget::dist_f16_with_query_norm> query_score{vec, dim, query_norm_sq};
            return scan_dataset_ids_with_cos_scores(dataset, count, inv_score, query_score, result);
        }
        case DataType::i16: {
            const InvNormScore<&ComputeTarget::dot_i16> inv_score{vec, dim, query_inv};
            const QueryNormScore<&ComputeTarget::dist_i16_with_query_norm> query_score{vec, dim, query_norm_sq};
            return scan_dataset_ids_with_cos_scores(dataset, count, inv_score, query_score, result);
        }
        default: return Ret("Scanner::find: unsupported data type.");
    }
}

// Dataset cosine item scan variant that returns both ids and distances while
// reusing the same inverse-norm-aware dispatch logic as the id-only path.
template <typename ComputeTarget>
Ret dispatch_dataset_cos_items(DataType type, const Dataset& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result) {
    const size_t dim = dataset.dim();
    const double query_norm_sq = query_squared_norm<ComputeTarget>(type, vec, dim);
    const double query_inv = query_inverse_norm(query_norm_sq);
    switch (type) {
        case DataType::f32: {
            const InvNormScore<&ComputeTarget::dot_f32> inv_score{vec, dim, query_inv};
            const QueryNormScore<&ComputeTarget::dist_f32_with_query_norm> query_score{vec, dim, query_norm_sq};
            return scan_dataset_items_with_cos_scores(dataset, count, inv_score, query_score, result);
        }
        case DataType::f16: {
            const InvNormScore<&ComputeTarget::dot_f16> inv_score{vec, dim, query_inv};
            const QueryNormScore<&ComputeTarget::dist_f16_with_query_norm> query_score{vec, dim, query_norm_sq};
            return scan_dataset_items_with_cos_scores(dataset, count, inv_score, query_score, result);
        }
        case DataType::i16: {
            const InvNormScore<&ComputeTarget::dot_i16> inv_score{vec, dim, query_inv};
            const QueryNormScore<&ComputeTarget::dist_i16_with_query_norm> query_score{vec, dim, query_norm_sq};
            return scan_dataset_items_with_cos_scores(dataset, count, inv_score, query_score, result);
        }
        default: return Ret("Scanner::find: unsupported data type.");
    }
}

// Reader cosine KNN variant that computes query normalization once and then
// dispatches to the type-specific scorer.
template <typename ComputeTarget>
Ret dispatch_reader_cos_ids(DataType type, const DataReader& reader, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) {
    const size_t dim = reader.dim();
    const double query_norm_sq = query_squared_norm<ComputeTarget>(type, vec, dim);
    const double query_inv = query_inverse_norm(query_norm_sq);
    switch (type) {
        case DataType::f32: {
            const InvNormScore<&ComputeTarget::dot_f32> inv_score{vec, dim, query_inv};
            const QueryNormScore<&ComputeTarget::dist_f32_with_query_norm> query_score{vec, dim, query_norm_sq};
            return scan_reader_ids_with_cos_scores(reader, count, inv_score, query_score, result);
        }
        case DataType::f16: {
            const InvNormScore<&ComputeTarget::dot_f16> inv_score{vec, dim, query_inv};
            const QueryNormScore<&ComputeTarget::dist_f16_with_query_norm> query_score{vec, dim, query_norm_sq};
            return scan_reader_ids_with_cos_scores(reader, count, inv_score, query_score, result);
        }
        case DataType::i16: {
            const InvNormScore<&ComputeTarget::dot_i16> inv_score{vec, dim, query_inv};
            const QueryNormScore<&ComputeTarget::dist_i16_with_query_norm> query_score{vec, dim, query_norm_sq};
            return scan_reader_ids_with_cos_scores(reader, count, inv_score, query_score, result);
        }
        default: return Ret("Scanner::find: unsupported data type.");
    }
}

// Reader cosine item scan variant that returns full distance rows rather than ids only.
template <typename ComputeTarget>
Ret dispatch_reader_cos_items(DataType type, const DataReader& reader, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result) {
    const size_t dim = reader.dim();
    const double query_norm_sq = query_squared_norm<ComputeTarget>(type, vec, dim);
    const double query_inv = query_inverse_norm(query_norm_sq);
    switch (type) {
        case DataType::f32: {
            const InvNormScore<&ComputeTarget::dot_f32> inv_score{vec, dim, query_inv};
            const QueryNormScore<&ComputeTarget::dist_f32_with_query_norm> query_score{vec, dim, query_norm_sq};
            return scan_reader_items_with_cos_scores(reader, count, inv_score, query_score, result);
        }
        case DataType::f16: {
            const InvNormScore<&ComputeTarget::dot_f16> inv_score{vec, dim, query_inv};
            const QueryNormScore<&ComputeTarget::dist_f16_with_query_norm> query_score{vec, dim, query_norm_sq};
            return scan_reader_items_with_cos_scores(reader, count, inv_score, query_score, result);
        }
        case DataType::i16: {
            const InvNormScore<&ComputeTarget::dot_i16> inv_score{vec, dim, query_inv};
            const QueryNormScore<&ComputeTarget::dist_i16_with_query_norm> query_score{vec, dim, query_norm_sq};
            return scan_reader_items_with_cos_scores(reader, count, inv_score, query_score, result);
        }
        default: return Ret("Scanner::find: unsupported data type.");
    }
}

// Backend dispatch stays at this outer layer so scanner hot loops execute
// entirely within one concrete L1/L2/cosine implementation family.
template <typename L1Target, typename L2Target, typename CosTarget>
Ret dispatch_dataset_ids_with_backend(DataType type, DistFunc func, const Dataset& dataset, size_t count,
        const uint8_t* vec, std::vector<uint64_t>& result) {
    switch (func) {
        case DistFunc::L1:
            return dispatch_dataset_ids<L1Target>(type, dataset, count, vec, result);
        case DistFunc::L2:
            return dispatch_dataset_ids<L2Target>(type, dataset, count, vec, result);
        case DistFunc::COS:
            return dispatch_dataset_cos_ids<CosTarget>(type, dataset, count, vec, result);
        default:
            return Ret("Scanner::find: unsupported distance function.");
    }
}

template <typename L1Target, typename L2Target, typename CosTarget>
Ret dispatch_dataset_items_with_backend(DataType type, DistFunc func, const Dataset& dataset, size_t count,
        const uint8_t* vec, std::vector<DistItem>& result) {
    switch (func) {
        case DistFunc::L1:
            return dispatch_dataset_items<L1Target>(type, dataset, count, vec, result);
        case DistFunc::L2:
            return dispatch_dataset_items<L2Target>(type, dataset, count, vec, result);
        case DistFunc::COS:
            return dispatch_dataset_cos_items<CosTarget>(type, dataset, count, vec, result);
        default:
            return Ret("Scanner::find: unsupported distance function.");
    }
}

template <typename L1Target, typename L2Target, typename CosTarget>
Ret dispatch_reader_ids_with_backend(DataType type, DistFunc func, const DataReader& reader, size_t count,
        const uint8_t* vec, std::vector<uint64_t>& result) {
    switch (func) {
        case DistFunc::L1:
            return dispatch_reader_ids<L1Target>(type, reader, count, vec, result);
        case DistFunc::L2:
            return dispatch_reader_ids<L2Target>(type, reader, count, vec, result);
        case DistFunc::COS:
            return dispatch_reader_cos_ids<CosTarget>(type, reader, count, vec, result);
        default:
            return Ret("Scanner::find: not implemented");
    }
}

template <typename L1Target, typename L2Target, typename CosTarget>
Ret dispatch_reader_items_with_backend(DataType type, DistFunc func, const DataReader& reader, size_t count,
        const uint8_t* vec, std::vector<DistItem>& result) {
    switch (func) {
        case DistFunc::L1:
            return dispatch_reader_items<L1Target>(type, reader, count, vec, result);
        case DistFunc::L2:
            return dispatch_reader_items<L2Target>(type, reader, count, vec, result);
        case DistFunc::COS:
            return dispatch_reader_cos_items<CosTarget>(type, reader, count, vec, result);
        default:
            return Ret("Scanner::find: not implemented");
    }
}

} // namespace

Ret Scanner::find(const DataReader& reader, DistFunc func, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) const {
    try {
        return find_(reader, func, count, vec, result);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret Scanner::find_items(const DataReader& reader, DistFunc func, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result) const {
    try {
        return find_items_(reader, func, count, vec, result);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret Scanner::find(const Dataset& dataset, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) const {
    try {
        return find_(dataset, count, vec, result);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret Scanner::find_items(const Dataset& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result) const {
    try {
        return find_items_(dataset, count, vec, result);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

// Dataset id lookup validates once, clears output, then commits to one backend
// for the full query. That keeps the template-specialized scan path stable from
// the outer entrypoint down to the per-row distance kernel.
Ret Scanner::find_(const Dataset& dataset, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) const {
    if (vec == nullptr || count == 0) {
        return Ret("Scanner::find: invalid arguments.");
    }

    result.clear();
    const DistFunc func = dataset.dist_func();
    const DataType type = dataset.type();
    log_query_dispatch("dataset", func, type, dataset.dim(), count, false);
    switch (get_singleton().compute_unit().kind()) {
#if defined(SKETCH_ENABLE_AVX512VNNI) && SKETCH_ENABLE_AVX512VNNI && (defined(__x86_64__) || defined(__i386__))
        case ComputeBackendKind::avx512_vnni:
            return dispatch_dataset_ids_with_backend<ComputeL1_AVX512_VNNI, ComputeL2_AVX512_VNNI, ComputeCos_AVX512_VNNI>(
                type, func, dataset, count, vec, result);
#endif
#if defined(SKETCH_ENABLE_AVX512F) && SKETCH_ENABLE_AVX512F && (defined(__x86_64__) || defined(__i386__))
        case ComputeBackendKind::avx512f:
            return dispatch_dataset_ids_with_backend<ComputeL1_AVX512, ComputeL2_AVX512, ComputeCos_AVX512>(
                type, func, dataset, count, vec, result);
#endif
#if defined(SKETCH_ENABLE_AVX2) && SKETCH_ENABLE_AVX2 && (defined(__x86_64__) || defined(__i386__))
        case ComputeBackendKind::avx2:
            return dispatch_dataset_ids_with_backend<ComputeL1_AVX2, ComputeL2_AVX2, ComputeCos_AVX2>(
                type, func, dataset, count, vec, result);
#endif
#if defined(__aarch64__)
        case ComputeBackendKind::neon:
            return dispatch_dataset_ids_with_backend<ComputeL1_Neon, ComputeL2_Neon, ComputeCos_Neon>(
                type, func, dataset, count, vec, result);
#endif
        case ComputeBackendKind::scalar:
        default:
            return dispatch_dataset_ids_with_backend<ComputeL1, ComputeL2, ComputeCos>(
                type, func, dataset, count, vec, result);
    }
}

// The item-returning dataset path mirrors the id-only version but materializes
// full `DistItem` rows from the same backend-specific heap build.
Ret Scanner::find_items_(const Dataset& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>& result) const {
    if (vec == nullptr || count == 0) {
        return Ret("Scanner::find: invalid arguments.");
    }

    result.clear();
    const DistFunc func = dataset.dist_func();
    const DataType type = dataset.type();
    log_query_dispatch("dataset", func, type, dataset.dim(), count, true);
    switch (get_singleton().compute_unit().kind()) {
#if defined(SKETCH_ENABLE_AVX512VNNI) && SKETCH_ENABLE_AVX512VNNI && (defined(__x86_64__) || defined(__i386__))
        case ComputeBackendKind::avx512_vnni:
            return dispatch_dataset_items_with_backend<ComputeL1_AVX512_VNNI, ComputeL2_AVX512_VNNI, ComputeCos_AVX512_VNNI>(
                type, func, dataset, count, vec, result);
#endif
#if defined(SKETCH_ENABLE_AVX512F) && SKETCH_ENABLE_AVX512F && (defined(__x86_64__) || defined(__i386__))
        case ComputeBackendKind::avx512f:
            return dispatch_dataset_items_with_backend<ComputeL1_AVX512, ComputeL2_AVX512, ComputeCos_AVX512>(
                type, func, dataset, count, vec, result);
#endif
#if defined(SKETCH_ENABLE_AVX2) && SKETCH_ENABLE_AVX2 && (defined(__x86_64__) || defined(__i386__))
        case ComputeBackendKind::avx2:
            return dispatch_dataset_items_with_backend<ComputeL1_AVX2, ComputeL2_AVX2, ComputeCos_AVX2>(
                type, func, dataset, count, vec, result);
#endif
#if defined(__aarch64__)
        case ComputeBackendKind::neon:
            return dispatch_dataset_items_with_backend<ComputeL1_Neon, ComputeL2_Neon, ComputeCos_Neon>(
                type, func, dataset, count, vec, result);
#endif
        case ComputeBackendKind::scalar:
        default:
            return dispatch_dataset_items_with_backend<ComputeL1, ComputeL2, ComputeCos>(
                type, func, dataset, count, vec, result);
    }
}

// Reader scans use the same one-backend-per-query rule as dataset scans, but
// skip the accumulator shadowing layer because a `DataReader` is already a
// concrete persisted view.
Ret Scanner::find_(const DataReader& reader, DistFunc func, size_t count, const uint8_t* vec,
    std::vector<uint64_t>& result) const {
    if (vec == nullptr || count == 0) {
        return Ret("Scanner::find: invalid arguments.");
    }

    result.clear();
    const DataType type = reader.type();
    log_query_dispatch("reader", func, type, reader.dim(), count, false);
    switch (get_singleton().compute_unit().kind()) {
#if defined(SKETCH_ENABLE_AVX512VNNI) && SKETCH_ENABLE_AVX512VNNI && (defined(__x86_64__) || defined(__i386__))
        case ComputeBackendKind::avx512_vnni:
            return dispatch_reader_ids_with_backend<ComputeL1_AVX512_VNNI, ComputeL2_AVX512_VNNI, ComputeCos_AVX512_VNNI>(
                type, func, reader, count, vec, result);
#endif
#if defined(SKETCH_ENABLE_AVX512F) && SKETCH_ENABLE_AVX512F && (defined(__x86_64__) || defined(__i386__))
        case ComputeBackendKind::avx512f:
            return dispatch_reader_ids_with_backend<ComputeL1_AVX512, ComputeL2_AVX512, ComputeCos_AVX512>(
                type, func, reader, count, vec, result);
#endif
#if defined(SKETCH_ENABLE_AVX2) && SKETCH_ENABLE_AVX2 && (defined(__x86_64__) || defined(__i386__))
        case ComputeBackendKind::avx2:
            return dispatch_reader_ids_with_backend<ComputeL1_AVX2, ComputeL2_AVX2, ComputeCos_AVX2>(
                type, func, reader, count, vec, result);
#endif
#if defined(__aarch64__)
        case ComputeBackendKind::neon:
            return dispatch_reader_ids_with_backend<ComputeL1_Neon, ComputeL2_Neon, ComputeCos_Neon>(
                type, func, reader, count, vec, result);
#endif
        case ComputeBackendKind::scalar:
        default:
            return dispatch_reader_ids_with_backend<ComputeL1, ComputeL2, ComputeCos>(
                type, func, reader, count, vec, result);
    }
}

// Reader item lookup is the final public variant: same backend selection and
// scoring rules as `find_`, but returning ids together with computed distances.
Ret Scanner::find_items_(const DataReader& reader, DistFunc func, size_t count, const uint8_t* vec,
    std::vector<DistItem>& result) const {
    if (vec == nullptr || count == 0) {
        return Ret("Scanner::find: invalid arguments.");
    }

    result.clear();
    const DataType type = reader.type();
    log_query_dispatch("reader", func, type, reader.dim(), count, true);
    switch (get_singleton().compute_unit().kind()) {
#if defined(SKETCH_ENABLE_AVX512VNNI) && SKETCH_ENABLE_AVX512VNNI && (defined(__x86_64__) || defined(__i386__))
        case ComputeBackendKind::avx512_vnni:
            return dispatch_reader_items_with_backend<ComputeL1_AVX512_VNNI, ComputeL2_AVX512_VNNI, ComputeCos_AVX512_VNNI>(
                type, func, reader, count, vec, result);
#endif
#if defined(SKETCH_ENABLE_AVX512F) && SKETCH_ENABLE_AVX512F && (defined(__x86_64__) || defined(__i386__))
        case ComputeBackendKind::avx512f:
            return dispatch_reader_items_with_backend<ComputeL1_AVX512, ComputeL2_AVX512, ComputeCos_AVX512>(
                type, func, reader, count, vec, result);
#endif
#if defined(SKETCH_ENABLE_AVX2) && SKETCH_ENABLE_AVX2 && (defined(__x86_64__) || defined(__i386__))
        case ComputeBackendKind::avx2:
            return dispatch_reader_items_with_backend<ComputeL1_AVX2, ComputeL2_AVX2, ComputeCos_AVX2>(
                type, func, reader, count, vec, result);
#endif
#if defined(__aarch64__)
        case ComputeBackendKind::neon:
            return dispatch_reader_items_with_backend<ComputeL1_Neon, ComputeL2_Neon, ComputeCos_Neon>(
                type, func, reader, count, vec, result);
#endif
        case ComputeBackendKind::scalar:
        default:
            return dispatch_reader_items_with_backend<ComputeL1, ComputeL2, ComputeCos>(
                type, func, reader, count, vec, result);
    }
}

} // namespace sketch2
