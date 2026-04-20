// NumKong-backed scanner and kernel resolver.
//
// NumKong provides optimized f32 and f16 implementations for dot-product,
// squared Euclidean, and angular (cosine) distances. i16 is intentionally
// unsupported and must be rejected by callers.

#include "core/compute/numkong.h"

#include "core/compute/cosine_distance.h"
#include "core/compute/scanner_dataset_scan.h"
#include "core/compute/scanner_heap_utils.h"
#include "core/compute/scanner_log_utils.h"
#include "core/compute/scanner_query_context.h"
#include "core/utils/timer.h"

#include "numkong/capabilities.h"
#include "numkong/dot/serial.h"
#include "numkong/spatial/serial.h"

#if NK_TARGET_X8664_
#include "numkong/dot/haswell.h"
#include "numkong/dot/skylake.h"
#include "numkong/spatial/haswell.h"
#include "numkong/spatial/skylake.h"
#endif

#if NK_TARGET_ARM64_
#include <arm_neon.h>
#include "numkong/dot/neon.h"
#include "numkong/dot/neonfhm.h"
#include "numkong/dot/sve.h"
#include "numkong/dot/svehalf.h"
#include "numkong/spatial/neon.h"
#include "numkong/spatial/sve.h"
#include "numkong/spatial/svehalf.h"
#endif

#if NK_TARGET_WASM_
#include "numkong/dot/v128relaxed.h"
#include "numkong/spatial/v128relaxed.h"
#endif

#include <algorithm>
#include <cmath>
#include <exception>

namespace sketch2 {

namespace {

template <typename Fn>
struct NkResolvedKernel {
    Fn fn = nullptr;
    const char* backend = "serial";
    nk_capability_t capability = nk_cap_serial_k;
};

using NkDotF32Fn = void (*)(const nk_f32_t*, const nk_f32_t*, nk_size_t, nk_f64_t*);
using NkDotF16Fn = void (*)(const nk_f16_t*, const nk_f16_t*, nk_size_t, nk_f32_t*);
using NkSpatialF32Fn = void (*)(const nk_f32_t*, const nk_f32_t*, nk_size_t, nk_f64_t*);
using NkSpatialF16Fn = void (*)(const nk_f16_t*, const nk_f16_t*, nk_size_t, nk_f32_t*);

#if NK_TARGET_ARM64_ && NK_TARGET_NEON
inline void fused_dot_and_squared_norm_f32_neon(const nk_f32_t* a, const nk_f32_t* b, size_t dim,
        nk_f64_t* dot_out, nk_f64_t* norm_out) {
    float64x2_t dot_v = vdupq_n_f64(0);
    float64x2_t norm_v = vdupq_n_f64(0);
    size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);

        float64x2_t av_l = vcvt_f64_f32(vget_low_f32(av));
        float64x2_t av_h = vcvt_high_f64_f32(av);
        float64x2_t bv_l = vcvt_f64_f32(vget_low_f32(bv));
        float64x2_t bv_h = vcvt_high_f64_f32(bv);

        dot_v = vfmaq_f64(dot_v, av_l, bv_l);
        dot_v = vfmaq_f64(dot_v, av_h, bv_h);
        norm_v = vfmaq_f64(norm_v, av_l, av_l);
        norm_v = vfmaq_f64(norm_v, av_h, av_h);
    }
    nk_f64_t dot = vaddvq_f64(dot_v);
    nk_f64_t norm = vaddvq_f64(norm_v);
    for (; i < dim; ++i) {
        const nk_f64_t av = static_cast<nk_f64_t>(a[i]);
        dot += av * static_cast<nk_f64_t>(b[i]);
        norm += av * av;
    }
    *dot_out = dot;
    *norm_out = norm;
}
#endif

bool is_nk_supported(DataType type) {
    return type != DataType::i16;
}

nk_capability_t init_thread_capabilities() {
    nk_capability_t caps = nk_capabilities_available();
    if (caps == 0) {
        caps = nk_cap_serial_k;
    }
    nk_configure_thread(caps);
    return caps;
}

nk_capability_t thread_capabilities() {
    thread_local const nk_capability_t caps = init_thread_capabilities();
    return caps;
}

NkResolvedKernel<NkDotF32Fn> resolve_dot_f32_backend(nk_capability_t caps);
NkResolvedKernel<NkDotF16Fn> resolve_dot_f16_backend(nk_capability_t caps);
NkResolvedKernel<NkSpatialF32Fn> resolve_l2_f32_backend(nk_capability_t caps);
NkResolvedKernel<NkSpatialF16Fn> resolve_l2_f16_backend(nk_capability_t caps);
NkResolvedKernel<NkSpatialF32Fn> resolve_cos_f32_backend(nk_capability_t caps);
NkResolvedKernel<NkSpatialF16Fn> resolve_cos_f16_backend(nk_capability_t caps);

const NkResolvedKernel<NkDotF32Fn>& dot_f32_kernel() {
    thread_local const auto kernel = resolve_dot_f32_backend(thread_capabilities());
    return kernel;
}

const NkResolvedKernel<NkDotF16Fn>& dot_f16_kernel() {
    thread_local const auto kernel = resolve_dot_f16_backend(thread_capabilities());
    return kernel;
}

const NkResolvedKernel<NkSpatialF32Fn>& l2_f32_kernel() {
    thread_local const auto kernel = resolve_l2_f32_backend(thread_capabilities());
    return kernel;
}

const NkResolvedKernel<NkSpatialF16Fn>& l2_f16_kernel() {
    thread_local const auto kernel = resolve_l2_f16_backend(thread_capabilities());
    return kernel;
}

const NkResolvedKernel<NkSpatialF32Fn>& cos_f32_kernel() {
    thread_local const auto kernel = resolve_cos_f32_backend(thread_capabilities());
    return kernel;
}

const NkResolvedKernel<NkSpatialF16Fn>& cos_f16_kernel() {
    thread_local const auto kernel = resolve_cos_f16_backend(thread_capabilities());
    return kernel;
}

NkResolvedKernel<NkDotF32Fn> resolve_dot_f32_backend(nk_capability_t caps) {
#if NK_TARGET_X8664_ && NK_TARGET_SKYLAKE
    if (caps & nk_cap_skylake_k) return {&nk_dot_f32_skylake, "skylake", nk_cap_skylake_k};
#endif
#if NK_TARGET_X8664_ && NK_TARGET_HASWELL
    if (caps & nk_cap_haswell_k) return {&nk_dot_f32_haswell, "haswell", nk_cap_haswell_k};
#endif
#if NK_TARGET_ARM64_ && NK_TARGET_SVE
    if (caps & nk_cap_sve_k) return {&nk_dot_f32_sve, "sve", nk_cap_sve_k};
#endif
#if NK_TARGET_ARM64_ && NK_TARGET_NEON
    if (caps & nk_cap_neon_k) return {&nk_dot_f32_neon, "neon", nk_cap_neon_k};
#endif
#if NK_TARGET_WASM_ && NK_TARGET_V128RELAXED
    if (caps & nk_cap_v128relaxed_k) return {&nk_dot_f32_v128relaxed, "v128relaxed", nk_cap_v128relaxed_k};
#endif
    return {&nk_dot_f32_serial, "serial", nk_cap_serial_k};
}

NkResolvedKernel<NkDotF16Fn> resolve_dot_f16_backend(nk_capability_t caps) {
#if NK_TARGET_X8664_ && NK_TARGET_SKYLAKE
    if (caps & nk_cap_skylake_k) return {&nk_dot_f16_skylake, "skylake", nk_cap_skylake_k};
#endif
#if NK_TARGET_X8664_ && NK_TARGET_HASWELL
    if (caps & nk_cap_haswell_k) return {&nk_dot_f16_haswell, "haswell", nk_cap_haswell_k};
#endif
#if NK_TARGET_ARM64_ && NK_TARGET_SVEHALF
    if (caps & nk_cap_svehalf_k) return {&nk_dot_f16_svehalf, "svehalf", nk_cap_svehalf_k};
#endif
#if NK_TARGET_ARM64_ && NK_TARGET_NEONFHM
    if (caps & nk_cap_neonfhm_k) return {&nk_dot_f16_neonfhm, "neonfhm", nk_cap_neonfhm_k};
#endif
#if NK_TARGET_ARM64_ && NK_TARGET_NEON
    if (caps & nk_cap_neon_k) return {&nk_dot_f16_neon, "neon", nk_cap_neon_k};
#endif
#if NK_TARGET_WASM_ && NK_TARGET_V128RELAXED
    if (caps & nk_cap_v128relaxed_k) return {&nk_dot_f16_v128relaxed, "v128relaxed", nk_cap_v128relaxed_k};
#endif
    return {&nk_dot_f16_serial, "serial", nk_cap_serial_k};
}

NkResolvedKernel<NkSpatialF32Fn> resolve_l2_f32_backend(nk_capability_t caps) {
#if NK_TARGET_X8664_ && NK_TARGET_SKYLAKE
    if (caps & nk_cap_skylake_k) return {&nk_sqeuclidean_f32_skylake, "skylake", nk_cap_skylake_k};
#endif
#if NK_TARGET_X8664_ && NK_TARGET_HASWELL
    if (caps & nk_cap_haswell_k) return {&nk_sqeuclidean_f32_haswell, "haswell", nk_cap_haswell_k};
#endif
#if NK_TARGET_ARM64_ && NK_TARGET_SVE
    if (caps & nk_cap_sve_k) return {&nk_sqeuclidean_f32_sve, "sve", nk_cap_sve_k};
#endif
#if NK_TARGET_ARM64_ && NK_TARGET_NEON
    if (caps & nk_cap_neon_k) return {&nk_sqeuclidean_f32_neon, "neon", nk_cap_neon_k};
#endif
#if NK_TARGET_WASM_ && NK_TARGET_V128RELAXED
    if (caps & nk_cap_v128relaxed_k) return {&nk_sqeuclidean_f32_v128relaxed, "v128relaxed", nk_cap_v128relaxed_k};
#endif
    return {&nk_sqeuclidean_f32_serial, "serial", nk_cap_serial_k};
}

NkResolvedKernel<NkSpatialF16Fn> resolve_l2_f16_backend(nk_capability_t caps) {
#if NK_TARGET_X8664_ && NK_TARGET_SKYLAKE
    if (caps & nk_cap_skylake_k) return {&nk_sqeuclidean_f16_skylake, "skylake", nk_cap_skylake_k};
#endif
#if NK_TARGET_X8664_ && NK_TARGET_HASWELL
    if (caps & nk_cap_haswell_k) return {&nk_sqeuclidean_f16_haswell, "haswell", nk_cap_haswell_k};
#endif
#if NK_TARGET_ARM64_ && NK_TARGET_SVEHALF
    if (caps & nk_cap_svehalf_k) return {&nk_sqeuclidean_f16_svehalf, "svehalf", nk_cap_svehalf_k};
#endif
#if NK_TARGET_ARM64_ && NK_TARGET_NEON
    if (caps & nk_cap_neon_k) return {&nk_sqeuclidean_f16_neon, "neon", nk_cap_neon_k};
#endif
#if NK_TARGET_WASM_ && NK_TARGET_V128RELAXED
    if (caps & nk_cap_v128relaxed_k) return {&nk_sqeuclidean_f16_v128relaxed, "v128relaxed", nk_cap_v128relaxed_k};
#endif
    return {&nk_sqeuclidean_f16_serial, "serial", nk_cap_serial_k};
}

NkResolvedKernel<NkSpatialF32Fn> resolve_cos_f32_backend(nk_capability_t caps) {
#if NK_TARGET_X8664_ && NK_TARGET_SKYLAKE
    if (caps & nk_cap_skylake_k) return {&nk_angular_f32_skylake, "skylake", nk_cap_skylake_k};
#endif
#if NK_TARGET_X8664_ && NK_TARGET_HASWELL
    if (caps & nk_cap_haswell_k) return {&nk_angular_f32_haswell, "haswell", nk_cap_haswell_k};
#endif
#if NK_TARGET_ARM64_ && NK_TARGET_SVE
    if (caps & nk_cap_sve_k) return {&nk_angular_f32_sve, "sve", nk_cap_sve_k};
#endif
#if NK_TARGET_ARM64_ && NK_TARGET_NEON
    if (caps & nk_cap_neon_k) return {&nk_angular_f32_neon, "neon", nk_cap_neon_k};
#endif
#if NK_TARGET_WASM_ && NK_TARGET_V128RELAXED
    if (caps & nk_cap_v128relaxed_k) return {&nk_angular_f32_v128relaxed, "v128relaxed", nk_cap_v128relaxed_k};
#endif
    return {&nk_angular_f32_serial, "serial", nk_cap_serial_k};
}

NkResolvedKernel<NkSpatialF16Fn> resolve_cos_f16_backend(nk_capability_t caps) {
#if NK_TARGET_X8664_ && NK_TARGET_SKYLAKE
    if (caps & nk_cap_skylake_k) return {&nk_angular_f16_skylake, "skylake", nk_cap_skylake_k};
#endif
#if NK_TARGET_X8664_ && NK_TARGET_HASWELL
    if (caps & nk_cap_haswell_k) return {&nk_angular_f16_haswell, "haswell", nk_cap_haswell_k};
#endif
#if NK_TARGET_ARM64_ && NK_TARGET_SVEHALF
    if (caps & nk_cap_svehalf_k) return {&nk_angular_f16_svehalf, "svehalf", nk_cap_svehalf_k};
#endif
#if NK_TARGET_ARM64_ && NK_TARGET_NEON
    if (caps & nk_cap_neon_k) return {&nk_angular_f16_neon, "neon", nk_cap_neon_k};
#endif
#if NK_TARGET_WASM_ && NK_TARGET_V128RELAXED
    if (caps & nk_cap_v128relaxed_k) return {&nk_angular_f16_v128relaxed, "v128relaxed", nk_cap_v128relaxed_k};
#endif
    return {&nk_angular_f16_serial, "serial", nk_cap_serial_k};
}

double nk_dist_l2_f32(const uint8_t* a, const uint8_t* b, size_t dim) {
    nk_f64_t result;
    const auto& kernel = l2_f32_kernel();
    kernel.fn(reinterpret_cast<const nk_f32_t*>(a),
              reinterpret_cast<const nk_f32_t*>(b),
              static_cast<nk_size_t>(dim), &result);
    return static_cast<double>(result);
}

double nk_dist_l2_f16(const uint8_t* a, const uint8_t* b, size_t dim) {
    nk_f32_t result;
    const auto& kernel = l2_f16_kernel();
    kernel.fn(reinterpret_cast<const nk_f16_t*>(a),
              reinterpret_cast<const nk_f16_t*>(b),
              static_cast<nk_size_t>(dim), &result);
    return static_cast<double>(result);
}

double nk_dist_cos_f32(const uint8_t* a, const uint8_t* b, size_t dim) {
    nk_f64_t result;
    const auto& kernel = cos_f32_kernel();
    kernel.fn(reinterpret_cast<const nk_f32_t*>(a),
              reinterpret_cast<const nk_f32_t*>(b),
              static_cast<nk_size_t>(dim), &result);
    return static_cast<double>(result);
}

double nk_dist_cos_f16(const uint8_t* a, const uint8_t* b, size_t dim) {
    nk_f32_t result;
    const auto& kernel = cos_f16_kernel();
    kernel.fn(reinterpret_cast<const nk_f16_t*>(a),
              reinterpret_cast<const nk_f16_t*>(b),
              static_cast<nk_size_t>(dim), &result);
    return static_cast<double>(result);
}

double nk_dot_product_f32(const uint8_t* a, const uint8_t* b, size_t dim) {
    nk_f64_t result;
    const auto& kernel = dot_f32_kernel();
    kernel.fn(reinterpret_cast<const nk_f32_t*>(a),
              reinterpret_cast<const nk_f32_t*>(b),
              static_cast<nk_size_t>(dim), &result);
    return static_cast<double>(result);
}

double nk_dot_product_f16(const uint8_t* a, const uint8_t* b, size_t dim) {
    nk_f32_t result;
    const auto& kernel = dot_f16_kernel();
    kernel.fn(reinterpret_cast<const nk_f16_t*>(a),
              reinterpret_cast<const nk_f16_t*>(b),
              static_cast<nk_size_t>(dim), &result);
    return static_cast<double>(result);
}

double nk_squared_norm_f32(const uint8_t* a, size_t dim) {
    nk_f64_t result = 0;
    const auto& kernel = dot_f32_kernel();
    kernel.fn(reinterpret_cast<const nk_f32_t*>(a),
              reinterpret_cast<const nk_f32_t*>(a),
              static_cast<nk_size_t>(dim), &result);
    return static_cast<double>(result);
}

double nk_squared_norm_f16(const uint8_t* a, size_t dim) {
    nk_f32_t result = 0;
    const auto& kernel = dot_f16_kernel();
    kernel.fn(reinterpret_cast<const nk_f16_t*>(a),
              reinterpret_cast<const nk_f16_t*>(a),
              static_cast<nk_size_t>(dim), &result);
    return static_cast<double>(result);
}

double nk_dist_cos_qn_f32(const uint8_t* a, const uint8_t* b, size_t dim, double query_norm_sq) {
    nk_f64_t dot_result = 0;
    nk_f64_t a_norm_sq = 0;
#if NK_TARGET_ARM64_ && NK_TARGET_NEON
    fused_dot_and_squared_norm_f32_neon(reinterpret_cast<const nk_f32_t*>(a),
                                        reinterpret_cast<const nk_f32_t*>(b),
                                        dim, &dot_result, &a_norm_sq);
#else
    dot_result = static_cast<nk_f64_t>(nk_dot_product_f32(a, b, dim));
    a_norm_sq = static_cast<nk_f64_t>(nk_squared_norm_f32(a, dim));
#endif
    return finalize_cosine_distance(static_cast<double>(dot_result), static_cast<double>(a_norm_sq),
                                    query_norm_sq);
}

double nk_dist_cos_qn_f16(const uint8_t* a, const uint8_t* b, size_t dim, double query_norm_sq) {
    const double dot_result = nk_dot_product_f16(a, b, dim);
    const double a_norm_sq = nk_squared_norm_f16(a, dim);
    return finalize_cosine_distance(static_cast<double>(dot_result),
                                    static_cast<double>(a_norm_sq),
                                    query_norm_sq);
}

Ret validate_nk_kernel_support(DistFunc func, DataType type) {
    try {
        const ComputeKernels kernels = resolve_nk_kernels(func, type);
        if (kernels.dist == nullptr) {
            return Ret(std::string("Numkong::find_items: missing dist kernel for ")
                + dist_func_to_string(func) + "/" + data_type_to_string(type) + ".");
        }
        if ((func == DistFunc::L2 || func == DistFunc::COS)
                && (kernels.dot == nullptr || kernels.squared_norm == nullptr)) {
            return Ret(std::string("Numkong::find_items: missing stored-norm helpers for ")
                + dist_func_to_string(func) + "/" + data_type_to_string(type) + ".");
        }
        if (func == DistFunc::COS && kernels.dist_with_query_norm == nullptr) {
            return Ret(std::string("Numkong::find_items: missing query-norm kernel for ")
                + dist_func_to_string(func) + "/" + data_type_to_string(type) + ".");
        }
        return Ret(0);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

#define NK_FOR_EACH_DOT_KERNEL(X) \
    X(f32, nk_dot_product_f32) \
    X(f16, nk_dot_product_f16)

#define NK_FOR_EACH_L2_KERNEL(X) \
    X(f32, nk_dot_product_f32, nk_dist_l2_f32, nk_squared_norm_f32) \
    X(f16, nk_dot_product_f16, nk_dist_l2_f16, nk_squared_norm_f16)

#define NK_FOR_EACH_COS_KERNEL(X) \
    X(f32, nk_dot_product_f32, nk_dist_cos_f32, nk_dist_cos_qn_f32, nk_squared_norm_f32) \
    X(f16, nk_dot_product_f16, nk_dist_cos_f16, nk_dist_cos_qn_f16, nk_squared_norm_f16)

#define NK_DISPATCH_COS_CASE(TYPE_TAG, DOT_KERNEL, DIST_KERNEL, DIST_QN_KERNEL, SQ_NORM_KERNEL) \
    case DataType::TYPE_TAG: { \
        const double query_norm_sq = SQ_NORM_KERNEL(vec, dim); \
        log_query_branch(query_id, "cos_with_optional_norms", \
            query_norm_sq, query_norm_sq == 0.0); \
        const QueryCosContext query{ \
            vec, dim, query_norm_sq, query_inverse_norm(query_norm_sq)}; \
        CHECK((scan_dataset_heap_with_optional_cosine_norms< \
            DOT_KERNEL, DIST_QN_KERNEL>( \
            query_id, dataset, count, &heap, query, func, bitset))); \
        break; \
    }

#define NK_DISPATCH_DOT_CASE(TYPE_TAG, DOT_KERNEL) \
    case DataType::TYPE_TAG: \
        CHECK(scan_dataset_heap_with_dot<DOT_KERNEL>( \
            query_id, dataset, count, &heap, query, func, bitset)); \
        break;

#define NK_DISPATCH_L2_CASE(TYPE_TAG, DOT_KERNEL, DIST_KERNEL, SQ_NORM_KERNEL) \
    case DataType::TYPE_TAG: { \
        const double query_norm_sq = SQ_NORM_KERNEL(vec, dim); \
        log_query_branch(query_id, "l2_with_optional_norms", \
            query_norm_sq, query_norm_sq == 0.0); \
        const QueryL2Context query{vec, dim, query_norm_sq}; \
        CHECK((scan_dataset_heap_with_optional_l2_norms< \
            DOT_KERNEL, DIST_KERNEL>( \
            query_id, dataset, count, &heap, query, func, bitset))); \
        break; \
    }

} // namespace

Ret find_items_nk(const DatasetReader& dataset, size_t count, const uint8_t* vec,
        std::vector<DistItem>* result, const BitsetFilter* bitset, uint64_t query_id) {
    if (vec == nullptr || count == 0 || result == nullptr) {
        return Ret("Numkong::find_items: invalid arguments.");
    }
    if (dataset.type() == DataType::i16) {
        return Ret("Numkong::find_items: NumKong does not support i16 datasets.");
    }

    result->clear();
    const DistFunc func = dataset.dist_func();
    const size_t dim = dataset.dim();
    const DataType type = dataset.type();
    CHECK(validate_nk_kernel_support(func, type));
    if (query_id == 0) {
        query_id = next_scanner_query_id();
    }
    log_query_start(query_id, dataset.name(), func, type, dim, count,
        ComputeEngine::numkong, bitset != nullptr, nk_compute_backend_name(func, type),
        nk_compute_compiled_capabilities(), nk_compute_available_capabilities());

    DistHeap heap(DistItemCompare{func});
    heap.reserve(count);
    Timer timer("numkong::query");

    if (func == DistFunc::COS) {
        switch (type) {
            NK_FOR_EACH_COS_KERNEL(NK_DISPATCH_COS_CASE);
            default:
                return Ret("Numkong::find_items: unsupported DataType for COS.");
        }
    } else if (func == DistFunc::DOT) {
        log_query_branch(query_id, "dot");
        const QueryDotContext query{vec, dim};
        switch (type) {
            NK_FOR_EACH_DOT_KERNEL(NK_DISPATCH_DOT_CASE);
            default:
                return Ret("Numkong::find_items: unsupported DataType for DOT.");
        }
    } else if (func == DistFunc::L2) {
        switch (type) {
            NK_FOR_EACH_L2_KERNEL(NK_DISPATCH_L2_CASE);
            default:
                return Ret("Numkong::find_items: unsupported DataType for L2.");
        }
    } else {
        return Ret("Numkong::find_items: unsupported DistFunc.");
    }

    extract_items(&heap, result);
    log_query_finish(query_id, dataset.name(), func, type, dim, count,
        ComputeEngine::numkong, timer.elapsed_ms(), *result);
    return Ret(0);
}

bool nk_compute_uses_dynamic_dispatch() {
    return nk_capabilities_compiled() != nk_cap_serial_k;
}

uint64_t nk_compute_compiled_capabilities() {
    return static_cast<uint64_t>(nk_capabilities_compiled());
}

uint64_t nk_compute_available_capabilities() {
    return static_cast<uint64_t>(thread_capabilities());
}

const char* nk_compute_backend_name_for_capabilities(DistFunc func, DataType type, uint64_t capabilities) {
    const nk_capability_t caps = static_cast<nk_capability_t>(capabilities);
    if (!is_nk_supported(type)) {
        return "unsupported";
    }
    switch (func) {
        case DistFunc::L2:
            switch (type) {
                case DataType::f32: return resolve_l2_f32_backend(caps).backend;
                case DataType::f16: return resolve_l2_f16_backend(caps).backend;
                default: return "unsupported";
            }
        case DistFunc::COS:
            switch (type) {
                case DataType::f32: return resolve_cos_f32_backend(caps).backend;
                case DataType::f16: return resolve_cos_f16_backend(caps).backend;
                default: return "unsupported";
            }
        case DistFunc::DOT:
            switch (type) {
                case DataType::f32: return resolve_dot_f32_backend(caps).backend;
                case DataType::f16: return resolve_dot_f16_backend(caps).backend;
                default: return "unsupported";
            }
        default:
            return "unsupported";
    }
}

const char* nk_compute_backend_name(DistFunc func, DataType type) {
    return nk_compute_backend_name_for_capabilities(func, type, nk_compute_available_capabilities());
}

ComputeKernels resolve_nk_kernels(DistFunc func, DataType type) {
    if (type == DataType::i16) {
        throw std::runtime_error("resolve_nk_kernels: i16 is not supported by NumKong.");
    }

    ComputeKernels k;
    switch (func) {
        case DistFunc::DOT:
            switch (type) {
#define NK_RESOLVE_DOT_CASE(TYPE_TAG, DOT_KERNEL) \
                case DataType::TYPE_TAG: k.dist = &DOT_KERNEL; k.dot = &DOT_KERNEL; break;
                NK_FOR_EACH_DOT_KERNEL(NK_RESOLVE_DOT_CASE)
#undef NK_RESOLVE_DOT_CASE
                default:
                    throw std::runtime_error("resolve_nk_kernels: unsupported DataType for DOT.");
            }
            break;
        case DistFunc::L2:
            switch (type) {
#define NK_RESOLVE_L2_CASE(TYPE_TAG, DOT_KERNEL, DIST_KERNEL, SQ_NORM_KERNEL) \
                case DataType::TYPE_TAG: \
                    k.dist = &DIST_KERNEL; \
                    k.squared_norm = &SQ_NORM_KERNEL; \
                    k.dot = &DOT_KERNEL; \
                    break;
                NK_FOR_EACH_L2_KERNEL(NK_RESOLVE_L2_CASE)
#undef NK_RESOLVE_L2_CASE
                default:
                    throw std::runtime_error("resolve_nk_kernels: unsupported DataType for L2.");
            }
            break;
        case DistFunc::COS:
            switch (type) {
#define NK_RESOLVE_COS_CASE(TYPE_TAG, DOT_KERNEL, DIST_KERNEL, DIST_QN_KERNEL, SQ_NORM_KERNEL) \
                case DataType::TYPE_TAG: \
                    k.dist = &DIST_KERNEL; \
                    k.dist_with_query_norm = &DIST_QN_KERNEL; \
                    k.squared_norm = &SQ_NORM_KERNEL; \
                    k.dot = &DOT_KERNEL; \
                    break;
                NK_FOR_EACH_COS_KERNEL(NK_RESOLVE_COS_CASE)
#undef NK_RESOLVE_COS_CASE
                default:
                    throw std::runtime_error("resolve_nk_kernels: unsupported DataType for COS.");
            }
            break;
        default:
            throw std::runtime_error("resolve_nk_kernels: unsupported DistFunc.");
    }
    return k;
}

#undef NK_DISPATCH_COS_CASE
#undef NK_DISPATCH_DOT_CASE
#undef NK_DISPATCH_L2_CASE
#undef NK_FOR_EACH_DOT_KERNEL
#undef NK_FOR_EACH_L2_KERNEL
#undef NK_FOR_EACH_COS_KERNEL

} // namespace sketch2
