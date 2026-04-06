// NumKong-backed distance kernels for L2, Cosine, and Dot-product metrics.
//
// NumKong provides optimized f32 and f16 implementations for squared Euclidean,
// angular (cosine), and dot-product distances.  For metric/type combinations it
// does not cover (L1, all i16 variants) the resolver falls back to Highway.

#include "core/calc/nk_kernels.h"
#include "core/calc/cosine_distance.h"
#include "core/calc/hwy_kernels.h"

#include "numkong/capabilities.h"
#include "numkong/reduce.h"
#include "numkong/dot/serial.h"
#include "numkong/spatial/serial.h"

#if NK_TARGET_X8664_
#include "numkong/dot/haswell.h"
#include "numkong/dot/skylake.h"
#include "numkong/spatial/haswell.h"
#include "numkong/spatial/skylake.h"
#endif

#if NK_TARGET_ARM64_
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
using NkNormF32Fn = void (*)(const nk_f32_t*, nk_size_t, nk_size_t, nk_f64_t*, nk_f64_t*);
using NkNormF16Fn = void (*)(const nk_f16_t*, nk_size_t, nk_size_t, nk_f32_t*, nk_f32_t*);

template <typename T, typename Acc>
inline void fused_dot_and_squared_norm(const T* a, const T* b, size_t dim, Acc* dot_out, Acc* norm_out) {
    Acc dot = 0;
    Acc norm = 0;
    for (size_t i = 0; i < dim; ++i) {
        const Acc av = static_cast<Acc>(a[i]);
        dot += av * static_cast<Acc>(b[i]);
        norm += av * av;
    }
    *dot_out = dot;
    *norm_out = norm;
}

bool is_nk_supported(DistFunc func, DataType type) {
    return func != DistFunc::L1 && type != DataType::i16;
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
NkResolvedKernel<NkNormF32Fn> resolve_norm_f32_backend(nk_capability_t caps);
NkResolvedKernel<NkNormF16Fn> resolve_norm_f16_backend(nk_capability_t caps);
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

const NkResolvedKernel<NkNormF32Fn>& norm_f32_kernel() {
    thread_local const auto kernel = resolve_norm_f32_backend(thread_capabilities());
    return kernel;
}

const NkResolvedKernel<NkNormF16Fn>& norm_f16_kernel() {
    thread_local const auto kernel = resolve_norm_f16_backend(thread_capabilities());
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

NkResolvedKernel<NkNormF32Fn> resolve_norm_f32_backend(nk_capability_t caps) {
#if NK_TARGET_X8664_ && NK_TARGET_SKYLAKE
    if (caps & nk_cap_skylake_k) return {&nk_reduce_moments_f32_skylake, "skylake", nk_cap_skylake_k};
#endif
#if NK_TARGET_X8664_ && NK_TARGET_HASWELL
    if (caps & nk_cap_haswell_k) return {&nk_reduce_moments_f32_haswell, "haswell", nk_cap_haswell_k};
#endif
#if NK_TARGET_ARM64_ && NK_TARGET_NEON
    if (caps & nk_cap_neon_k) return {&nk_reduce_moments_f32_neon, "neon", nk_cap_neon_k};
#endif
#if NK_TARGET_WASM_ && NK_TARGET_V128RELAXED
    if (caps & nk_cap_v128relaxed_k) return {&nk_reduce_moments_f32_v128relaxed, "v128relaxed", nk_cap_v128relaxed_k};
#endif
    return {&nk_reduce_moments_f32_serial, "serial", nk_cap_serial_k};
}

NkResolvedKernel<NkNormF16Fn> resolve_norm_f16_backend(nk_capability_t caps) {
#if NK_TARGET_X8664_ && NK_TARGET_SKYLAKE
    if (caps & nk_cap_skylake_k) return {&nk_reduce_moments_f16_skylake, "skylake", nk_cap_skylake_k};
#endif
#if NK_TARGET_X8664_ && NK_TARGET_HASWELL
    if (caps & nk_cap_haswell_k) return {&nk_reduce_moments_f16_haswell, "haswell", nk_cap_haswell_k};
#endif
#if NK_TARGET_ARM64_ && NK_TARGET_NEON
    if (caps & nk_cap_neon_k) return {&nk_reduce_moments_f16_neon, "neon", nk_cap_neon_k};
#endif
#if NK_TARGET_WASM_ && NK_TARGET_V128RELAXED
    if (caps & nk_cap_v128relaxed_k) return {&nk_reduce_moments_f16_v128relaxed, "v128relaxed", nk_cap_v128relaxed_k};
#endif
    return {&nk_reduce_moments_f16_serial, "serial", nk_cap_serial_k};
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

// ---------------------------------------------------------------------------
// L2 squared Euclidean
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Cosine (angular) distance
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Dot product
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Squared norm (dedicated sum-of-squares reduction)
// ---------------------------------------------------------------------------

double nk_squared_norm_f32(const uint8_t* a, size_t dim) {
    nk_f64_t sum = 0;
    nk_f64_t sumsq = 0;
    const auto& kernel = norm_f32_kernel();
    kernel.fn(reinterpret_cast<const nk_f32_t*>(a),
              static_cast<nk_size_t>(dim),
              sizeof(nk_f32_t), &sum, &sumsq);
    return static_cast<double>(sumsq);
}

double nk_squared_norm_f16(const uint8_t* a, size_t dim) {
    nk_f32_t sum = 0;
    nk_f32_t sumsq = 0;
    const auto& kernel = norm_f16_kernel();
    kernel.fn(reinterpret_cast<const nk_f16_t*>(a),
              static_cast<nk_size_t>(dim),
              sizeof(nk_f16_t), &sum, &sumsq);
    return static_cast<double>(sumsq);
}

// ---------------------------------------------------------------------------
// Cosine distance with pre-computed query norm.
// NumKong does not expose a fused "dot(a,b) + dot(a,a)" primitive here, so we
// accumulate both in one pass locally to avoid reading `a` twice.
// ---------------------------------------------------------------------------

double nk_dist_cos_qn_f32(const uint8_t* a, const uint8_t* b, size_t dim, double query_norm_sq) {
    nk_f64_t dot_result = 0;
    nk_f64_t a_norm_sq = 0;
    fused_dot_and_squared_norm(reinterpret_cast<const nk_f32_t*>(a),
                               reinterpret_cast<const nk_f32_t*>(b),
                               dim, &dot_result, &a_norm_sq);
    return finalize_cosine_distance(static_cast<double>(dot_result), static_cast<double>(a_norm_sq),
                                    query_norm_sq);
}

double nk_dist_cos_qn_f16(const uint8_t* a, const uint8_t* b, size_t dim, double query_norm_sq) {
    nk_f32_t dot_result = 0;
    nk_f32_t a_norm_sq = 0;
    fused_dot_and_squared_norm(reinterpret_cast<const nk_f16_t*>(a),
                               reinterpret_cast<const nk_f16_t*>(b),
                               dim, &dot_result, &a_norm_sq);
    return finalize_cosine_distance(static_cast<double>(dot_result), static_cast<double>(a_norm_sq),
                                    query_norm_sq);
}

} // namespace

bool nk_calc_uses_dynamic_dispatch() {
    return nk_capabilities_compiled() != nk_cap_serial_k;
}

uint64_t nk_calc_compiled_capabilities() {
    return static_cast<uint64_t>(nk_capabilities_compiled());
}

uint64_t nk_calc_available_capabilities() {
    return static_cast<uint64_t>(thread_capabilities());
}

const char* nk_calc_backend_name_for_capabilities(DistFunc func, DataType type, uint64_t capabilities) {
    const nk_capability_t caps = static_cast<nk_capability_t>(capabilities);
    if (!is_nk_supported(func, type)) {
        return "highway";
    }
    switch (func) {
        case DistFunc::L2:
            switch (type) {
                case DataType::f32: return resolve_l2_f32_backend(caps).backend;
                case DataType::f16: return resolve_l2_f16_backend(caps).backend;
                default: return "highway";
            }
        case DistFunc::COS:
            switch (type) {
                case DataType::f32: return resolve_cos_f32_backend(caps).backend;
                case DataType::f16: return resolve_cos_f16_backend(caps).backend;
                default: return "highway";
            }
        case DistFunc::L1:
        default:
            return "highway";
    }
}

const char* nk_calc_backend_name(DistFunc func, DataType type) {
    return nk_calc_backend_name_for_capabilities(func, type, nk_calc_available_capabilities());
}

// ---------------------------------------------------------------------------
// Resolver
// ---------------------------------------------------------------------------

CalcKernels resolve_nk_kernels(DistFunc func, DataType type) {
    // NumKong covers L2 and COS for f32/f16.
    // L1 and all i16 variants fall back to Highway.
    if (func == DistFunc::L1 || type == DataType::i16) {
        return resolve_hwy_kernels(func, type);
    }

    CalcKernels k;
    switch (func) {
        case DistFunc::L2:
            switch (type) {
                case DataType::f32: k.dist = &nk_dist_l2_f32; break;
                case DataType::f16: k.dist = &nk_dist_l2_f16; break;
                default:
                    throw std::runtime_error("resolve_nk_kernels: unsupported DataType for L2.");
            }
            break;
        case DistFunc::COS:
            switch (type) {
                case DataType::f32:
                    k.dist = &nk_dist_cos_f32;
                    k.dist_with_query_norm = &nk_dist_cos_qn_f32;
                    k.squared_norm = &nk_squared_norm_f32;
                    k.dot = &nk_dot_product_f32;
                    break;
                case DataType::f16:
                    k.dist = &nk_dist_cos_f16;
                    k.dist_with_query_norm = &nk_dist_cos_qn_f16;
                    k.squared_norm = &nk_squared_norm_f16;
                    k.dot = &nk_dot_product_f16;
                    break;
                default:
                    throw std::runtime_error("resolve_nk_kernels: unsupported DataType for COS.");
            }
            break;
        default:
            throw std::runtime_error("resolve_nk_kernels: unsupported DistFunc.");
    }
    return k;
}

} // namespace sketch2
