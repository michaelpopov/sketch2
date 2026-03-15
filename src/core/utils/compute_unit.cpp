// Defines runtime compute-backend detection helpers.

#include "compute_unit.h"

#include <cstdlib>
#include <cstring>

namespace sketch2 {

namespace {

constexpr const char* kComputeBackendEnv = "SKETCH2_COMPUTE_BACKEND";

// Pick the fastest backend this binary can legally execute on the current CPU.
// If compile-time support or runtime feature bits are missing, stay on scalar
// so the selected backend always matches the code that was actually built.
ComputeBackendKind detect_best_backend() {
#if defined(__aarch64__)
    return ComputeBackendKind::neon;
#elif defined(SKETCH_ENABLE_AVX2) && SKETCH_ENABLE_AVX2 && (defined(__x86_64__) || defined(__i386__))
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2") &&
            __builtin_cpu_supports("f16c") &&
            __builtin_cpu_supports("fma")) {
        return ComputeBackendKind::avx2;
    }
#endif
    return ComputeBackendKind::scalar;
#else
    return ComputeBackendKind::scalar;
#endif
}

} // namespace

// Honor an explicit environment override first, but fall back to auto-detect
// when the override is missing, unknown, or unsupported on this machine.
ComputeUnit ComputeUnit::detect_best() {
    const char* forced = std::getenv(kComputeBackendEnv);
    if (forced != nullptr && forced[0] != '\0') {
        ComputeBackendKind kind = ComputeBackendKind::scalar;
        if (parse(forced, &kind) && is_supported(kind)) {
            return ComputeUnit(kind);
        }
    }
    return ComputeUnit(detect_best_backend());
}

// Support checks combine build-time availability with runtime CPU probing.
// That keeps testing overrides from selecting backends that were not compiled
// into this binary, while still allowing one binary to adapt per host.
bool ComputeUnit::is_supported(ComputeBackendKind kind) {
    switch (kind) {
        case ComputeBackendKind::scalar:
            return true;
        case ComputeBackendKind::avx2:
#if defined(SKETCH_ENABLE_AVX2) && SKETCH_ENABLE_AVX2 && (defined(__x86_64__) || defined(__i386__))
#if defined(__GNUC__) || defined(__clang__)
            __builtin_cpu_init();
            return __builtin_cpu_supports("avx2") &&
                __builtin_cpu_supports("f16c") &&
                __builtin_cpu_supports("fma");
#else
            return false;
#endif
#else
            return false;
#endif
        case ComputeBackendKind::neon:
#if defined(__aarch64__)
            return true;
#else
            return false;
#endif
        default:
            return false;
    }
}

// Parse accepts user-facing backend names plus "auto", which resolves eagerly
// so callers can store a concrete backend kind instead of a deferred sentinel.
bool ComputeUnit::parse(const char* name, ComputeBackendKind* kind) {
    if (name == nullptr || kind == nullptr) {
        return false;
    }
    if (std::strcmp(name, "scalar") == 0) {
        *kind = ComputeBackendKind::scalar;
        return true;
    }
    if (std::strcmp(name, "avx2") == 0) {
        *kind = ComputeBackendKind::avx2;
        return true;
    }
    if (std::strcmp(name, "neon") == 0) {
        *kind = ComputeBackendKind::neon;
        return true;
    }
    if (std::strcmp(name, "auto") == 0) {
        *kind = detect_best_backend();
        return true;
    }
    return false;
}

ComputeBackendKind ComputeUnit::kind() const {
    return kind_;
}

const char* ComputeUnit::name() const {
    switch (kind_) {
        case ComputeBackendKind::scalar:
            return "scalar";
        case ComputeBackendKind::avx2:
            return "avx2";
        case ComputeBackendKind::neon:
            return "neon";
        default:
            return "unknown";
    }
}

} // namespace sketch2
