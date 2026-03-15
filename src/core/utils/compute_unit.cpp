// Defines runtime compute-backend detection helpers.

#include "compute_unit.h"

#include "log.h"

#include <cstdlib>
#include <cstring>

namespace sketch2 {

namespace {

constexpr const char* kComputeBackendEnv = "SKETCH2_COMPUTE_BACKEND";

const char* auto_detection_reason(ComputeBackendKind kind) {
    switch (kind) {
        case ComputeBackendKind::neon:
            return "auto-detected NEON on AArch64.";
        case ComputeBackendKind::avx512_vnni:
            return "auto-detected AVX-512 VNNI because the required AVX-512F/BW/VL/VNNI CPU features are present.";
        case ComputeBackendKind::avx512f:
            return "auto-detected AVX-512F because AVX-512F is present and no higher-priority supported backend was available.";
        case ComputeBackendKind::avx2:
            return "auto-detected AVX2 because AVX2/F16C/FMA are present and no higher-priority supported backend was available.";
        case ComputeBackendKind::scalar:
            return "fell back to scalar because no supported SIMD backend was available on this build/CPU.";
        default:
            return "selected an unknown backend.";
    }
}

#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
void ensure_x86_cpu_features_initialized() {
    static const bool initialized = []() {
        __builtin_cpu_init();
        return true;
    }();
    (void)initialized;
}

bool cpu_supports_avx2_backend() {
    return __builtin_cpu_supports("avx2") &&
        __builtin_cpu_supports("f16c") &&
        __builtin_cpu_supports("fma");
}

bool cpu_supports_avx512f_backend() {
    return __builtin_cpu_supports("avx512f");
}

bool cpu_supports_avx512_vnni_backend() {
    return __builtin_cpu_supports("avx512f") &&
        __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512vl") &&
        __builtin_cpu_supports("avx512vnni");
}

bool cpu_supports_backend(ComputeBackendKind kind) {
    ensure_x86_cpu_features_initialized();
    switch (kind) {
        case ComputeBackendKind::avx512_vnni:
            return cpu_supports_avx512_vnni_backend();
        case ComputeBackendKind::avx512f:
            return cpu_supports_avx512f_backend();
        case ComputeBackendKind::avx2:
            return cpu_supports_avx2_backend();
        default:
            return false;
    }
}
#endif

// Pick the fastest backend this binary can legally execute on the current CPU.
// If compile-time support or runtime feature bits are missing, stay on scalar
// so the selected backend always matches the code that was actually built.
ComputeBackendKind detect_best_backend() {
#if defined(__aarch64__)
    return ComputeBackendKind::neon;
#elif (defined(__x86_64__) || defined(__i386__))
#if defined(SKETCH_ENABLE_AVX512VNNI) && SKETCH_ENABLE_AVX512VNNI && (defined(__GNUC__) || defined(__clang__))
    if (cpu_supports_backend(ComputeBackendKind::avx512_vnni)) {
        return ComputeBackendKind::avx512_vnni;
    }
#endif
#if defined(SKETCH_ENABLE_AVX512F) && SKETCH_ENABLE_AVX512F && (defined(__GNUC__) || defined(__clang__))
    if (cpu_supports_backend(ComputeBackendKind::avx512f)) {
        return ComputeBackendKind::avx512f;
    }
#endif
#if defined(SKETCH_ENABLE_AVX2) && SKETCH_ENABLE_AVX2 && (defined(__GNUC__) || defined(__clang__))
    if (cpu_supports_backend(ComputeBackendKind::avx2)) {
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
            LOG_INFO << "Compute backend set to '" << ComputeUnit(kind).name()
                     << "' because SKETCH2_COMPUTE_BACKEND explicitly requested it and that backend is supported.";
            return ComputeUnit(kind);
        }

        const ComputeBackendKind detected = detect_best_backend();
        if (parse(forced, &kind)) {
            LOG_INFO << "Compute backend set to '" << ComputeUnit(detected).name()
                     << "' because SKETCH2_COMPUTE_BACKEND requested '" << forced
                     << "', but that backend is not supported on this build/CPU; "
                     << auto_detection_reason(detected);
        } else {
            LOG_INFO << "Compute backend set to '" << ComputeUnit(detected).name()
                     << "' because SKETCH2_COMPUTE_BACKEND requested unknown backend '" << forced
                     << "'; " << auto_detection_reason(detected);
        }
        return ComputeUnit(detected);
    }

    const ComputeBackendKind detected = detect_best_backend();
    LOG_INFO << "Compute backend set to '" << ComputeUnit(detected).name()
             << "' because " << auto_detection_reason(detected);
    return ComputeUnit(detected);
}

// Support checks combine build-time availability with runtime CPU probing.
// That keeps testing overrides from selecting backends that were not compiled
// into this binary, while still allowing one binary to adapt per host.
bool ComputeUnit::is_supported(ComputeBackendKind kind) {
    switch (kind) {
        case ComputeBackendKind::scalar:
            return true;
        case ComputeBackendKind::avx512_vnni:
#if defined(SKETCH_ENABLE_AVX512VNNI) && SKETCH_ENABLE_AVX512VNNI && (defined(__x86_64__) || defined(__i386__))
#if defined(__GNUC__) || defined(__clang__)
            return cpu_supports_backend(kind);
#else
            return false;
#endif
#else
            return false;
#endif
        case ComputeBackendKind::avx512f:
#if defined(SKETCH_ENABLE_AVX512F) && SKETCH_ENABLE_AVX512F && (defined(__x86_64__) || defined(__i386__))
#if defined(__GNUC__) || defined(__clang__)
            return cpu_supports_backend(kind);
#else
            return false;
#endif
#else
            return false;
#endif
        case ComputeBackendKind::avx2:
#if defined(SKETCH_ENABLE_AVX2) && SKETCH_ENABLE_AVX2 && (defined(__x86_64__) || defined(__i386__))
#if defined(__GNUC__) || defined(__clang__)
            return cpu_supports_backend(kind);
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
    if (std::strcmp(name, "avx512f") == 0) {
        *kind = ComputeBackendKind::avx512f;
        return true;
    }
    if (std::strcmp(name, "avx512_vnni") == 0) {
        *kind = ComputeBackendKind::avx512_vnni;
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
        case ComputeBackendKind::avx512f:
            return "avx512f";
        case ComputeBackendKind::avx512_vnni:
            return "avx512_vnni";
        case ComputeBackendKind::neon:
            return "neon";
        default:
            return "unknown";
    }
}

} // namespace sketch2
