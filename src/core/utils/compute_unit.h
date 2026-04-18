// Declares the process-wide calc backend identity used by query dispatch.

#pragma once

#include <cstdint>

namespace sketch2 {

enum class ComputeBackendKind : uint8_t {
    highway,
    nk,
};

constexpr ComputeBackendKind compiled_compute_backend_kind() {
#if SKETCH_CALC_ENGINE_HIGHWAY
    return ComputeBackendKind::highway;
#elif SKETCH_CALC_ENGINE_NUMKONG
    return ComputeBackendKind::nk;
#else
#error "Exactly one compute engine must be compiled."
#endif
}

constexpr const char* compiled_compute_backend_name() {
#if SKETCH_CALC_ENGINE_HIGHWAY
    return "highway";
#elif SKETCH_CALC_ENGINE_NUMKONG
    return "numkong";
#else
#error "Exactly one calc engine must be compiled."
#endif
}

class ComputeUnit {
public:
    ComputeUnit() = default;
    explicit constexpr ComputeUnit(ComputeBackendKind kind) : kind_(kind) {}

    static ComputeUnit detect_best();
    static bool is_supported(ComputeBackendKind kind);
    static bool parse(const char* name, ComputeBackendKind* kind);

    ComputeBackendKind kind() const;
    const char* name() const;

private:
    ComputeBackendKind kind_ = compiled_compute_backend_kind();
};

} // namespace sketch2
