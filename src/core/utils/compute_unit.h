// Declares the process-wide Highway runtime identity used by query dispatch.

#pragma once

#include <cstdint>

namespace sketch2 {

enum class ComputeBackendKind : uint8_t {
    highway,
};

constexpr ComputeBackendKind compiled_compute_backend_kind() {
    return ComputeBackendKind::highway;
}

constexpr const char* compiled_compute_backend_name() {
    return "highway";
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
