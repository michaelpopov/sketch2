// Declares the runtime-selected compute backend used by query dispatch.

#pragma once

#include <cstdint>

namespace sketch2 {

enum class ComputeBackendKind : uint8_t {
    scalar,
    avx2,
    avx512f,
    avx512_vnni,
    neon,
};

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
    ComputeBackendKind kind_ = ComputeBackendKind::scalar;
};

} // namespace sketch2
