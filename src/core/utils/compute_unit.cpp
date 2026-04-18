// Defines runtime calc-engine selection helpers.

#include "compute_unit.h"

#include "log.h"

#include <cstdlib>
#include <cstring>

namespace sketch2 {

namespace {

ComputeBackendKind detect_best_backend() {
    return ComputeBackendKind::highway;
}

} // namespace

ComputeUnit ComputeUnit::detect_best() {
    const ComputeBackendKind detected = detect_best_backend();
    LOG_INFO << "Compute backend set to '" << ComputeUnit(detected).name()
             << "' because it is the default calc engine.";
    return ComputeUnit(detected);
}

bool ComputeUnit::is_supported(ComputeBackendKind kind) {
    switch (kind) {
        case ComputeBackendKind::highway:
        case ComputeBackendKind::nk:
            return true;
        default:
            return false;
    }
}

bool ComputeUnit::parse(const char* name, ComputeBackendKind* kind) {
    if (name == nullptr || kind == nullptr) {
        return false;
    }
    if (std::strcmp(name, "highway") == 0) {
        *kind = ComputeBackendKind::highway;
        return true;
    }
    if (std::strcmp(name, "numkong") == 0) {
        *kind = ComputeBackendKind::nk;
        return true;
    }
    return false;
}

ComputeBackendKind ComputeUnit::kind() const {
    return kind_;
}

const char* ComputeUnit::name() const {
    switch (kind_) {
        case ComputeBackendKind::highway:
            return "highway";
        case ComputeBackendKind::nk:
            return "numkong";
        default:
            return "unknown";
    }
}

} // namespace sketch2
