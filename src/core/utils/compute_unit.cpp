// Defines runtime compute engine selection helpers.

#include "compute_unit.h"

#include "log.h"

#include <cstdlib>
#include <cstring>

namespace sketch2 {

namespace {

ComputeBackendKind detect_best_backend() {
    return compiled_compute_backend_kind();
}

} // namespace

ComputeUnit ComputeUnit::detect_best() {
    const ComputeBackendKind detected = detect_best_backend();
    LOG_INFO << "Compute backend set to '" << ComputeUnit(detected).name()
             << "' because it is the default compute engine.";
    return ComputeUnit(detected);
}

bool ComputeUnit::is_supported(ComputeBackendKind kind) {
    return kind == compiled_compute_backend_kind();
}

bool ComputeUnit::parse(const char* name, ComputeBackendKind* kind) {
    if (name == nullptr || kind == nullptr) {
        return false;
    }
    if (std::strcmp(name, compiled_compute_backend_name()) == 0) {
        *kind = compiled_compute_backend_kind();
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
    }
    return "unknown";
}

} // namespace sketch2
