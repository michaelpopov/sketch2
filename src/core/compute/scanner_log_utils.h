// Shared scanner logging helpers.

#pragma once

#include "core/compute/compute_engine.h"
#include "core/utils/log.h"

namespace sketch2 {

inline const char* scanner_dist_func_name(DistFunc func) {
    switch (func) {
        case DistFunc::DOT: return "DOT";
        case DistFunc::L2: return "L2";
        case DistFunc::COS: return "COS";
        default: return "unknown";
    }
}

inline void log_query(const std::string& source, DistFunc func, DataType type, size_t dim,
        size_t count, ComputeEngine engine, int64_t elapsed_ms) {
    LOG_TRACE << "ScannerEx query: source=" << source
              << " engine=" << calc_engine_name(engine)
              << " metric=" << scanner_dist_func_name(func)
              << " type=" << data_type_to_string(type)
              << " dim=" << dim
              << " k=" << count
              << " time=" << elapsed_ms << " ms";
}

} // namespace sketch2
