#pragma once
#include "utils/shared_types.h"
#include <cstdint>
#include <string>
#include <stdexcept>

namespace sketch2 {

enum class DistFunc {
    L1,
    L2,
};

inline const char* dist_func_to_string(DistFunc func) {
    switch (func) {
        case DistFunc::L1: return "l1";
        case DistFunc::L2: return "l2";
        default: throw std::runtime_error("Invalid distance function.");
    }
}

inline DistFunc dist_func_from_string(const std::string& func_str) {
    if (func_str == "l1") return DistFunc::L1;
    if (func_str == "l2") return DistFunc::L2;
    throw std::runtime_error("Invalid distance function string.");
}

inline void validate_dist_func(DistFunc func) {
    (void)dist_func_to_string(func);
}

struct DistItem {
    uint64_t id;
    double   dist;

    struct Compare {
        bool operator()(const DistItem& a, const DistItem& b) const {
            if (a.dist != b.dist) {
                return a.dist < b.dist;
            }
            return a.id < b.id;
        }
    };
};

class ICompute {
public:
    virtual ~ICompute() = default;
    // Returns the distance between two vectors of the given type and dimension.
    virtual double dist(const uint8_t*, const uint8_t*, DataType, size_t /*dim*/) = 0;
};

} // namespace sketch2
