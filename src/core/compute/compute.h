#pragma once
#include "utils/shared_types.h"
#include <cstdint>

namespace sketch2 {

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
