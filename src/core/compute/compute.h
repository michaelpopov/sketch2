#pragma once
#include "utils/shared_types.h"
#include <cstdint>

namespace sketch2 {

enum class DistFunc {
    L1,
    L2,
};

class ICompute {
public:
    virtual ~ICompute() = default;
    // Returns the distance between two vectors of the given type and dimension.
    virtual double dist(const uint8_t* a, const uint8_t* b, DataType type, size_t dim) = 0;
};

} // namespace sketch2
