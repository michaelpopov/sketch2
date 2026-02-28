#pragma once
#include "utils/shared_types.h"
#include "core/compute/compute.h"
#include <cstdint>
#include <string>
#include <vector>

namespace sketch2 {

class DataReader;
class Dataset;

class Scanner {
public:
    // Returns up to count vector ids nearest to vec, ordered by distance.
    // func selects the distance function.
    // vec must match the type and dimension of the file.
    Ret find(const DataReader& reader, DistFunc func, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) const;

    Ret find(const Dataset& dataset, DistFunc func, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) const;

private:
    Ret find_(const DataReader& reader, DistFunc func, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) const;
    Ret find_(const Dataset& dataset, DistFunc func, size_t count, const uint8_t* vec,
        std::vector<uint64_t>& result) const;
};

} // namespace sketch2
