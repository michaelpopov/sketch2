#pragma once
#include "utils/shared_types.h"
#include "core/compute/compute.h"
#include "core/storage/data_reader.h"
#include <cstdint>
#include <string>
#include <vector>

namespace sketch2 {

class Scanner {
public:
    Ret init(const std::string& path);

    // Returns up to count vector ids nearest to vec, ordered by distance.
    // func selects the distance function.
    // vec must match the type and dimension of the file.
    std::vector<uint64_t> find(DistFunc func, size_t count, const uint8_t* vec) const;

private:
    DataReader reader_;
};

} // namespace sketch2
