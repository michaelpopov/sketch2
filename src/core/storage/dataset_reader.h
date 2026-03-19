#pragma once
#include "dataset.h"

namespace sketch2 {

class DatasetReader : public Dataset {
public:
    using Dataset::Dataset;
    ~DatasetReader() override = default;
};

} // namespace sketch2
