// Declares the shared bitset filter used by scanner APIs.

#pragma once

namespace sketch2 {

class ChunkedBitsView;

struct BitsetFilter {
    const ChunkedBitsView* view = nullptr;
};

} // namespace sketch2
