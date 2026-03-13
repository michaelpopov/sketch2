// Declares a compact growable bitset.

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace sketch2 {

// DynamicBitset exists to provide a small growable bitset for hot storage paths
// that need compact visibility flags. It supports resizing plus indexed get/set
// operations without pulling in a heavier container abstraction.
class DynamicBitset {
public:
    void resize(size_t size);
    size_t size() const { return bit_count_; }
    bool get(size_t index) const {
        if (index >= bit_count_) {
            throw std::out_of_range("DynamicBitset::get: index out of range");
        }

        const size_t word_index = index / kWordBits;
        const size_t bit_index = index % kWordBits;
        return (words_[word_index] & (uint64_t{1} << bit_index)) != 0;
    }
    void set(size_t index, bool value = true);

private:
    static constexpr size_t kWordBits = sizeof(uint64_t) * 8;

    void clear_unused_tail_bits();

    size_t bit_count_ = 0;
    std::vector<uint64_t> words_;
};

} // namespace sketch2
