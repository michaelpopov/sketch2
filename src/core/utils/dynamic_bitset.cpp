#include "dynamic_bitset.h"

#include <stdexcept>

namespace sketch2 {

size_t DynamicBitset::words_for_bits(size_t size) {
    return (size + kWordBits - 1) / kWordBits;
}

void DynamicBitset::clear_unused_tail_bits() {
    if (words_.empty()) {
        return;
    }

    const size_t used_bits_in_tail = bit_count_ % kWordBits;
    if (used_bits_in_tail == 0) {
        return;
    }

    const uint64_t mask = (uint64_t{1} << used_bits_in_tail) - 1;
    words_.back() &= mask;
}

void DynamicBitset::resize(size_t size) {
    bit_count_ = size;
    words_.resize(words_for_bits(size));
    clear_unused_tail_bits();
}

size_t DynamicBitset::size() const {
    return bit_count_;
}

bool DynamicBitset::get(size_t index) const {
    if (index >= bit_count_) {
        throw std::out_of_range("DynamicBitset::get: index out of range");
    }

    const size_t word_index = index / kWordBits;
    const size_t bit_index = index % kWordBits;
    return (words_[word_index] & (uint64_t{1} << bit_index)) != 0;
}

void DynamicBitset::set(size_t index, bool value) {
    if (index >= bit_count_) {
        throw std::out_of_range("DynamicBitset::set: index out of range");
    }

    const size_t word_index = index / kWordBits;
    const size_t bit_index = index % kWordBits;
    const uint64_t mask = uint64_t{1} << bit_index;

    if (value) {
        words_[word_index] |= mask;
    } else {
        words_[word_index] &= ~mask;
    }
}

} // namespace sketch2
