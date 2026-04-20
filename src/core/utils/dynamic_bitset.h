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
    bool empty() const { return bit_count_ == 0; }
    bool get(size_t index) const {
        if (index >= bit_count_) {
            throw std::out_of_range("DynamicBitset::get: index out of range");
        }

        return get_unchecked(index);
    }
    bool get_unchecked(size_t index) const {
        const size_t word_index = index / kWordBits;
        const size_t bit_index = index % kWordBits;
        return (words_[word_index] & (uint64_t{1} << bit_index)) != 0;
    }
    bool any() const {
        for (uint64_t word : words_) {
            if (word != 0) {
                return true;
            }
        }
        return false;
    }
    size_t find_next_unset(size_t index) const {
        if (index >= bit_count_) {
            return bit_count_;
        }

        size_t word_index = index / kWordBits;
        uint64_t clear_bits = ~words_[word_index];
        clear_bits &= ~((uint64_t{1} << (index % kWordBits)) - 1);

        while (true) {
            if (clear_bits != 0) {
                const size_t candidate =
                    word_index * kWordBits + static_cast<size_t>(__builtin_ctzll(clear_bits));
                return candidate < bit_count_ ? candidate : bit_count_;
            }

            ++word_index;
            if (word_index >= words_.size()) {
                return bit_count_;
            }
            clear_bits = ~words_[word_index];
        }
    }
    void set(size_t index, bool value = true);
    void append(uint64_t word) {
        words_.push_back(word);
        bit_count_ = words_.size() * kWordBits;
    }

private:
    static constexpr size_t kWordBits = sizeof(uint64_t) * 8;

    void clear_unused_tail_bits();

    size_t bit_count_ = 0;
    std::vector<uint64_t> words_;
};

} // namespace sketch2
