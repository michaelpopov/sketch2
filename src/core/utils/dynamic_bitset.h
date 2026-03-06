#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sketch2 {

class DynamicBitset {
public:
    void resize(size_t size);
    size_t size() const;
    bool get(size_t index) const;
    void set(size_t index, bool value = true);

private:
    static constexpr size_t kWordBits = sizeof(uint64_t) * 8;

    static size_t words_for_bits(size_t size);
    void clear_unused_tail_bits();

    size_t bit_count_ = 0;
    std::vector<uint64_t> words_;
};

} // namespace sketch2
