#pragma once
#include "utils/shared_types.h"
#include <vector>

namespace sketch2 {

enum class PatternType {
    Sequential,
    Detailed,
    Random,
};

struct GeneratorConfig {
    PatternType pattern_type;
    size_t count;
    size_t min_id;
    DataType type;
    size_t dim;
    size_t max_val;
};

Ret generate_input_file(const std::string& path, const GeneratorConfig& config);

template <typename T>
class InputVector {
public:
    InputVector(size_t dim, T max_val) : max_val_(max_val) { vec_.resize(dim); }
    const T* data() const { return vec_.data(); }
    void next() {
        // Handle overflow
        if (col_ >= vec_.size()) {
            col_ = 0;
            for (size_t i = 0; i < vec_.size(); i++) {
                vec_[col_] = static_cast<T>(0);
            }
            return;
        }

        vec_[col_] += 0.01;

        if (vec_[col_] == max_val_) {
            col_++;
        }
    }

private:
    const T max_val_;
    std::vector<T> vec_;
    size_t col_ = 0;
};

template <>
inline void InputVector<int16_t>::next() {
    // Handle overflow
    if (col_ >= vec_.size()) {
        col_ = 0;
        for (size_t i = 0; i < vec_.size(); i++) {
            vec_[col_] = 0;
        }
        return;
    }

    vec_[col_]++;
    if (vec_[col_] == max_val_) {
        col_++;
    }
}

} // namespace sketch2
