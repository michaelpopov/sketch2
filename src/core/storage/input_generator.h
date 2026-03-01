#pragma once
#include "utils/shared_types.h"
#include <map>
#include <vector>
#include <optional>

namespace sketch2 {

struct ManualInputGenerator {
    void add(uint64_t id, int val) { items[id] = val; }
    void deleted(uint64_t id) { items[id] = std::optional<int>(); }

    DataType type = DataType::i16;
    size_t dim = 16;
    std::map<uint64_t, std::optional<int>> items;
};

enum class PatternType {
    Sequential,
    Detailed,
};

struct GeneratorConfig {
    PatternType pattern_type;
    size_t count;
    size_t min_id;
    DataType type;
    size_t dim;
    size_t max_val;
    size_t every_n_deleted = 0;
};

Ret generate_input_file(const std::string& path, const GeneratorConfig& config);
Ret generate_input_file(const std::string& path, const ManualInputGenerator& gen);

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
                vec_[i] = static_cast<T>(0);
            }
            return;
        }

        const T increment = static_cast<T>(0.01);
        vec_[col_] += increment;

        if (vec_[col_] >= max_val_) {
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
            vec_[i] = 0;
        }
        return;
    }

    vec_[col_]++;
    if (vec_[col_] == max_val_) {
        col_++;
    }
}

} // namespace sketch2
