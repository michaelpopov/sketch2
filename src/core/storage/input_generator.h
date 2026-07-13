// Declares input-generation helpers and patterns for synthetic datasets.

#pragma once
#include "utils/float8.h"
#include "utils/shared_types.h"

#include <algorithm>
#include <map>
#include <vector>
#include <optional>

namespace sketch2 {

struct ManualInputGenerator {
    void add(uint64_t id, int val) { items[id] = val; }
    void deleted(uint64_t id) { items[id] = std::optional<int>(); }

    DataType type = DataType::i16;
    size_t dim = 16;
    // std::map provides deterministic sorted-id iteration. For f8, live
    // entries receive consecutive base-72 ordinals in this order; tombstones
    // are emitted but do not consume an ordinal.
    std::map<uint64_t, std::optional<int>> items;
};

enum class PatternType {
    Sequential,
    Detailed,
    CosCompatible,
    DotCompatible,
    PerfTest,
};

struct GeneratorConfig {
    PatternType pattern_type;
    size_t count;
    size_t min_id;
    DataType type;
    size_t dim;
    size_t max_val;
    size_t every_n_deleted = 0;
    bool binary = false;
};

Ret generate_input_file(const std::string& path, const GeneratorConfig& config);
Ret generate_input_file(const std::string& path, const ManualInputGenerator& gen);
Ret generate_dummy_metadata(const std::string& path, size_t count, size_t start_id = 0);

// InputVector exists to generate deterministic per-dimension test vectors for
// synthetic input files. It advances one column at a time so tests can produce
// predictable sequences without hand-writing every vector.
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

// f8 has no arithmetic operators by design.  Detailed f8 generation instead
// walks the canonical finite codebook.  max_val is an inclusive upper bound on
// the sorted codebook prefix; values outside [-28, 28] clamp to an endpoint.
// The progression mirrors the existing InputVector behavior: a column reaches
// its upper entry before the next column begins, and the next call after the
// final column completes resets every component to the first codebook value.
template <>
class InputVector<float8> {
public:
    InputVector(size_t dim, float max_val)
        : max_codebook_index_(float8_codebook::upper_bound_index(max_val)),
          vec_(dim, float8_codebook::value_at(0)),
          codebook_indices_(dim, 0) {}

    InputVector(size_t dim, float8 max_val)
        : InputVector(dim, static_cast<float>(max_val)) {}

    const float8* data() const { return vec_.data(); }

    void next() {
        if (col_ >= vec_.size()) {
            std::fill(vec_.begin(), vec_.end(), float8_codebook::value_at(0));
            std::fill(codebook_indices_.begin(), codebook_indices_.end(), 0);
            col_ = 0;
            return;
        }

        if (codebook_indices_[col_] < max_codebook_index_) {
            ++codebook_indices_[col_];
            vec_[col_] = float8_codebook::value_at(codebook_indices_[col_]);
        }
        if (codebook_indices_[col_] == max_codebook_index_) {
            ++col_;
        }
    }

private:
    const size_t max_codebook_index_;
    std::vector<float8> vec_;
    std::vector<size_t> codebook_indices_;
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
