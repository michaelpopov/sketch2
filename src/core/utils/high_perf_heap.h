// Defines a vector-backed heap with fused root replacement operations.

#pragma once

#include <cassert>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

namespace sketch2 {

template <typename T, typename Compare = std::less<T>>
class HighPerfHeap {
public:
    using value_type = T;
    using container_type = std::vector<value_type>;
    using value_compare = Compare;
    using size_type = typename container_type::size_type;
    using const_reference = const value_type&;

    HighPerfHeap() = default;

    explicit HighPerfHeap(value_compare compare)
        : compare_(std::move(compare)) {}

    HighPerfHeap(size_type reserve_capacity, value_compare compare = value_compare())
        : compare_(std::move(compare)) {
        data_.reserve(reserve_capacity);
    }

    HighPerfHeap(container_type&& data, value_compare compare = value_compare())
        : data_(std::move(data)), compare_(std::move(compare)) {
        heapify();
    }

    template <typename InputIt>
    HighPerfHeap(InputIt first, InputIt last, value_compare compare = value_compare())
        : data_(first, last), compare_(std::move(compare)) {
        heapify();
    }

    [[nodiscard]] bool empty() const noexcept {
        return data_.empty();
    }

    [[nodiscard]] size_type size() const noexcept {
        return data_.size();
    }

    [[nodiscard]] size_type capacity() const noexcept {
        return data_.capacity();
    }

    [[nodiscard]] const_reference top() const {
        assert(!data_.empty());
        return data_.front();
    }

    [[nodiscard]] const container_type& data() const noexcept {
        return data_;
    }

    [[nodiscard]] const value_compare& comparator() const noexcept {
        return compare_;
    }

    void clear() noexcept {
        data_.clear();
    }

    void reserve(size_type new_capacity) {
        data_.reserve(new_capacity);
    }

    void reset(container_type&& data) {
        data_ = std::move(data);
        heapify();
    }

    void push(const value_type& value) {
        data_.push_back(value);
        sift_up_from(data_.size() - 1);
    }

    void push(value_type&& value) {
        data_.push_back(std::move(value));
        sift_up_from(data_.size() - 1);
    }

    template <typename... Args>
    void emplace(Args&&... args) {
        data_.emplace_back(std::forward<Args>(args)...);
        sift_up_from(data_.size() - 1);
    }

    void pop() {
        assert(!data_.empty());
        if (data_.size() == 1) {
            data_.pop_back();
            return;
        }

        value_type last = std::move(data_.back());
        data_.pop_back();
        sift_down_with_value(0, std::move(last));
    }

    [[nodiscard]] value_type pop_top() {
        assert(!data_.empty());

        value_type top_value = std::move(data_.front());
        if (data_.size() == 1) {
            data_.pop_back();
            return top_value;
        }

        value_type last = std::move(data_.back());
        data_.pop_back();
        sift_down_with_value(0, std::move(last));
        return top_value;
    }

    template <typename U>
    void replace_top(U&& value) {
        assert(!data_.empty());
        sift_down_with_value(0, std::forward<U>(value));
    }

    template <typename U>
    bool replace_top_if(U&& value) {
        if (data_.empty() || !compare_(value, data_.front())) {
            return false;
        }

        sift_down_with_value(0, std::forward<U>(value));
        return true;
    }

    template <typename U>
    bool push_or_replace_top(U&& value, size_type max_size) {
        if (max_size == 0) {
            return false;
        }

        if (data_.size() < max_size) {
            data_.push_back(std::forward<U>(value));
            sift_up_from(data_.size() - 1);
            return true;
        }

        if (!compare_(value, data_.front())) {
            return false;
        }

        sift_down_with_value(0, std::forward<U>(value));
        return true;
    }

private:
    container_type data_;
    value_compare compare_;

    void heapify() {
        if (data_.size() < 2) {
            return;
        }

        for (size_type i = data_.size() / 2; i > 0; --i) {
            sift_down_from(i - 1);
        }
    }

    void sift_up_from(size_type index) {
        if (index == 0) {
            return;
        }
        value_type value = std::move(data_[index]);
        while (index > 0) {
            const size_type parent = (index - 1) / 2;
            if (!compare_(data_[parent], value)) {
                break;
            }

            data_[index] = std::move(data_[parent]);
            index = parent;
        }
        data_[index] = std::move(value);
    }

    void sift_down_from(size_type index) {
        sift_down_with_value(index, std::move(data_[index]));
    }

    template <typename U>
    void sift_down_with_value(size_type index, U&& value) {
        value_type held(std::forward<U>(value));
        const size_type heap_size = data_.size();
        size_type child = index * 2 + 1;

        while (child < heap_size) {
            const size_type right = child + 1;
            if (right < heap_size && compare_(data_[child], data_[right])) {
                child = right;
            }

            if (!compare_(held, data_[child])) {
                break;
            }

            data_[index] = std::move(data_[child]);
            index = child;
            child = index * 2 + 1;
        }

        data_[index] = std::move(held);
    }
};

} // namespace sketch2
