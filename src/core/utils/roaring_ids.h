// Declares a uint64_t id container backed by CRoaring.

#pragma once

#include "roaring.hh"
#include "shared_types.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>

namespace sketch2 {

// Class RoaringIds is a wrapper/umbrella class for CRoaring
// structure that stores uint64_t values and provides access
// to these values.
class RoaringIds {
public:
    static constexpr size_t npos = std::numeric_limits<size_t>::max();

    RoaringIds() = default;
    RoaringIds(const RoaringIds&) = delete;
    RoaringIds& operator=(const RoaringIds&) = delete;
    RoaringIds(RoaringIds&&) noexcept = default;
    RoaringIds& operator=(RoaringIds&&) noexcept = default;

    // Class Iterator allows accessing values in the underlying CRoaring
    // structure sequentially.
    class Iterator {
        friend class RoaringIds;
    public:
        void next();
        bool eof() const;
        uint64_t id() const;
        size_t index() const;
        uint64_t operator*() const;
        Iterator& operator++();
        bool operator==(const Iterator& other) const;
        bool operator!=(const Iterator& other) const;

    private:
        explicit Iterator(const RoaringIds* roaring_ids);

        const RoaringIds* roaring_ids_ = nullptr;
        size_t index_ = 0;
        roaring::api::roaring_uint32_iterator_t iterator_{};
    };

    // Initialize the underlying CRoaring structure for
    // adding new values.
    Ret init_writable(uint64_t base);

    // Initialize the underlying CRoaring structure from
    // the "frozen view" buffer for the read-only access.
    Ret init_frozen_view(const uint8_t* data, size_t size, uint64_t base);

    // Add a value to underlying CRoaring structure.
    Ret add(uint64_t id);

    // Reset to an uninitialized empty state.
    void clear();

    // Number of ids in the underlying CRoaring structure.
    size_t count() const;

    // Check whether the underlying CRoaring structure contains no ids.
    bool empty() const;

    // Check whether an id is present in the underlying CRoaring structure.
    bool contains(uint64_t id) const;

    // Compact the underlying CRoaring structure for copying
    // frozen view into a buffer
    void compact();

    // Size of a buffer required for holding the frozen view
    // of the underlying CRoaring structure.
    size_t serialized_size_bytes() const;

    // Get an id at a specific position.
    uint64_t id(size_t index) const;

    // Get an id at a specific position when the caller has already validated
    // the index. "Unchecked" means this skips the explicit count() precheck;
    // CRoaring select failures and uint64_t overflow are still reported.
    uint64_t id_unchecked(size_t index) const;

    // Find the index of the value in the underlying CRoaring
    // structure.
    size_t lower_bound_index(uint64_t id) const;

    // Generate a frozen view of the underlying CRoaring instance
    // and copy it into the buffer.
    Ret serialize(char* buffer) const;

    // Get an iterator to access values in the underlying CRoaring
    // structure sequentially.
    Iterator begin() const;
    Iterator end() const;

private:
    struct BitmapDeleter {
        void operator()(roaring::api::roaring_bitmap_t* bitmap) const;
    };

    using BitmapPtr = std::unique_ptr<roaring::api::roaring_bitmap_t, BitmapDeleter>;

    roaring::api::roaring_bitmap_t* bitmap();
    const roaring::api::roaring_bitmap_t* bitmap() const;

    BitmapPtr bitmap_;
    bool read_only_ = false;
    uint64_t base_ = 0;
};

} // namespace sketch2
