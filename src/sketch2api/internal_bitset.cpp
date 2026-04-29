#include "internal_bitset.h"

#include "core/bitset/bitset_file_cache.h"
#include "core/bitset/bitset_filter_control.h"
#include "core/bitset/chunked_bits.h"

#include <memory>

namespace sketch2 {

namespace {

Ret create_bitset_filter_builder_(void** state, const char* name) {
    if (state == nullptr) {
        return Ret("bitset filter builder: invalid builder state");
    }

    auto* chunked_bits = static_cast<ChunkedBits*>(*state);
    if (chunked_bits == nullptr) {
        auto new_bits = std::make_unique<ChunkedBits>();
        chunked_bits = new_bits.get();
        if (chunked_bits->set_name(name).code() != 0) {
            return Ret("sketch2: invalid bitset filter name");
        }
        new_bits.release();
        *state = chunked_bits;
        return Ret(0);
    }

    const char* requested_name = name != nullptr ? name : "";
    if (chunked_bits->name() != requested_name) {
        // SQLite-style aggregator bindings call xFinal (i.e. finish) even after
        // xStep returns an error. Destroying the builder here ensures finish()
        // sees a null state and produces an empty control with no name, so a
        // mid-stream error never publishes a partially-built named filter file.
        delete chunked_bits;
        *state = nullptr;
        return Ret("sketch2: inconsistent bitset filter name");
    }
    return Ret(0);
}

} // namespace

Ret sk_bitset_add_id_(void** state, uint64_t id, const char* name) {
    CHECK(create_bitset_filter_builder_(state, name));
    auto* chunked_bits = static_cast<ChunkedBits*>(*state);
    if (chunked_bits->add(id).code() != 0) {
        return Ret("bitset filter builder: add failed");
    }
    return Ret(0);
}

Ret sk_bitset_add_id_name_(void** state, uint64_t id) {
    if (state == nullptr || *state == nullptr) {
        return Ret("bitset filter builder: invalid builder state");
    }
    auto* chunked_bits = static_cast<ChunkedBits*>(*state);
    if (chunked_bits->add(id).code() != 0) {
        return Ret("bitset filter builder: add failed");
    }
    return Ret(0);
}

// Caller contract: ids must be sorted in non-decreasing order. ChunkedBits::add
// validates this and returns an error otherwise; the sorted invariant is what
// lets the per-chunk slices skip the buffered-builder sort and use CRoaring's
// range-coalescing bulk path directly.
Ret sk_bitset_add_multiple_ids_name_(
        void** state, const uint64_t* ids, uint64_t size) {
    if (state == nullptr || *state == nullptr) {
        return Ret("bitset filter builder: invalid builder state");
    }
    if (size == 0) {
        return Ret("bitset filter builder: ids size must be greater than zero");
    }
    if (ids == nullptr) {
        return Ret("bitset filter builder: ids pointer is null");
    }

    auto* chunked_bits = static_cast<ChunkedBits*>(*state);
    CHECK(chunked_bits->add(ids, size));
    return Ret(0);
}

Ret sk_bitset_create_builder_(void** state, const char* name) {
    return create_bitset_filter_builder_(state, name);
}

Ret sk_bitset_finish_(void** state, void** out) {
    *out = nullptr;

    std::unique_ptr<ChunkedBits> bits(static_cast<ChunkedBits*>(*state));
    *state = nullptr;

    BitsetFilterControlPtr control;
    const Ret control_ret = bits != nullptr
        ? BitsetFilterControl::create(*bits, &control)
        : BitsetFilterControl::create_empty(&control);
    if (control_ret.code() != 0) {
        return control_ret;
    }
    *out = control.release();
    return Ret(0);
}

Ret sk_bitset_load_(const char* name, void** out) {
    BitsetFilterControlPtr control;
    CHECK(load_named_bitset_filter(name, &control));
    *out = control.release();
    return Ret(0);
}

void sk_bitset_delete_(void* ptr) {
    delete static_cast<BitsetFilterControl*>(ptr);
}

Ret sk_bitset_drop_(const char* name, bool* removed_out) {
    return drop_named_bitset_filter(name, removed_out);
}

Ret sk_bitset_cache_remove_(const char* name, bool* removed_out) {
    if (validate_chunked_bits_name(name).code() != 0) {
        return Ret("bitset_cache_remove: invalid bitset filter name");
    }
    *removed_out = bitset_file_cache().remove(name);
    return Ret(0);
}

Ret sk_bitset_cache_clear_() {
    bitset_file_cache().clear();
    return Ret(0);
}

int sk_bitset_storage_kind_for_testing_(const void* ptr) {
    const auto* control = static_cast<const BitsetFilterControl*>(ptr);
    switch (bitset_filter_storage_kind_for_testing(control)) {
        case BitsetFilterStorageKind::Heap:
            return 0;
        case BitsetFilterStorageKind::MappedFile:
            return 1;
        case BitsetFilterStorageKind::MappedFileTemporary:
            return 2;
    }
    return -1;
}

} // namespace sketch2
