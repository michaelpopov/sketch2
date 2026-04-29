#pragma once
#include "sketch2api_utils.h"

namespace sketch2 {

SKETCH2API_HIDDEN Ret sk_bitset_add_id_(
    void** state, uint64_t id, const char* name);
SKETCH2API_HIDDEN Ret sk_bitset_add_id_name_(
    void** state, uint64_t id);
SKETCH2API_HIDDEN Ret sk_bitset_add_multiple_ids_name_(
    void** state, const uint64_t* ids, uint64_t size);
SKETCH2API_HIDDEN Ret sk_bitset_create_builder_(
    void** state, const char* name);
SKETCH2API_HIDDEN Ret sk_bitset_finish_(void** state, void** out);
SKETCH2API_HIDDEN Ret sk_bitset_load_(const char* name, void** out);
SKETCH2API_HIDDEN void sk_bitset_delete_(void* ptr);
SKETCH2API_HIDDEN Ret sk_bitset_drop_(const char* name, bool* removed_out);
SKETCH2API_HIDDEN Ret sk_bitset_cache_remove_(
    const char* name, bool* removed_out);
SKETCH2API_HIDDEN Ret sk_bitset_cache_clear_();
SKETCH2API_HIDDEN int sk_bitset_storage_kind_for_testing_(const void* ptr);

} // namespace sketch2
