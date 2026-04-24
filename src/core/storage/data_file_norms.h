// Predicates and conversions for the inline-norm flags stored in DataFileHeader.

#pragma once
#include "core/storage/data_file.h"
#include "utils/shared_types.h"
#include <cstdint>
#include <stdexcept>

namespace sketch2 {

inline uint32_t data_file_norm_flags(const DataFileHeader& hdr) {
    return hdr.flags & kDataFileNormKindMask;
}

inline bool data_file_has_norms(const DataFileHeader& hdr) {
    return data_file_norm_flags(hdr) != 0u;
}

inline bool data_file_has_cosine_inv_norms(const DataFileHeader& hdr) {
    return data_file_norm_flags(hdr) == kDataFileHasCosineInvNorms;
}

inline bool data_file_has_squared_norms(const DataFileHeader& hdr) {
    return data_file_norm_flags(hdr) == kDataFileHasSquaredNorms;
}

inline uint32_t data_file_norm_flags_for_dist(DistFunc dist_func) {
    switch (dist_func) {
        case DistFunc::COS:
            return kDataFileHasCosineInvNorms;
        case DistFunc::L2:
            return kDataFileHasSquaredNorms;
        case DistFunc::DOT:
            return 0u;
        default:
            throw std::runtime_error("data_file_norm_flags_for_dist: unsupported distance function");
    }
}

inline bool data_file_has_valid_norm_flags(const DataFileHeader& hdr) {
    const uint32_t norm_flags = data_file_norm_flags(hdr);
    return norm_flags == 0u
        || norm_flags == kDataFileHasCosineInvNorms
        || norm_flags == kDataFileHasSquaredNorms;
}

inline bool data_file_matches_stored_norms_dist(const DataFileHeader& hdr, DistFunc dist_func) {
    return data_file_norm_flags(hdr) == data_file_norm_flags_for_dist(dist_func);
}

inline DistFunc dist_func_for_data_file_norm_flags(uint32_t norm_flags) {
    switch (norm_flags) {
        case 0u:
            return DistFunc::DOT;
        case kDataFileHasCosineInvNorms:
            return DistFunc::COS;
        case kDataFileHasSquaredNorms:
            return DistFunc::L2;
        default:
            throw std::runtime_error("dist_func_for_data_file_norm_flags: invalid norm flags");
    }
}

inline bool dataset_requires_stored_norms(DistFunc dist_func) {
    return dist_func == DistFunc::COS || dist_func == DistFunc::L2;
}

} // namespace sketch2
