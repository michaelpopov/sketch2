// Shared scanner logging helpers.

#pragma once

#include "core/compute/dist_item.h"
#include "core/storage/data_reader.h"
#include "core/utils/log.h"

#include <atomic>
#include <vector>

namespace sketch2 {

inline const char* scanner_dist_func_name(DistFunc func) {
    switch (func) {
        case DistFunc::DOT: return "DOT";
        case DistFunc::L2: return "L2";
        case DistFunc::COS: return "COS";
        default: return "unknown";
    }
}

inline uint64_t next_scanner_query_id() {
    static std::atomic<uint64_t> next_id{1};
    return next_id.fetch_add(1, std::memory_order_relaxed);
}

inline const char* scanner_query_path_name(const char* path) {
    return path ? path : "unknown";
}

inline void log_query_start(uint64_t query_id, const std::string& source, DistFunc func, DataType type,
        size_t dim, size_t count, bool has_bitset) {
    LOG_TRACE << "Scanner query start: query_id=" << query_id
             << " source=" << source
             << " metric=" << scanner_dist_func_name(func)
             << " type=" << data_type_to_string(type)
             << " dim=" << dim
             << " k=" << count
             << " bitset=" << (has_bitset ? "yes" : "no");
}

inline void log_query_branch(uint64_t query_id, const char* path, double query_norm_sq = -1.0,
        bool query_norm_is_zero = false) {
    LOG_TRACE << "Scanner query path: query_id=" << query_id
              << " path=" << scanner_query_path_name(path)
              << " query_norm_sq=" << query_norm_sq
              << " query_norm_zero=" << (query_norm_is_zero ? "yes" : "no");
}

inline void log_query_finish(uint64_t query_id, const std::string& source, DistFunc func, DataType type,
        size_t dim, size_t count, int64_t elapsed_ms,
        const std::vector<DistItem>& result) {
    LOG_TRACE << "Scanner query finish: query_id=" << query_id
             << " source=" << source
             << " metric=" << scanner_dist_func_name(func)
             << " type=" << data_type_to_string(type)
             << " dim=" << dim
             << " k=" << count
             << " hits=" << result.size()
             << " best_score=" << (result.empty() ? 0.0 : result.front().score)
             << " worst_score=" << (result.empty() ? 0.0 : result.back().score)
             << " time=" << elapsed_ms << " ms";
}

inline void log_collected_dataset_readers(uint64_t query_id, const std::string& source, size_t reader_count) {
    LOG_TRACE << "Scanner query readers: query_id=" << query_id
              << " source=" << source
              << " readers=" << reader_count;
}

inline void log_dataset_scan_mode(uint64_t query_id, size_t reader_count, bool has_thread_pool,
        bool uses_parallel, bool has_bitset) {
    LOG_TRACE << "Scanner scan mode: query_id=" << query_id
              << " mode=" << (uses_parallel ? "parallel" : "serial")
              << " readers=" << reader_count
              << " thread_pool=" << (has_thread_pool ? "yes" : "no")
              << " bitset=" << (has_bitset ? "yes" : "no");
}

inline void log_reader_scan_plan(uint64_t query_id, const DataReader& reader, const char* path,
        bool uses_stored_norms, bool has_bitset) {
    LOG_TRACE << "Scanner reader path: query_id=" << query_id
              << " reader=" << reader.path()
              << " path=" << scanner_query_path_name(path)
              << " stored_norms=" << (uses_stored_norms ? "yes" : "no")
              << " rows=" << reader.count()
              << " deleted=" << reader.deleted_count()
              << " has_delta=" << (reader.has_delta() ? "yes" : "no")
              << " bitset=" << (has_bitset ? "yes" : "no");
}

inline void log_parallel_reader_merge(uint64_t query_id, size_t reader_count, size_t merged_candidates,
        size_t final_heap_size) {
    LOG_TRACE << "Scanner merge: query_id=" << query_id
              << " readers=" << reader_count
              << " merged_candidates=" << merged_candidates
              << " final_heap_size=" << final_heap_size;
}

} // namespace sketch2
