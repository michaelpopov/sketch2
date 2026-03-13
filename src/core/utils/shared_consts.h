#pragma once

#include <cstddef>
#include <cstdint>

namespace sketch2 {

// Dataset metadata file name shared by Dataset INI loading and dataset-management callers.
inline constexpr char kMetadataFileName[] = "sketch2.metadata";

// Default in-memory accumulator capacity used by Dataset metadata and scanner/storage tests.
inline constexpr size_t kAccumulatorBufferSize = 64 * 1024;

// Default dataset id range size used when splitting vectors into per-range files.
inline constexpr uint64_t kRangeSize = 10'000;

// Minimum supported vector dimension enforced by Dataset/InputReader/InputGenerator validation.
inline constexpr uint64_t kMinDimension = 4;

// Maximum supported vector dimension constrained by current file format and parser APIs.
inline constexpr uint64_t kMaxDimension = 4096;

// File format magic written by DataWriter/DataMerger and validated by DataReader/WAL replay.
inline constexpr uint32_t kMagic = 0x534B5632; // "SKV2"

// Binary storage format version written into data/WAL headers and checked when reopening files.
// Version 4 adds explicit per-record vector stride so persisted vectors stay 32-byte aligned.
inline constexpr uint16_t kVersion = 4;

// Vector payload alignment used by data files and accumulator storage for SIMD-friendly access.
inline constexpr uint32_t kDataAlignment = 32;

// Id section alignment used by data-file layout code when placing active/deleted id arrays.
inline constexpr uint32_t kIdsAlignment = 8;

// Shared stdio buffer size used by DataWriter, DataMerger, and Dataset merge/store paths.
inline constexpr size_t kFileBufferSize = 4 * 1024 * 1024;

// FNV-1a seed used by AccumulatorWal record checksums to detect torn/corrupt writes.
inline constexpr uint32_t kFnvOffsetBasis = 2166136261u;

// FNV-1a multiplier used alongside kFnvOffsetBasis in AccumulatorWal checksum calculation.
inline constexpr uint32_t kFnvPrime = 16777619u;

// Temporary output suffix used by Dataset while rebuilding a data file before rename.
inline constexpr char kTempExt[] = ".temp";

// Primary data-file suffix used by Dataset directory scans and file creation paths.
inline constexpr char kDataExt[] = ".data";

// Delta-file suffix used by Dataset for pending updates that have not been merged yet.
inline constexpr char kDeltaExt[] = ".delta";

// Merge-file suffix used by Dataset while combining base data and accumulator/delta updates.
inline constexpr char kMergeExt[] = ".merge";

// Advisory lock suffix used by Dataset for per-range file mutation serialization.
inline constexpr char kLockExt[] = ".lock";

// Dataset owner lock file name used by Dataset::ensure_owner_lock_() and parasol::sk_drop().
inline constexpr char kOwnerLockFileName[] = "sketch2.owner.lock";

// Accumulator WAL file name used by Dataset to persist owner-side pending updates across restarts.
inline constexpr char kAccumulatorWalFileName[] = "sketch2.accumulator.wal";

// Temporary text-input file name used by parasol dataset creation/import workflows.
inline constexpr char kInputFileName[] = "data.input";

// SQLite virtual table module name registered by the vlite extension entry point.
inline constexpr char kVliteModuleName[] = "vlite";

// SQLite virtual table schema declared during vlite module connect/create.
inline constexpr char kVliteSchema[] =
    "CREATE TABLE x(query TEXT HIDDEN, k INTEGER HIDDEN, id INTEGER, distance REAL)";

} // namespace sketch2
