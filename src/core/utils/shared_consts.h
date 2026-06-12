// Defines shared constants used across storage, compute, and API layers.

#pragma once

#include <cstddef>
#include <cstdint>

namespace sketch2 {

// Current Sketch2 version
inline constexpr char kSketch2Version[] = "0.1.0";

// Dataset metadata file name shared by Dataset INI loading and dataset-management callers.
inline constexpr char kMetadataFileName[] = "sketch2.metadata";

// Default dataset id range size used when splitting vectors into per-range files.
inline constexpr uint64_t kRangeSize = 10'000;

// Default ratio used by DatasetWriter to decide when updates are large enough
// to merge into a base data file instead of remaining as deltas.
inline constexpr uint64_t kDataMergeRatio = 2;

// Default number of fractional digits used when formatting vectors as strings.
inline constexpr size_t kDefaultVectorStringDigits = 2;

// Minimum supported vector dimension enforced by Dataset/InputReader/InputGenerator validation.
inline constexpr uint64_t kMinDimension = 4;

// Maximum supported vector dimension constrained by current file format and parser APIs.
inline constexpr uint64_t kMaxDimension = 4096;

// File format magic written by DataWriter/DataMerger and validated by DataReader.
inline constexpr uint32_t kMagic = 0x534B5632; // "SKV2"

// Binary storage format version written into data headers and checked when reopening files.
// Version 13 stores aligned vector records with any optional norm inline in the record stride,
// followed by region-aligned frozen Roaring id trailers.
inline constexpr uint16_t kVersion = 13;

// Vector payload alignment used by data files and accumulator storage for SIMD-friendly access.
inline constexpr uint32_t kDataAlignment = 32;

// Region alignment used by data files for independently mmap-able sections.
inline constexpr uint64_t kDataRegionAlignment = 4096;
static_assert(kDataRegionAlignment % 32u == 0,
    "Data region alignment must satisfy frozen Roaring's 32-byte alignment requirement");

// Data-file header flags describing which float values are persisted inline in
// active vector records.
inline constexpr uint32_t kDataFileHasCosineInvNorms = 1u;
inline constexpr uint32_t kDataFileHasSquaredNorms = 2u;
inline constexpr uint32_t kDataFileNormKindMask =
    kDataFileHasCosineInvNorms | kDataFileHasSquaredNorms;

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

// Dataset owner lock suffix; the path is <first dataset dir>/<dataset name>.lock.
inline constexpr char kLockExt[] = ".lock";

// Accumulator WAL file name used by Dataset to persist owner-side pending updates across restarts.
inline constexpr char kAccumulatorWalFileName[] = "sketch2.accumulator.wal";

// SQLite virtual table module name registered by the vlite extension entry point.
inline constexpr char kVliteModuleName[] = "vlite";

// SQLite virtual table schema declared during vlite module connect/create.
inline constexpr char kVliteSchema[] =
    "CREATE TABLE x(query TEXT HIDDEN, k INTEGER HIDDEN, id INTEGER, score REAL)";

// Number of items in the input block.
inline constexpr size_t kIndexedBinaryBlockItems = sizeof(uint64_t) * 8;

// Hard upper bound on the number of named bitset filter files kept open and
// mapped by BitsetFileCache. Inserts beyond this limit are rejected; callers
// fall back to opening the file from disk on each access. Adjusted by
// sk_bitset_cache_remove / sk_bitset_cache_clear.
inline constexpr size_t kMaxLoadedBitsetCount = 128;

// Input format markers.
constexpr const char* BinFileMarker = "bin";
constexpr const char* BinIndexedFileMarker = "binind";


} // namespace sketch2
