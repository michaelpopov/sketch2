// Centralizes architecture detection and SIMD backend feature flags.

#pragma once

// --- Architecture Detection ---

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define SKETCH_ARCH_X86 1
#else
#define SKETCH_ARCH_X86 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define SKETCH_ARCH_ARM64 1
#else
#define SKETCH_ARCH_ARM64 0
#endif

// --- Backend Feature Flags ---

#if SKETCH_ARCH_X86
#if defined(SKETCH_ENABLE_AVX2) && SKETCH_ENABLE_AVX2
#define SKETCH_HAS_AVX2 1
#else
#define SKETCH_HAS_AVX2 0
#endif

#if defined(SKETCH_ENABLE_AVX512F) && SKETCH_ENABLE_AVX512F
#define SKETCH_HAS_AVX512F 1
#else
#define SKETCH_HAS_AVX512F 0
#endif

#if defined(SKETCH_ENABLE_AVX512VNNI) && SKETCH_ENABLE_AVX512VNNI
#define SKETCH_HAS_AVX512VNNI 1
#else
#define SKETCH_HAS_AVX512VNNI 0
#endif

#define SKETCH_HAS_AVX512 (SKETCH_HAS_AVX512F || SKETCH_HAS_AVX512VNNI)
#else
#define SKETCH_HAS_AVX2 0
#define SKETCH_HAS_AVX512F 0
#define SKETCH_HAS_AVX512VNNI 0
#define SKETCH_HAS_AVX512 0
#endif

#if SKETCH_ARCH_ARM64
#define SKETCH_HAS_NEON 1
#else
#define SKETCH_HAS_NEON 0
#endif

// --- Target Attributes ---

#if SKETCH_ARCH_X86 && (defined(__GNUC__) || defined(__clang__))
#define SKETCH_AVX2_TARGET __attribute__((target("avx2,f16c,fma")))
#define SKETCH_AVX512F_TARGET __attribute__((target("avx512f")))
#define SKETCH_AVX512VNNI_TARGET __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
#else
#define SKETCH_AVX2_TARGET
#define SKETCH_AVX512F_TARGET
#define SKETCH_AVX512VNNI_TARGET
#endif
