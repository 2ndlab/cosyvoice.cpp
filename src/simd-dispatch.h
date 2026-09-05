#pragma once

// ---------------------------------------------------------------------------
// Pure SIMD dispatch: capability struct + presets + runtime tier selection.
//
// This header is intentionally free of intrinsic types and SIMD helpers (no
// <immintrin.h>, no __m*/*, no simd-math.h): caller TUs only need the dispatch
// logic and must stay unpolluted. All SIMD implementations live in tier-only
// headers -- simd-math.h (log/sincos) and simd-kernels-impl.h (the kernel
// bodies + their support helpers) -- included solely by the tier objects.
//
// Build-time SIMD inclusion knobs. On x86 each class is optional; on non-x86
// the SSE4.2+FMA class is emulated via SIMDe if present, and the build
// auto-falls back to COSYVOICE_NO_SIMD (scalar-only) when SIMDe is missing.
//   COSYVOICE_NO_SIMD        force scalar-only: no SIMD code at all.
//   COSYVOICE_HAS_SCALAR     compile the pure-scalar fallback tier (default ON).
//   COSYVOICE_HAS_SSE42/AVX/AVX2/AVX512  include that class (default ON).
// Disabling a lower class automatically disables every class that requires it.
// These macros are load-bearing: the kernel bodies gate every AVX/AVX-512
// intrinsic block with `#if defined(COSYVOICE_HAS_*)`, so a platform without
// the intrinsics never parses them (no <immintrin.h>/SIMDe declarations
// needed). The gates are NOT simply per-class: a lower-class block also IS the
// vector tail (or even the main loop) of higher presets that fall through to
// it -- `if constexpr (C.avx)` covers the avx2 preset too, and `C.sse42`
// covers avx/avx2 -- so those blocks read
// `HAS_SSE42 || HAS_AVX || HAS_AVX2` / `HAS_AVX || HAS_AVX2` / `HAS_AVX2`.
// The 512 block stays a lone `HAS_AVX512`: every 512 branch masks its own
// tail and returns, so nothing below it ever sees the 512 preset.
// The AVX-512 tier requires F+BW+DQ+VL together (16-bit tail masks need BW's
// kmovw; sincos' _mm512_test_epi32_mask needs DQ's vptestmd; the compilers
// themselves emit EVEX xmm16+/ymm16+ spill forms that need VL). A part with
// only some of them (e.g. Knights Landing: F alone) falls through to AVX2.
// Every kernel's 512 branch is `if constexpr (C.avx512) {...return;} else
// {...}`, so lower-tier code is not even instantiated for the 512 preset.
// ---------------------------------------------------------------------------

#include <stdexcept>
#include <utility>

struct simd_caps
{
    bool sse42    :1 = false;
    bool avx      :1 = false;
    bool fma3     :1 = false;
    bool avx2     :1 = false;
    bool avx512   :1 = false;   // every AVX-512 sub-set this build requires
                                //  (F+BW+DQ + OS state saves) is usable; that
                                //  and only that enables the 512 tier.

    bool operator==(const simd_caps&) const = default;
};

constexpr simd_caps simd_none       { .sse42=false, .avx=false, .fma3=false, .avx2=false, .avx512=false };
constexpr simd_caps simd_sse42      { .sse42=true,  .avx=false, .fma3=false, .avx2=false, .avx512=false };
constexpr simd_caps simd_sse42_fma  { .sse42=true,  .avx=false, .fma3=true,  .avx2=false, .avx512=false };
constexpr simd_caps simd_avx        { .sse42=true,  .avx=true,  .fma3=false, .avx2=false, .avx512=false };
constexpr simd_caps simd_avx2       { .sse42=true,  .avx=true,  .fma3=true,  .avx2=true,  .avx512=false };
constexpr simd_caps simd_avx512     { .sse42=true,  .avx=true,  .fma3=true,  .avx2=true,  .avx512=true  };

// g_simd_caps is defined by simd_detect.cpp, which is built only on x86 (and
// only when SIMD is not disabled). The extern is declared under the same
// conditions so non-x86 callers never reference a missing symbol.
#if !defined(COSYVOICE_NO_SIMD) && (defined(__x86_64__) || defined(_M_X64))
extern const simd_caps g_simd_caps;
#endif

template<template<simd_caps> class Kernel, typename... Args>
inline
auto simd_dispatch(Args&&... args)
{
#ifdef COSYVOICE_NO_SIMD
    return Kernel<simd_none>::run(std::forward<Args>(args)...);
#elif !defined(__x86_64__) && !defined(_M_X64)
    // non-x86: SSE4.2+FMA3 emulated via SIMDe/NEON (in the sse42 tier object).
    return Kernel<simd_sse42_fma>::run(std::forward<Args>(args)...);
#else
    // x86: dispatch by detected features, most capable first. No exact-equality
    // matching: a class the build disabled is simply never selected and the CPU
    // falls through to the next lower tier that the build did include.
#if defined(COSYVOICE_HAS_AVX512)
    if (g_simd_caps.avx512)
        return Kernel<simd_avx512>::run(std::forward<Args>(args)...);
#endif
#if defined(COSYVOICE_HAS_AVX2)
    if (g_simd_caps.avx2 && g_simd_caps.fma3 && g_simd_caps.sse42)
        return Kernel<simd_avx2>::run(std::forward<Args>(args)...);
#endif
#if defined(COSYVOICE_HAS_AVX)
    if (g_simd_caps.avx && g_simd_caps.sse42)
        return Kernel<simd_avx>::run(std::forward<Args>(args)...);
#endif
#if defined(COSYVOICE_HAS_SSE42)
    if (g_simd_caps.sse42)
        return Kernel<simd_sse42>::run(std::forward<Args>(args)...);
#endif
#if defined(COSYVOICE_HAS_SCALAR)
    return Kernel<simd_none>::run(std::forward<Args>(args)...);
#endif
    throw std::runtime_error("no matching SIMD tier for this CPU");
#endif
}