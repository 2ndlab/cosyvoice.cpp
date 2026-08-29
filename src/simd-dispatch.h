#pragma once

#include "simd-math.h"

#include <cstddef>
#include <utility>

#ifndef COSYVOICE_NO_SIMD
    #if defined(__x86_64__) || defined(_M_X64)
        #if defined(_MSC_VER)
            #include <intrin.h>
        #else
            #include <cpuid.h>
        #endif
    #endif
#endif

struct simd_caps
{
    bool sse42:1 = false;
    bool avx  :1 = false;
    bool fma3 :1 = false;
    bool avx2 :1 = false;

    bool operator==(const simd_caps&) const = default;
};

constexpr simd_caps simd_none      { .sse42=false, .avx=false, .fma3=false, .avx2=false };
constexpr simd_caps simd_sse42     { .sse42=true,  .avx=false, .fma3=false, .avx2=false };
constexpr simd_caps simd_sse42_fma { .sse42=true,  .avx=false, .fma3=true,  .avx2=false };
constexpr simd_caps simd_avx       { .sse42=true,  .avx=true,  .fma3=false, .avx2=false };
constexpr simd_caps simd_avx2      { .sse42=true,  .avx=true,  .fma3=true,  .avx2=true  };

#ifndef COSYVOICE_NO_SIMD

inline
simd_caps simd_detect()
{
#   if defined(__x86_64__) || defined(_M_X64)
    simd_caps caps;

    #if defined(_MSC_VER)
    int regs[4] = { 0, 0, 0, 0 };
    __cpuidex(regs, 1, 0);
    const unsigned ecx1 = static_cast<unsigned>(regs[2]);
    #else
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx))
        return caps;
    const unsigned ecx1 = ecx;
    #endif

    caps.sse42 = (ecx1 & (1u << 19)) != 0;

    const bool osxsave = (ecx1 & (1u << 27)) != 0;
    const bool avx_bit = (ecx1 & (1u << 28)) != 0;
    if (!osxsave || !avx_bit)
        return caps;

        #if defined(_MSC_VER)
    const unsigned xcr0 = static_cast<unsigned>(_xgetbv(0));
        #else
    unsigned xcr0_lo = 0, xcr0_hi = 0;
    __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
    const unsigned xcr0 = xcr0_lo;
        #endif
    if ((xcr0 & 0x6) != 0x6)
        return caps;

    caps.avx  = true;
    caps.fma3 = (ecx1 & (1u << 12)) != 0;

        #if defined(_MSC_VER)
    __cpuidex(regs, 7, 0);
    const bool avx2_bit = (static_cast<unsigned>(regs[1]) & (1u << 5)) != 0;
        #else
    bool avx2_bit = false;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
        avx2_bit = (ebx & (1u << 5)) != 0;
        #endif
    if (avx2_bit)
    {
        caps.avx2  = true;
        caps.fma3  = true; // no AVX2 hardware lacks FMA3; keeps the avx2 preset self-consistent
    }
    return caps;
    #else
    return simd_sse42_fma; // ARM64 via SIMDe: 128-bit paths, FMA maps natively to NEON
    #endif
}

inline const simd_caps g_simd_caps = simd_detect();

#endif

template <template <simd_caps> class Kernel, typename... Args>
inline
auto simd_dispatch(Args&&... args)
{
#ifdef COSYVOICE_NO_SIMD
    return Kernel<simd_none>::run(std::forward<Args>(args)...);
#else
    if (g_simd_caps.avx2)
        return Kernel<simd_avx2>::run(std::forward<Args>(args)...);
    if (g_simd_caps.avx)
        return Kernel<simd_avx>::run(std::forward<Args>(args)...);
    if (g_simd_caps.sse42)
        return Kernel<simd_sse42>::run(std::forward<Args>(args)...);
    return Kernel<simd_none>::run(std::forward<Args>(args)...);
#endif
}

#ifndef COSYVOICE_NO_SIMD

inline
float simd_hsum_ps(__m128 v)
{
    __m128 dup = _mm_movehdup_ps(v);
    __m128 sum = _mm_add_ps(v, dup);
    dup = _mm_movehl_ps(dup, sum);
    return _mm_cvtss_f32(_mm_add_ss(sum, dup));
}

template <simd_caps C>
inline
__m128 simd_fmadd_ps(__m128 a, __m128 b, __m128 c)
{
    if constexpr (C.fma3)
        return _mm_fmadd_ps(a, b, c);
    else
        return _mm_add_ps(c, _mm_mul_ps(a, b));
}

template <simd_caps C>
inline
__m128 simd_fmsub_ps(__m128 a, __m128 b, __m128 c)
{
    if constexpr (C.fma3)
        return _mm_fmsub_ps(a, b, c);
    else
        return _mm_sub_ps(_mm_mul_ps(a, b), c);
}

template <simd_caps C>
inline
__m256 simd_fmadd_ps(__m256 a, __m256 b, __m256 c)
{
    if constexpr (C.fma3)
        return _mm256_fmadd_ps(a, b, c);
    else
        return _mm256_add_ps(c, _mm256_mul_ps(a, b));
}

template <simd_caps C>
inline
__m256 simd_fmsub_ps(__m256 a, __m256 b, __m256 c)
{
    if constexpr (C.fma3)
        return _mm256_fmsub_ps(a, b, c);
    else
        return _mm256_sub_ps(_mm256_mul_ps(a, b), c);
}

inline
__m128 simd_load4_strided_ps(const float* base, int stride)
{
    return _mm_set_ps(base[3 * stride], base[2 * stride], base[1 * stride], base[0]);
}

inline
__m128 simd_load4_indexed_ps(const float* base, const int* idx)
{
    return _mm_set_ps(base[idx[3]], base[idx[2]], base[idx[1]], base[idx[0]]);
}

inline
__m256 simd_load8_strided_ps(const float* base, int stride)
{
    return _mm256_setr_ps(base[0], base[stride], base[2 * stride], base[3 * stride],
        base[4 * stride], base[5 * stride], base[6 * stride], base[7 * stride]);
}

inline
__m256 simd_load8_indexed_ps(const float* base, const int* idx)
{
    return _mm256_setr_ps(base[idx[0]], base[idx[1]], base[idx[2]], base[idx[3]],
        base[idx[4]], base[idx[5]], base[idx[6]], base[idx[7]]);
}

#endif
