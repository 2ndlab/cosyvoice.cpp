#include "simd-dispatch.h"

#ifdef _DEBUG
    #include <format>
    #include "cosyvoice-internal.h"
#endif

#if defined(__x86_64__) || defined(_M_X64)
    #if defined(_MSC_VER)
        #include <intrin.h>
    #else
        #include <cpuid.h>
    #endif
#endif

inline
static simd_caps simd_detect()
{
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
    const unsigned ebx7 = static_cast<unsigned>(regs[1]);
#else
    unsigned ebx7 = 0;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
        ebx7 = ebx;
#endif
    const bool avx2_bit = (ebx7 & (1u << 5)) != 0;
    if (avx2_bit)
    {
        caps.avx2  = true;
        caps.fma3  = true; // no AVX2 hardware lacks FMA3; keeps the avx2 preset self-consistent
    }
    const bool avx512f_bit  = (ebx7 & (1u << 16)) != 0;
    const bool avx512dq_bit = (ebx7 & (1u << 17)) != 0;
    const bool avx512bw_bit = (ebx7 & (1u << 30)) != 0;
    const bool avx512vl_bit = (ebx7 & (1u << 31)) != 0;
    // The 512 tier requires F+DQ+BW (16-bit tail masks lower to BW's kmovw;
    // sincos' _mm512_test_epi32_mask to DQ's vptestmd). VL is required too:
    // both MSVC and gcc/clang encode register spills/prologues in these
    // objects as EVEX vmovq/vmovdqa to xmm16+/ymm16+ (AVX512VL territory).
    // Parts with only some of these sub-sets (e.g. Knights Landing: F alone)
    // fall through to AVX2.
    if (avx512f_bit && avx512dq_bit && avx512bw_bit && avx512vl_bit && (xcr0 & 0xe6) == 0xe6) // OS must also enable opmask(bit5), ZMM_Hi256(bit6), Hi16_ZMM(bit7)
    {
        caps.avx512 = true;
        caps.fma3   = true;                    // AVX-512F includes FMA
    }

    return caps;
}

const simd_caps g_simd_caps = simd_detect();
