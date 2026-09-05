#pragma once

#if defined(__x86_64__) || defined(_M_X64)
    // Always include the intrinsic header on x86-64: the SIMD helpers below must be
    // declarable even for scalar-only builds so kernel templates that reference them
    // inside `if constexpr` branches stay parseable (MSVC name-checks discarded
    // branches). Capability macros keep the AVX/AVX-512 overloads behind the ISA the
    // toolchain actually provides.
    #include <immintrin.h>
#elif !defined(COSYVOICE_NO_SIMD)
// non-x86 (SIMDe present -- CMake only defines COSYVOICE_HAS_* and includes
// this when it is): the build provides the SSE4.2+FMA3 class, so only the
// 128-bit + FMA headers are needed. Every AVX/AVX-512 name lives inside a
// `#if defined(COSYVOICE_HAS_AVX/AVX2/AVX512)` block in the kernel impls, and
// those macros are never defined on non-x86 -- the intrinsics are not merely
// uninstantiated, they are not even in the translation unit.
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/sse4.2.h>
#include <simde/x86/fma.h>
#endif

// Everything below is SIMD implementation text: when SIMD is off there is
// nothing here to compile (scalar kernels never touch it -- their calls are
// all deferred dependent expressions).
#ifndef COSYVOICE_NO_SIMD

inline __m128 simd_log_ps(__m128 x)
{
    x = _mm_max_ps(x, _mm_set1_ps(1.17549435082228751e-38f));

    __m128i bits = _mm_castps_si128(x);
    __m128 e = _mm_cvtepi32_ps(_mm_sub_epi32(_mm_srli_epi32(bits, 23), _mm_set1_epi32(127)));
    __m128 f = _mm_castsi128_ps(
        _mm_or_si128(_mm_andnot_si128(_mm_set1_epi32(static_cast<int>(0xFF800000u)), bits),
                     _mm_set1_epi32(static_cast<int>(0x3F000000u))));

    const __m128 one = _mm_set1_ps(1.f);
    __m128 mask = _mm_cmpgt_ps(f, _mm_set1_ps(0.707106781186547524f));

    __m128 u = _mm_add_ps(_mm_sub_ps(f, one), _mm_andnot_ps(mask, f));
    e = _mm_add_ps(e, _mm_and_ps(mask, one));

    __m128 z = _mm_mul_ps(u, u);
    __m128 y = _mm_set1_ps(7.0376836292e-2f);
    y = _mm_add_ps(_mm_mul_ps(y, u), _mm_set1_ps(-1.1514610310e-1f));
    y = _mm_add_ps(_mm_mul_ps(y, u), _mm_set1_ps(1.1676998740e-1f));
    y = _mm_add_ps(_mm_mul_ps(y, u), _mm_set1_ps(-1.2420140846e-1f));
    y = _mm_add_ps(_mm_mul_ps(y, u), _mm_set1_ps(1.4249322787e-1f));
    y = _mm_add_ps(_mm_mul_ps(y, u), _mm_set1_ps(-1.6668057665e-1f));
    y = _mm_add_ps(_mm_mul_ps(y, u), _mm_set1_ps(2.0000714765e-1f));
    y = _mm_add_ps(_mm_mul_ps(y, u), _mm_set1_ps(-2.4999993993e-1f));
    y = _mm_add_ps(_mm_mul_ps(y, u), _mm_set1_ps(3.3333331174e-1f));
    y = _mm_mul_ps(y, u);
    y = _mm_mul_ps(y, z);
    y = _mm_sub_ps(y, _mm_mul_ps(z, _mm_set1_ps(0.5f)));
    y = _mm_add_ps(y, u);
    y = _mm_add_ps(y, _mm_mul_ps(e, _mm_set1_ps(-2.12194440e-4f)));
    return _mm_add_ps(y, _mm_mul_ps(e, _mm_set1_ps(0.693359375f)));
}

inline __m128 simd_log10_ps(__m128 x)
{
    return _mm_mul_ps(simd_log_ps(x), _mm_set1_ps(0.43429448190325182765f));
}

inline void simd_sincos_ps(__m128 x, __m128* out_sin, __m128* out_cos)
{
    const __m128 fopi = _mm_set1_ps(0.636619772367581343f);
    const __m128 dp1 = _mm_set1_ps(1.5703125f);
    const __m128 dp2 = _mm_set1_ps(0.00048370361328125f);
    const __m128 dp3 = _mm_set1_ps(1.231816068643494e-7f);

    __m128i q = _mm_cvtps_epi32(_mm_mul_ps(x, fopi));
    __m128 qf = _mm_cvtepi32_ps(q);
    __m128 r = _mm_sub_ps(x, _mm_mul_ps(qf, dp1));
    r = _mm_sub_ps(r, _mm_mul_ps(qf, dp2));
    r = _mm_sub_ps(r, _mm_mul_ps(qf, dp3));

    __m128 r2 = _mm_mul_ps(r, r);
    __m128 s = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-1.9515295891e-4f), r2), _mm_set1_ps(8.3321608736e-3f));
    s = _mm_add_ps(_mm_mul_ps(s, r2), _mm_set1_ps(-1.6666654611e-1f));
    s = _mm_mul_ps(s, r2);
    s = _mm_mul_ps(s, r);
    s = _mm_add_ps(s, r);

    __m128 c = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(2.443315711809948e-5f), r2), _mm_set1_ps(-1.388731625493765e-3f));
    c = _mm_add_ps(_mm_mul_ps(c, r2), _mm_set1_ps(4.166664568298827e-2f));
    c = _mm_mul_ps(c, r2);
    c = _mm_sub_ps(c, _mm_set1_ps(0.5f));
    c = _mm_mul_ps(c, r2);
    c = _mm_add_ps(c, _mm_set1_ps(1.f));

    const __m128i one_i = _mm_set1_epi32(1);
    const __m128i two_i = _mm_set1_epi32(2);
    __m128 swap = _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_and_si128(q, one_i), one_i));
    __m128 quad_xor = _mm_castsi128_ps(_mm_slli_epi32(_mm_and_si128(q, two_i), 30));

    __m128 ns = _mm_sub_ps(_mm_setzero_ps(), s);
    __m128 sv = _mm_or_ps(_mm_and_ps(swap, c), _mm_andnot_ps(swap, s));
    __m128 cv = _mm_or_ps(_mm_and_ps(swap, ns), _mm_andnot_ps(swap, c));
    *out_sin = _mm_xor_ps(sv, quad_xor);
    *out_cos = _mm_xor_ps(cv, quad_xor);
}

#if defined(__AVX__) || (defined(_MSC_VER) && defined(_M_X64))

inline void simd_sincos_ps(__m256 x, __m256* out_sin, __m256* out_cos)
{
    __m128 sl, cl;
    simd_sincos_ps(_mm256_castps256_ps128(x), &sl, &cl);
    __m128 sh, ch;
    simd_sincos_ps(_mm256_extractf128_ps(x, 1), &sh, &ch);
    *out_sin = _mm256_insertf128_ps(_mm256_castps128_ps256(sl), sh, 1);
    *out_cos = _mm256_insertf128_ps(_mm256_castps128_ps256(cl), ch, 1);
}

#endif

#if defined(__AVX__) || (defined(_MSC_VER) && defined(_M_X64))

inline __m256 simd_log_ps(__m256 x)
{
    return _mm256_insertf128_ps(
        _mm256_castps128_ps256(simd_log_ps(_mm256_castps256_ps128(x))),
        simd_log_ps(_mm256_extractf128_ps(x, 1)), 1);
}

inline __m256 simd_log10_ps(__m256 x)
{
    return _mm256_insertf128_ps(
        _mm256_castps128_ps256(simd_log10_ps(_mm256_castps256_ps128(x))),
        simd_log10_ps(_mm256_extractf128_ps(x, 1)), 1);
}

#endif
#if defined(__AVX512F__) || (defined(_MSC_VER) && defined(_M_X64))

inline __m512 simd_log_ps(__m512 x)
{
    x = _mm512_max_ps(x, _mm512_set1_ps(1.17549435082228751e-38f));

    __m512i bits = _mm512_castps_si512(x);
    __m512 e = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_srli_epi32(bits, 23), _mm512_set1_epi32(127)));
    __m512 f = _mm512_castsi512_ps(
        _mm512_or_si512(_mm512_andnot_si512(_mm512_set1_epi32(static_cast<int>(0xFF800000u)), bits),
                        _mm512_set1_epi32(static_cast<int>(0x3F000000u))));

    const __m512 one = _mm512_set1_ps(1.f);
    const __mmask16 mask = _mm512_cmp_ps_mask(f, _mm512_set1_ps(0.707106781186547524f), _CMP_GT_OQ);

    __m512 u = _mm512_add_ps(_mm512_sub_ps(f, one), _mm512_maskz_mov_ps(static_cast<__mmask16>(~mask), f));
    e = _mm512_add_ps(e, _mm512_maskz_mov_ps(mask, one));

    __m512 z = _mm512_mul_ps(u, u);
    __m512 y = _mm512_set1_ps(7.0376836292e-2f);
    y = _mm512_add_ps(_mm512_mul_ps(y, u), _mm512_set1_ps(-1.1514610310e-1f));
    y = _mm512_add_ps(_mm512_mul_ps(y, u), _mm512_set1_ps(1.1676998740e-1f));
    y = _mm512_add_ps(_mm512_mul_ps(y, u), _mm512_set1_ps(-1.2420140846e-1f));
    y = _mm512_add_ps(_mm512_mul_ps(y, u), _mm512_set1_ps(1.4249322787e-1f));
    y = _mm512_add_ps(_mm512_mul_ps(y, u), _mm512_set1_ps(-1.6668057665e-1f));
    y = _mm512_add_ps(_mm512_mul_ps(y, u), _mm512_set1_ps(2.0000714765e-1f));
    y = _mm512_add_ps(_mm512_mul_ps(y, u), _mm512_set1_ps(-2.4999993993e-1f));
    y = _mm512_add_ps(_mm512_mul_ps(y, u), _mm512_set1_ps(3.3333331174e-1f));
    y = _mm512_mul_ps(y, u);
    y = _mm512_mul_ps(y, z);
    y = _mm512_sub_ps(y, _mm512_mul_ps(z, _mm512_set1_ps(0.5f)));
    y = _mm512_add_ps(y, u);
    y = _mm512_add_ps(y, _mm512_mul_ps(e, _mm512_set1_ps(-2.12194440e-4f)));
    return _mm512_add_ps(y, _mm512_mul_ps(e, _mm512_set1_ps(0.693359375f)));
}

inline __m512 simd_log10_ps(__m512 x)
{
    return _mm512_mul_ps(simd_log_ps(x), _mm512_set1_ps(0.43429448190325182765f));
}

inline void simd_sincos_ps(__m512 x, __m512* out_sin, __m512* out_cos)
{
    const __m512 fopi = _mm512_set1_ps(0.636619772367581343f);
    const __m512 dp1 = _mm512_set1_ps(1.5703125f);
    const __m512 dp2 = _mm512_set1_ps(0.00048370361328125f);
    const __m512 dp3 = _mm512_set1_ps(1.231816068643494e-7f);

    __m512i q = _mm512_cvtps_epi32(_mm512_mul_ps(x, fopi));
    __m512 qf = _mm512_cvtepi32_ps(q);
    __m512 r = _mm512_sub_ps(x, _mm512_mul_ps(qf, dp1));
    r = _mm512_sub_ps(r, _mm512_mul_ps(qf, dp2));
    r = _mm512_sub_ps(r, _mm512_mul_ps(qf, dp3));

    __m512 r2 = _mm512_mul_ps(r, r);
    __m512 s = _mm512_add_ps(_mm512_mul_ps(_mm512_set1_ps(-1.9515295891e-4f), r2), _mm512_set1_ps(8.3321608736e-3f));
    s = _mm512_add_ps(_mm512_mul_ps(s, r2), _mm512_set1_ps(-1.6666654611e-1f));
    s = _mm512_mul_ps(s, r2);
    s = _mm512_mul_ps(s, r);
    s = _mm512_add_ps(s, r);

    __m512 c = _mm512_add_ps(_mm512_mul_ps(_mm512_set1_ps(2.443315711809948e-5f), r2), _mm512_set1_ps(-1.388731625493765e-3f));
    c = _mm512_add_ps(_mm512_mul_ps(c, r2), _mm512_set1_ps(4.166664568298827e-2f));
    c = _mm512_mul_ps(c, r2);
    c = _mm512_sub_ps(c, _mm512_set1_ps(0.5f));
    c = _mm512_mul_ps(c, r2);
    c = _mm512_add_ps(c, _mm512_set1_ps(1.f));

    const __mmask16 swap = _mm512_test_epi32_mask(q, _mm512_set1_epi32(1));
    const __mmask16 flip = _mm512_test_epi32_mask(q, _mm512_set1_epi32(2));

    // Sign-bit flip for negation (vpxord), instead of `_mm512_setzero_ps()` +
    // vxorps: MSVC may encode the zeroing xor on a high register (xmm16+) as
    // EVEX.128, which drags in an AVX512VL dependency for no reason.
    const __m512i signbit = _mm512_set1_epi32(0x80000000);
    const __m512 ns = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(s), signbit));
    __m512 sv = _mm512_mask_blend_ps(swap, s, c);
    __m512 cv = _mm512_mask_blend_ps(swap, c, ns);
    *out_sin = _mm512_castsi512_ps(_mm512_mask_xor_epi32(_mm512_castps_si512(sv), flip, _mm512_castps_si512(sv), signbit));
    *out_cos = _mm512_castsi512_ps(_mm512_mask_xor_epi32(_mm512_castps_si512(cv), flip, _mm512_castps_si512(cv), signbit));
}

#endif

#endif  // !COSYVOICE_NO_SIMD
