#pragma once

// ---------------------------------------------------------------------------
// SIMD implementation header (tier objects only). Include ONLY from the tier
// object sources (after defining the simd_vec<C,N> specializations for that
// preset). Callers never include this file. It owns every intrinsic type and
// SIMD helper in the project: simd-math.h (immintrin/SIMDe + log/sincos) plus
// the fma/load/reduce support helpers, then the kernel bodies. simd-dispatch.h
// stays intrinsic-free so caller TUs are never polluted by <immintrin.h>.
// ---------------------------------------------------------------------------
#include "simd-math.h"
#include "simd-kernels.h"

#include <cstddef>

// Intrinsic-typed helpers: compiled only where SIMD is enabled at all (their
// uses inside kernel bodies are dependent calls, deferred until instantiation,
// so scalar builds never need the declarations).
#ifndef COSYVOICE_NO_SIMD

inline
float simd_hsum_ps(__m128 v)
{
    __m128 dup = _mm_movehdup_ps(v);
    __m128 sum = _mm_add_ps(v, dup);
    dup = _mm_movehl_ps(dup, sum);
    return _mm_cvtss_f32(_mm_add_ss(sum, dup));
}

// Params are dependent types (simd_vec<C,N>::f) so the FMA intrinsic call is a
// dependent expression: its name lookup is deferred to instantiation. A fma3
// preset is instantiated only by a tier whose flags -mfma / -mavx512f (or SIMDe)
// declare the intrinsic, so a mismatched build fails with an undeclared-name
// error at instantiation instead of silently emitting mul+add.
template <simd_caps C>
inline
auto simd_fmadd_ps(typename simd_vec<C,4>::f a, typename simd_vec<C,4>::f b, typename simd_vec<C,4>::f c)
{
    if constexpr (C.fma3)
        return _mm_fmadd_ps(a, b, c);
    else
        return _mm_add_ps(c, _mm_mul_ps(a, b));
}

template <simd_caps C>
inline
auto simd_fmsub_ps(typename simd_vec<C,4>::f a, typename simd_vec<C,4>::f b, typename simd_vec<C,4>::f c)
{
    if constexpr (C.fma3)
        return _mm_fmsub_ps(a, b, c);
    else
        return _mm_sub_ps(_mm_mul_ps(a, b), c);
}

// These helpers are templated on simd_caps ONLY so that kernel call sites can
// spell them `simd_load*_ps<C>(...)`: the explicit template-id makes the call a
// DEPENDENT expression, so gcc/clang defer the name lookup to instantiation
// instead of demanding the declaration exist while parsing a TU that (correctly)
// does not have the flag-set for the helper's vector type. Without the <C> the
// helpers are only ever invoked with non-dependent arguments (float*/int), so
// even inside an `if constexpr (C.avx*)` discarded branch the compilers would
// hard-check the name.
template <simd_caps C>
inline
__m128 simd_load4_strided_ps(const float* base, int stride)
{
    return _mm_set_ps(base[3 * stride], base[2 * stride], base[1 * stride], base[0]);
}

template <simd_caps C>
inline
__m128 simd_load4_indexed_ps(const float* base, const int* idx)
{
    return _mm_set_ps(base[idx[3]], base[idx[2]], base[idx[1]], base[idx[0]]);
}

#if defined(__AVX__) || (defined(_MSC_VER) && defined(_M_X64))

template <simd_caps C>
inline
auto simd_fmadd_ps(typename simd_vec<C,8>::f a, typename simd_vec<C,8>::f b, typename simd_vec<C,8>::f c)
{
    if constexpr (C.fma3)
        return _mm256_fmadd_ps(a, b, c);
    else
        return _mm256_add_ps(c, _mm256_mul_ps(a, b));
}

template <simd_caps C>
inline
auto simd_fmsub_ps(typename simd_vec<C,8>::f a, typename simd_vec<C,8>::f b, typename simd_vec<C,8>::f c)
{
    if constexpr (C.fma3)
        return _mm256_fmsub_ps(a, b, c);
    else
        return _mm256_sub_ps(_mm256_mul_ps(a, b), c);
}

template <simd_caps C>
inline
__m256 simd_load8_strided_ps(const float* base, int stride)
{
    return _mm256_setr_ps(base[0], base[stride], base[2 * stride], base[3 * stride],
        base[4 * stride], base[5 * stride], base[6 * stride], base[7 * stride]);
}

template <simd_caps C>
inline
__m256 simd_load8_indexed_ps(const float* base, const int* idx)
{
    return _mm256_setr_ps(base[idx[0]], base[idx[1]], base[idx[2]], base[idx[3]],
        base[idx[4]], base[idx[5]], base[idx[6]], base[idx[7]]);
}

#endif
#if defined(__AVX512F__) || (defined(_MSC_VER) && defined(_M_X64))

template <simd_caps C>
inline
auto simd_fmadd_ps(typename simd_vec<C,16>::f a, typename simd_vec<C,16>::f b, typename simd_vec<C,16>::f c)
{
    if constexpr (C.fma3)
        return _mm512_fmadd_ps(a, b, c);
    else
        return _mm512_add_ps(c, _mm512_mul_ps(a, b));
}

template <simd_caps C>
inline
auto simd_fmsub_ps(typename simd_vec<C,16>::f a, typename simd_vec<C,16>::f b, typename simd_vec<C,16>::f c)
{
    if constexpr (C.fma3)
        return _mm512_fmsub_ps(a, b, c);
    else
        return _mm512_sub_ps(_mm512_mul_ps(a, b), c);
}

template <simd_caps C>
inline
__mmask16 simd_tail_mask16(size_t count)
{
    return static_cast<__mmask16>((1u << count) - 1u);
}

// 512 reduction. The AVX-512 tier already requires the DQ/BW sub-sets (see
// simd-dispatch.h), so the compiler-generated `_mm512_reduce_*` sequences (e.g.
// MSVC's vextractf32x8-based lowering) are all legal and better than any
// hand-rolled shuffle chain.
inline
float simd_hsum512_ps(__m512 v)
{
    return _mm512_reduce_add_ps(v);
}

inline
float simd_hmax512_ps(__m512 v)
{
    return _mm512_reduce_max_ps(v);
}

// The template parameter C here is only load-bearing at the call site:
// `simd_load16_*_ps<C>(...)` is a dependent template-id, so TUs compiled
// without AVX-512 (where this whole region is #if-excluded and the kernel's
// 512 branch is discarded) never need these names to be declared.
template <simd_caps C>
inline
__m512 simd_load16_strided_ps(const float* base, int stride)
{
#if defined(_MSC_VER)
    // MSVC's register allocator assigns gather index vectors freely from the
    // full zmm0-31 bank, but an EVEX gather encodes its VSIB index in
    // SIB.index, where 1111 is RESERVED: index=zmm16+ has no legal encoding
    // and #UDs on real hardware. VEX 256-bit gather structurally cannot hit
    // that (its index field only names ymm0-15), so split into two VEX
    // gathers and merge with vshuff32x4 (imm 0x44 = [lo lanes | hi lanes]).
    const __m256i ilo = _mm256_setr_epi32(0, stride, 2 * stride, 3 * stride,
        4 * stride, 5 * stride, 6 * stride, 7 * stride);
    const __m256i ihi = _mm256_setr_epi32(8 * stride, 9 * stride, 10 * stride, 11 * stride,
        12 * stride, 13 * stride, 14 * stride, 15 * stride);
    const __m256 lo = _mm256_i32gather_ps(base, ilo, 4);
    const __m256 hi = _mm256_i32gather_ps(base, ihi, 4);
    return _mm512_shuffle_f32x4(_mm512_castps256_ps512(lo), _mm512_castps256_ps512(hi), 0x44);
#else
    // gcc/clang keep gather indices in the low bank (no reserved-encoding bug;
    // SDE-verified), so the single full-width gather is the fastest form.
    const __m512i vidx = _mm512_setr_epi32(0, stride, 2 * stride, 3 * stride,
        4 * stride, 5 * stride, 6 * stride, 7 * stride,
        8 * stride, 9 * stride, 10 * stride, 11 * stride,
        12 * stride, 13 * stride, 14 * stride, 15 * stride);
    return _mm512_i32gather_ps(vidx, base, 4);
#endif
}

template <simd_caps C>
inline
__m512 simd_load16_indexed_ps(const float* base, const int* idx)
{
#if defined(_MSC_VER)
    // Same reserved-VSIB-index workaround as simd_load16_strided_ps.
    const __m256i ilo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(idx));
    const __m256i ihi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(idx + 8));
    const __m256 lo = _mm256_i32gather_ps(base, ilo, 4);
    const __m256 hi = _mm256_i32gather_ps(base, ihi, 4);
    return _mm512_shuffle_f32x4(_mm512_castps256_ps512(lo), _mm512_castps256_ps512(hi), 0x44);
#elif defined(__x86_64__)
    const __m512i vidx = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(idx));
    return _mm512_i32gather_ps(vidx, base, 4);
#endif
}

// tail helper: strided load of `valid` (<16) leading elements, zero-filled rest; never reads out of range
template <simd_caps C>
inline
__m512 simd_tail_load16_strided_ps(const float* base, int stride, size_t valid)
{
    alignas(64) float tmp[16];
    size_t t = 0;
    for (; t != valid; ++t) tmp[t] = base[t * stride];
    for (; t != 16; ++t) tmp[t] = 0.f;
    return _mm512_load_ps(tmp);
}
#endif

// --- AVX10.1-256 tail helpers ------------------------------------------------
// The 256-bit AVX10 class masks its own remainder (no 128-bit vector tail);
// these mirror the 16-wide helpers above at width 8. (AVX10.2 is a strict
// superset of 10.1 and no kernel uses a 10.2-only instruction, so 10.2 parts
// dispatch here -- see simd-dispatch.h.) VL-type availability: -mavx10.1-256
// defines __AVX512VL__ on clang (and the __AVX10_1_256__ feature macro),
// MSVC x64 declares every intrinsic unconditionally.
#if defined(__AVX10_1_256__) || defined(__AVX512VL__) || (defined(_MSC_VER) && defined(_M_X64))

template <simd_caps C>
inline
__mmask8 simd_tail_mask8(size_t count)
{
    return static_cast<__mmask8>((1u << count) - 1u);
}

// strided tail load of `valid` (<8) leading elements, zero-filled rest; never
// reads out of range.
template <simd_caps C>
inline
__m256 simd_tail_load8_strided_ps(const float* base, int stride, size_t valid)
{
    alignas(32) float tmp[8];
    size_t t = 0;
    for (; t != valid; ++t) tmp[t] = base[t * stride];
    for (; t != 8; ++t) tmp[t] = 0.f;
    return _mm256_load_ps(tmp);
}

// indexed (gather-vector) tail load of `valid` (<8) entries, zero-filled rest.
template <simd_caps C>
inline
__m256 simd_tail_load8_indexed_ps(const float* base, const int* idx, size_t valid)
{
    alignas(32) float tmp[8];
    size_t t = 0;
    for (; t != valid; ++t) tmp[t] = base[idx[t]];
    for (; t != 8; ++t) tmp[t] = 0.f;
    return _mm256_load_ps(tmp);
}

#endif  // AVX10-256 tail helpers

// --- shared helpers pulled out of the kernel TUs ---------------------------
#if defined(__AVX512F__) || (defined(_MSC_VER) && defined(_M_X64))
template<simd_caps C>
inline
static void CMUL(__m512& ar, __m512& ai, __m512 br, __m512 bi)
{
    __m512 _tr = simd_fmsub_ps<C>(ar, br, _mm512_mul_ps(ai, bi));
    __m512 _ti = simd_fmadd_ps<C>(ar, bi, _mm512_mul_ps(ai, br));
    ar = _tr; ai = _ti;
}
#endif // AVX512F-type-capable

#if defined(__AVX__) || (defined(_MSC_VER) && defined(_M_X64))
template<simd_caps C>
inline
static void CMUL(__m256& ar, __m256& ai, __m256 br, __m256 bi)
{
    __m256 _tr = simd_fmsub_ps<C>(ar, br, _mm256_mul_ps(ai, bi));
    __m256 _ti = simd_fmadd_ps<C>(ar, bi, _mm256_mul_ps(ai, br));
    ar = _tr; ai = _ti;
}
#endif // AVX-type-capable

template<simd_caps C>
inline
static void CMUL(__m128& ar, __m128& ai, __m128 br, __m128 bi)
{
    __m128 _tr = simd_fmsub_ps<C>(ar, br, _mm_mul_ps(ai, bi));
    __m128 _ti = simd_fmadd_ps<C>(ar, bi, _mm_mul_ps(ai, br));
    ar = _tr; ai = _ti;
}

inline
static float hsum128_ps(__m128 v)
{
    v = _mm_hadd_ps(v, v);
    v = _mm_hadd_ps(v, v);
    return _mm_cvtss_f32(v);
}

#if defined(__AVX__) || (defined(_MSC_VER) && defined(_M_X64))
inline
static float hsum256_ps(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    return hsum128_ps(sum);
}
#endif // AVX-type-capable

static inline float hmax_ps(__m128 x)
{
    __m128 shuf1 = _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 max1 = _mm_max_ps(x, shuf1);
    __m128 shuf2 = _mm_shuffle_ps(max1, max1, _MM_SHUFFLE(1, 0, 3, 2));
    __m128 max2 = _mm_max_ps(max1, shuf2);
    return _mm_cvtss_f32(max2);
}

// __m256 needs the AVX type, which only exists on x86 toolchains that have it.
#if defined(__AVX__) || (defined(_MSC_VER) && defined(_M_X64))
static inline float hmax_ps(__m256 v)
{
    __m256 swapped = _mm256_permute2f128_ps(v, v, 0x01);
    __m256 max256 = _mm256_max_ps(v, swapped);
    __m128 low = _mm256_castps256_ps128(max256);
    return hmax_ps(low);
}
#endif


#endif  // !COSYVOICE_NO_SIMD

// --- kernel member definitions (compiled only in tier TUs) -------

// --- FFT butterfly kernels: private to this header -------------------------
// Only fft_work_kernel (below) references them, so they are instantiated
// transitively inside each tier TU: no public declarations, no extern-template
// entries, no per-DFT-node simd_dispatch or cross-TU call.
template<simd_caps C>
struct fft_bfly2_kernel
{
    static void run(float* Foutr, float* Fouti, int fstride, const float* tw_r, const float* tw_i, int m);
};

template<simd_caps C>
struct fft_bfly4_kernel
{
    static void run(float* Foutr, float* Fouti, int fstride, float* twr, float* twi, int m);
};

template<simd_caps C>
struct fft_bfly3_kernel
{
    static void run(float* Foutr, float* Fouti, int fstride, float* tw1_r, float* tw1_i, int m);
};

template<simd_caps C>
struct fft_bfly5_kernel
{
    static void run(float* Foutr, float* Fouti, int fstride, const float* tw_r, const float* tw_i, int m);
};

template<simd_caps C>
struct fft_bfly_generic_kernel
{
    static void run(float* Foutr, float* Fouti, int fstride, const float* twiddles_r, const float* twiddles_i, int Norig, int m, int p, float* scratchr, float* scratchi);
};

template<simd_caps C>
void fft_bfly2_kernel<C>::run(float* Foutr, float* Fouti, int fstride, const float* tw_r, const float* tw_i, int m)
{
    int i = 0;
#if defined(COSYVOICE_HAS_AVX512)
    if constexpr (C.avx512)
    {
        if (fstride == 1)
            for (; i + 15 < m; i += 16)
            {
                typename simd_vec<C, 16>::f F2r = _mm512_loadu_ps(Foutr + m + i);
                typename simd_vec<C, 16>::f F2i = _mm512_loadu_ps(Fouti + m + i);
                typename simd_vec<C, 16>::f Twr = _mm512_loadu_ps(tw_r + i);
                typename simd_vec<C, 16>::f Twi = _mm512_loadu_ps(tw_i + i);

                typename simd_vec<C, 16>::f tr = simd_fmsub_ps<C>(F2r, Twr, _mm512_mul_ps(F2i, Twi));
                typename simd_vec<C, 16>::f ti = simd_fmadd_ps<C>(F2r, Twi, _mm512_mul_ps(F2i, Twr));

                typename simd_vec<C, 16>::f Fr = _mm512_loadu_ps(Foutr + i);
                typename simd_vec<C, 16>::f Fi = _mm512_loadu_ps(Fouti + i);

                _mm512_storeu_ps(Foutr + m + i, _mm512_sub_ps(Fr, tr));
                _mm512_storeu_ps(Fouti + m + i, _mm512_sub_ps(Fi, ti));
                _mm512_storeu_ps(Foutr + i, _mm512_add_ps(Fr, tr));
                _mm512_storeu_ps(Fouti + i, _mm512_add_ps(Fi, ti));
            }
        else for (; i + 15 < m; i += 16)
        {
            typename simd_vec<C, 16>::f F2r = _mm512_loadu_ps(Foutr + m + i);
            typename simd_vec<C, 16>::f F2i = _mm512_loadu_ps(Fouti + m + i);
            typename simd_vec<C, 16>::f Twr = simd_load16_strided_ps<C>(tw_r + i * fstride, fstride);
            typename simd_vec<C, 16>::f Twi = simd_load16_strided_ps<C>(tw_i + i * fstride, fstride);

            typename simd_vec<C, 16>::f tr = simd_fmsub_ps<C>(F2r, Twr, _mm512_mul_ps(F2i, Twi));
            typename simd_vec<C, 16>::f ti = simd_fmadd_ps<C>(F2r, Twi, _mm512_mul_ps(F2i, Twr));

            typename simd_vec<C, 16>::f Fr = _mm512_loadu_ps(Foutr + i);
            typename simd_vec<C, 16>::f Fi = _mm512_loadu_ps(Fouti + i);

            _mm512_storeu_ps(Foutr + m + i, _mm512_sub_ps(Fr, tr));
            _mm512_storeu_ps(Fouti + m + i, _mm512_sub_ps(Fi, ti));
            _mm512_storeu_ps(Foutr + i, _mm512_add_ps(Fr, tr));
            _mm512_storeu_ps(Fouti + i, _mm512_add_ps(Fi, ti));
        }
        if (i < m)
        {
            const typename simd_vec<C, 16>::mask mask = simd_tail_mask16<C>(m - i);
            typename simd_vec<C, 16>::f F2r = _mm512_maskz_loadu_ps(mask, Foutr + m + i);
            typename simd_vec<C, 16>::f F2i = _mm512_maskz_loadu_ps(mask, Fouti + m + i);
            typename simd_vec<C, 16>::f Twr;
            typename simd_vec<C, 16>::f Twi;
            if (fstride == 1)
            {
                Twr = _mm512_maskz_loadu_ps(mask, tw_r + i);
                Twi = _mm512_maskz_loadu_ps(mask, tw_i + i);
            }
            else
            {
                Twr = simd_tail_load16_strided_ps<C>(tw_r + i * fstride, fstride, m - i);
                Twi = simd_tail_load16_strided_ps<C>(tw_i + i * fstride, fstride, m - i);
            }

            typename simd_vec<C, 16>::f tr = simd_fmsub_ps<C>(F2r, Twr, _mm512_mul_ps(F2i, Twi));
            typename simd_vec<C, 16>::f ti = simd_fmadd_ps<C>(F2r, Twi, _mm512_mul_ps(F2i, Twr));

            typename simd_vec<C, 16>::f Fr = _mm512_maskz_loadu_ps(mask, Foutr + i);
            typename simd_vec<C, 16>::f Fi = _mm512_maskz_loadu_ps(mask, Fouti + i);

            _mm512_mask_storeu_ps(Foutr + m + i, mask, _mm512_sub_ps(Fr, tr));
            _mm512_mask_storeu_ps(Fouti + m + i, mask, _mm512_sub_ps(Fi, ti));
            _mm512_mask_storeu_ps(Foutr + i, mask, _mm512_add_ps(Fr, tr));
            _mm512_mask_storeu_ps(Fouti + i, mask, _mm512_add_ps(Fi, ti));
        }
        return;
    }
    else
#endif
    {
#if defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx || C.avx10_1_256)
        {
            if (fstride == 1)
                for (; i + 7 < m; i += 8)
                {
                    typename simd_vec<C, 8>::f F2r = _mm256_loadu_ps(Foutr + m + i);
                    typename simd_vec<C, 8>::f F2i = _mm256_loadu_ps(Fouti + m + i);
                    typename simd_vec<C, 8>::f Twr = _mm256_loadu_ps(tw_r + i);
                    typename simd_vec<C, 8>::f Twi = _mm256_loadu_ps(tw_i + i);

                    typename simd_vec<C, 8>::f tr = simd_fmsub_ps<C>(F2r, Twr, _mm256_mul_ps(F2i, Twi));
                    typename simd_vec<C, 8>::f ti = simd_fmadd_ps<C>(F2r, Twi, _mm256_mul_ps(F2i, Twr));

                    typename simd_vec<C, 8>::f Fr = _mm256_loadu_ps(Foutr + i);
                    typename simd_vec<C, 8>::f Fi = _mm256_loadu_ps(Fouti + i);

                    _mm256_storeu_ps(Foutr + m + i, _mm256_sub_ps(Fr, tr));
                    _mm256_storeu_ps(Fouti + m + i, _mm256_sub_ps(Fi, ti));
                    _mm256_storeu_ps(Foutr + i, _mm256_add_ps(Fr, tr));
                    _mm256_storeu_ps(Fouti + i, _mm256_add_ps(Fi, ti));
                }
            else for (; i + 7 < m; i += 8)
            {
                typename simd_vec<C, 8>::f F2r = _mm256_loadu_ps(Foutr + m + i);
                typename simd_vec<C, 8>::f F2i = _mm256_loadu_ps(Fouti + m + i);
                typename simd_vec<C, 8>::f Twr;
                typename simd_vec<C, 8>::f Twi;
#if defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
                if constexpr (C.avx2 || C.avx10_1_256)
                {
                    const typename simd_vec<C, 8>::i vidx0 = _mm256_setr_epi32(
                        0 * fstride, 1 * fstride, 2 * fstride, 3 * fstride,
                        4 * fstride, 5 * fstride, 6 * fstride, 7 * fstride);
                    Twr = _mm256_i32gather_ps(tw_r + i * fstride, vidx0, sizeof(float));
                    Twi = _mm256_i32gather_ps(tw_i + i * fstride, vidx0, sizeof(float));
                }
                else
#endif
                {
                    Twr = simd_load8_strided_ps<C>(tw_r + i * fstride, fstride);
                    Twi = simd_load8_strided_ps<C>(tw_i + i * fstride, fstride);
                }

                typename simd_vec<C, 8>::f tr = simd_fmsub_ps<C>(F2r, Twr, _mm256_mul_ps(F2i, Twi));
                typename simd_vec<C, 8>::f ti = simd_fmadd_ps<C>(F2r, Twi, _mm256_mul_ps(F2i, Twr));

                typename simd_vec<C, 8>::f Fr = _mm256_loadu_ps(Foutr + i);
                typename simd_vec<C, 8>::f Fi = _mm256_loadu_ps(Fouti + i);

                _mm256_storeu_ps(Foutr + m + i, _mm256_sub_ps(Fr, tr));
                _mm256_storeu_ps(Fouti + m + i, _mm256_sub_ps(Fi, ti));
                _mm256_storeu_ps(Foutr + i, _mm256_add_ps(Fr, tr));
                _mm256_storeu_ps(Fouti + i, _mm256_add_ps(Fi, ti));
            }
        }
#endif
#if defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx10_1_256)
        {
        if (i < m)
        {
            const typename simd_vec<C, 8>::mask mask = simd_tail_mask8<C>(m - i);
            typename simd_vec<C, 8>::f F2r = _mm256_maskz_loadu_ps(mask, Foutr + m + i);
            typename simd_vec<C, 8>::f F2i = _mm256_maskz_loadu_ps(mask, Fouti + m + i);
            typename simd_vec<C, 8>::f Twr;
            typename simd_vec<C, 8>::f Twi;
            if (fstride == 1)
            {
                Twr = _mm256_maskz_loadu_ps(mask, tw_r + i);
                Twi = _mm256_maskz_loadu_ps(mask, tw_i + i);
            }
            else
            {
                Twr = simd_tail_load8_strided_ps<C>(tw_r + i * fstride, fstride, m - i);
                Twi = simd_tail_load8_strided_ps<C>(tw_i + i * fstride, fstride, m - i);
            }

            typename simd_vec<C, 8>::f tr = simd_fmsub_ps<C>(F2r, Twr, _mm256_mul_ps(F2i, Twi));
            typename simd_vec<C, 8>::f ti = simd_fmadd_ps<C>(F2r, Twi, _mm256_mul_ps(F2i, Twr));

            typename simd_vec<C, 8>::f Fr = _mm256_maskz_loadu_ps(mask, Foutr + i);
            typename simd_vec<C, 8>::f Fi = _mm256_maskz_loadu_ps(mask, Fouti + i);

            _mm256_mask_storeu_ps(Foutr + m + i, mask, _mm256_sub_ps(Fr, tr));
            _mm256_mask_storeu_ps(Fouti + m + i, mask, _mm256_sub_ps(Fi, ti));
            _mm256_mask_storeu_ps(Foutr + i, mask, _mm256_add_ps(Fr, tr));
            _mm256_mask_storeu_ps(Fouti + i, mask, _mm256_add_ps(Fi, ti));
        }
            return;
        }
#endif
#if defined(COSYVOICE_HAS_SSE42) || defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2)
        if constexpr (C.sse42)
        {
            for (; i + 3 < m; i += 4)
            {
                typename simd_vec<C, 4>::f F2r = _mm_loadu_ps(Foutr + m + i);
                typename simd_vec<C, 4>::f F2i = _mm_loadu_ps(Fouti + m + i);
                typename simd_vec<C, 4>::f Twr;
                typename simd_vec<C, 4>::f Twi;
                if (fstride == 1)
                {
                    Twr = _mm_loadu_ps(tw_r + i);
                    Twi = _mm_loadu_ps(tw_i + i);
                }
                else
                {
#if defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
                    if constexpr (C.avx2 || C.avx10_1_256)
                    {
                        const typename simd_vec<C, 4>::i vidx0_128 = _mm_setr_epi32(
                            0 * fstride, 1 * fstride, 2 * fstride, 3 * fstride);
                        Twr = _mm_i32gather_ps(tw_r + i * fstride, vidx0_128, sizeof(float));
                        Twi = _mm_i32gather_ps(tw_i + i * fstride, vidx0_128, sizeof(float));
                    }
                    else
#endif
                    {
                        Twr = simd_load4_strided_ps<C>(tw_r + i * fstride, fstride);
                        Twi = simd_load4_strided_ps<C>(tw_i + i * fstride, fstride);
                    }
                }

                typename simd_vec<C, 4>::f tr = simd_fmsub_ps<C>(F2r, Twr, _mm_mul_ps(F2i, Twi));
                typename simd_vec<C, 4>::f ti = simd_fmadd_ps<C>(F2r, Twi, _mm_mul_ps(F2i, Twr));

                typename simd_vec<C, 4>::f Fr = _mm_loadu_ps(Foutr + i);
                typename simd_vec<C, 4>::f Fi = _mm_loadu_ps(Fouti + i);

                _mm_storeu_ps(Foutr + m + i, _mm_sub_ps(Fr, tr));
                _mm_storeu_ps(Fouti + m + i, _mm_sub_ps(Fi, ti));
                _mm_storeu_ps(Foutr + i, _mm_add_ps(Fr, tr));
                _mm_storeu_ps(Fouti + i, _mm_add_ps(Fi, ti));
            }
        }
#endif

        for (; i < m; ++i)
        {
            const float twr = tw_r[i * fstride];
            const float twi = tw_i[i * fstride];

            float F2r = Foutr[m + i];
            float F2i = Fouti[m + i];

            float tr = F2r * twr - F2i * twi;
            float ti = F2r * twi + F2i * twr;

            float Fr = Foutr[i];
            float Fi = Fouti[i];

            Foutr[m + i] = Fr - tr;
            Fouti[m + i] = Fi - ti;
            Foutr[i] = Fr + tr;
            Fouti[i] = Fi + ti;
        }
    }
}

template<simd_caps C>
void fft_bfly4_kernel<C>::run(float* Foutr, float* Fouti, int fstride, float* twr, float* twi, int m)
{
    const int m2 = 2 * m;
    const int m3 = 3 * m;

    int k = 0;
#if defined(COSYVOICE_HAS_AVX512)
    if constexpr (C.avx512)
    {
        for (; k + 15 < m; k += 16)
        {
            float* p0r = Foutr + k;
            float* p0i = Fouti + k;
            float* p1r = Foutr + k + m;
            float* p1i = Fouti + k + m;
            float* p2r = Foutr + k + m2;
            float* p2i = Fouti + k + m2;
            float* p3r = Foutr + k + m3;
            float* p3i = Fouti + k + m3;

            typename simd_vec<C, 16>::f r0 = _mm512_loadu_ps(p0r);
            typename simd_vec<C, 16>::f i0 = _mm512_loadu_ps(p0i);
            typename simd_vec<C, 16>::f r1 = _mm512_loadu_ps(p1r);
            typename simd_vec<C, 16>::f i1 = _mm512_loadu_ps(p1i);
            typename simd_vec<C, 16>::f r2 = _mm512_loadu_ps(p2r);
            typename simd_vec<C, 16>::f i2 = _mm512_loadu_ps(p2i);
            typename simd_vec<C, 16>::f r3 = _mm512_loadu_ps(p3r);
            typename simd_vec<C, 16>::f i3 = _mm512_loadu_ps(p3i);

            typename simd_vec<C, 16>::f t1r = simd_load16_strided_ps<C>(twr + k * fstride, fstride);
            typename simd_vec<C, 16>::f t1i = simd_load16_strided_ps<C>(twi + k * fstride, fstride);
            typename simd_vec<C, 16>::f t2r = simd_load16_strided_ps<C>(twr + 2 * k * fstride, 2 * fstride);
            typename simd_vec<C, 16>::f t2i = simd_load16_strided_ps<C>(twi + 2 * k * fstride, 2 * fstride);
            typename simd_vec<C, 16>::f t3r = simd_load16_strided_ps<C>(twr + 3 * k * fstride, 3 * fstride);
            typename simd_vec<C, 16>::f t3i = simd_load16_strided_ps<C>(twi + 3 * k * fstride, 3 * fstride);

            typename simd_vec<C, 16>::f s0r = simd_fmsub_ps<C>(r1, t1r, _mm512_mul_ps(i1, t1i));
            typename simd_vec<C, 16>::f s0i = simd_fmadd_ps<C>(r1, t1i, _mm512_mul_ps(i1, t1r));
            typename simd_vec<C, 16>::f s1r = simd_fmsub_ps<C>(r2, t2r, _mm512_mul_ps(i2, t2i));
            typename simd_vec<C, 16>::f s1i = simd_fmadd_ps<C>(r2, t2i, _mm512_mul_ps(i2, t2r));
            typename simd_vec<C, 16>::f s2r = simd_fmsub_ps<C>(r3, t3r, _mm512_mul_ps(i3, t3i));
            typename simd_vec<C, 16>::f s2i = simd_fmadd_ps<C>(r3, t3i, _mm512_mul_ps(i3, t3r));

            typename simd_vec<C, 16>::f tmp_r = _mm512_sub_ps(r0, s1r);
            typename simd_vec<C, 16>::f tmp_i = _mm512_sub_ps(i0, s1i);

            r0 = _mm512_add_ps(r0, s1r);
            i0 = _mm512_add_ps(i0, s1i);

            typename simd_vec<C, 16>::f s3r = _mm512_add_ps(s0r, s2r);
            typename simd_vec<C, 16>::f s3i = _mm512_add_ps(s0i, s2i);
            typename simd_vec<C, 16>::f s4r = _mm512_sub_ps(s0r, s2r);
            typename simd_vec<C, 16>::f s4i = _mm512_sub_ps(s0i, s2i);

            _mm512_storeu_ps(p2r, _mm512_sub_ps(r0, s3r));
            _mm512_storeu_ps(p2i, _mm512_sub_ps(i0, s3i));
            _mm512_storeu_ps(p0r, _mm512_add_ps(r0, s3r));
            _mm512_storeu_ps(p0i, _mm512_add_ps(i0, s3i));
            _mm512_storeu_ps(p1r, _mm512_add_ps(tmp_r, s4i));
            _mm512_storeu_ps(p1i, _mm512_sub_ps(tmp_i, s4r));
            _mm512_storeu_ps(p3r, _mm512_sub_ps(tmp_r, s4i));
            _mm512_storeu_ps(p3i, _mm512_add_ps(tmp_i, s4r));
        }
        if (k < m)
        {
            const typename simd_vec<C, 16>::mask mask = simd_tail_mask16<C>(m - k);
            const size_t rem = static_cast<size_t>(m - k);
            float* p0r = Foutr + k;
            float* p0i = Fouti + k;
            float* p1r = Foutr + k + m;
            float* p1i = Fouti + k + m;
            float* p2r = Foutr + k + m2;
            float* p2i = Fouti + k + m2;
            float* p3r = Foutr + k + m3;
            float* p3i = Fouti + k + m3;

            typename simd_vec<C, 16>::f r0 = _mm512_maskz_loadu_ps(mask, p0r);
            typename simd_vec<C, 16>::f i0 = _mm512_maskz_loadu_ps(mask, p0i);
            typename simd_vec<C, 16>::f r1 = _mm512_maskz_loadu_ps(mask, p1r);
            typename simd_vec<C, 16>::f i1 = _mm512_maskz_loadu_ps(mask, p1i);
            typename simd_vec<C, 16>::f r2 = _mm512_maskz_loadu_ps(mask, p2r);
            typename simd_vec<C, 16>::f i2 = _mm512_maskz_loadu_ps(mask, p2i);
            typename simd_vec<C, 16>::f r3 = _mm512_maskz_loadu_ps(mask, p3r);
            typename simd_vec<C, 16>::f i3 = _mm512_maskz_loadu_ps(mask, p3i);

            typename simd_vec<C, 16>::f t1r = simd_tail_load16_strided_ps<C>(twr + k * fstride, fstride, rem);
            typename simd_vec<C, 16>::f t1i = simd_tail_load16_strided_ps<C>(twi + k * fstride, fstride, rem);
            typename simd_vec<C, 16>::f t2r = simd_tail_load16_strided_ps<C>(twr + 2 * k * fstride, 2 * fstride, rem);
            typename simd_vec<C, 16>::f t2i = simd_tail_load16_strided_ps<C>(twi + 2 * k * fstride, 2 * fstride, rem);
            typename simd_vec<C, 16>::f t3r = simd_tail_load16_strided_ps<C>(twr + 3 * k * fstride, 3 * fstride, rem);
            typename simd_vec<C, 16>::f t3i = simd_tail_load16_strided_ps<C>(twi + 3 * k * fstride, 3 * fstride, rem);

            typename simd_vec<C, 16>::f s0r = simd_fmsub_ps<C>(r1, t1r, _mm512_mul_ps(i1, t1i));
            typename simd_vec<C, 16>::f s0i = simd_fmadd_ps<C>(r1, t1i, _mm512_mul_ps(i1, t1r));
            typename simd_vec<C, 16>::f s1r = simd_fmsub_ps<C>(r2, t2r, _mm512_mul_ps(i2, t2i));
            typename simd_vec<C, 16>::f s1i = simd_fmadd_ps<C>(r2, t2i, _mm512_mul_ps(i2, t2r));
            typename simd_vec<C, 16>::f s2r = simd_fmsub_ps<C>(r3, t3r, _mm512_mul_ps(i3, t3i));
            typename simd_vec<C, 16>::f s2i = simd_fmadd_ps<C>(r3, t3i, _mm512_mul_ps(i3, t3r));

            typename simd_vec<C, 16>::f tmp_r = _mm512_sub_ps(r0, s1r);
            typename simd_vec<C, 16>::f tmp_i = _mm512_sub_ps(i0, s1i);

            r0 = _mm512_add_ps(r0, s1r);
            i0 = _mm512_add_ps(i0, s1i);

            typename simd_vec<C, 16>::f s3r = _mm512_add_ps(s0r, s2r);
            typename simd_vec<C, 16>::f s3i = _mm512_add_ps(s0i, s2i);
            typename simd_vec<C, 16>::f s4r = _mm512_sub_ps(s0r, s2r);
            typename simd_vec<C, 16>::f s4i = _mm512_sub_ps(s0i, s2i);

            _mm512_mask_storeu_ps(p2r, mask, _mm512_sub_ps(r0, s3r));
            _mm512_mask_storeu_ps(p2i, mask, _mm512_sub_ps(i0, s3i));
            _mm512_mask_storeu_ps(p0r, mask, _mm512_add_ps(r0, s3r));
            _mm512_mask_storeu_ps(p0i, mask, _mm512_add_ps(i0, s3i));
            _mm512_mask_storeu_ps(p1r, mask, _mm512_add_ps(tmp_r, s4i));
            _mm512_mask_storeu_ps(p1i, mask, _mm512_sub_ps(tmp_i, s4r));
            _mm512_mask_storeu_ps(p3r, mask, _mm512_sub_ps(tmp_r, s4i));
            _mm512_mask_storeu_ps(p3i, mask, _mm512_add_ps(tmp_i, s4r));
        }
        return;
    }
    else
#endif
    {
#if defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx || C.avx10_1_256)
        {
            for (; k + 7 < m; k += 8)
            {
                float* p0r = Foutr + k;
                float* p0i = Fouti + k;

                float* p1r = Foutr + k + m;
                float* p1i = Fouti + k + m;

                float* p2r = Foutr + k + m2;
                float* p2i = Fouti + k + m2;

                float* p3r = Foutr + k + m3;
                float* p3i = Fouti + k + m3;

                typename simd_vec<C, 8>::f r0 = _mm256_loadu_ps(p0r);
                typename simd_vec<C, 8>::f i0 = _mm256_loadu_ps(p0i);

                typename simd_vec<C, 8>::f r1 = _mm256_loadu_ps(p1r);
                typename simd_vec<C, 8>::f i1 = _mm256_loadu_ps(p1i);

                typename simd_vec<C, 8>::f r2 = _mm256_loadu_ps(p2r);
                typename simd_vec<C, 8>::f i2 = _mm256_loadu_ps(p2i);

                typename simd_vec<C, 8>::f r3 = _mm256_loadu_ps(p3r);
                typename simd_vec<C, 8>::f i3 = _mm256_loadu_ps(p3i);

                typename simd_vec<C, 8>::f t1r, t1i, t2r, t2i, t3r, t3i;

#if defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
                if constexpr (C.avx2 || C.avx10_1_256)
                {
                    const typename simd_vec<C, 8>::i vidx0 = _mm256_setr_epi32(
                        0, fstride, 2 * fstride, 3 * fstride,
                        4 * fstride, 5 * fstride, 6 * fstride, 7 * fstride);
                    const typename simd_vec<C, 8>::i vidx1 = _mm256_mullo_epi32(
                        vidx0, _mm256_set1_epi32(2));  /* 2 * fstride */
                    const typename simd_vec<C, 8>::i vidx2 = _mm256_mullo_epi32(
                        vidx0, _mm256_set1_epi32(3));

                    t1r = _mm256_i32gather_ps(twr + k * fstride, vidx0, 4);
                    t1i = _mm256_i32gather_ps(twi + k * fstride, vidx0, 4);

                    t2r = _mm256_i32gather_ps(twr + 2 * k * fstride,
                        vidx1, 4);
                    t2i = _mm256_i32gather_ps(twi + 2 * k * fstride,
                        vidx1, 4);

                    t3r = _mm256_i32gather_ps(twr + 3 * k * fstride,
                        vidx2, 4);
                    t3i = _mm256_i32gather_ps(twi + 3 * k * fstride,
                        vidx2, 4);
                }
                else
#endif
                {
                    if (fstride == 1)
                    {
                        t1r = _mm256_loadu_ps(twr + k);
                        t1i = _mm256_loadu_ps(twi + k);
                    }
                    else
                    {
                        t1r = simd_load8_strided_ps<C>(twr + k * fstride, fstride);
                        t1i = simd_load8_strided_ps<C>(twi + k * fstride, fstride);
                    }

                    t2r = simd_load8_strided_ps<C>(twr + 2 * k * fstride, 2 * fstride);
                    t2i = simd_load8_strided_ps<C>(twi + 2 * k * fstride, 2 * fstride);

                    t3r = simd_load8_strided_ps<C>(twr + 3 * k * fstride, 3 * fstride);
                    t3i = simd_load8_strided_ps<C>(twi + 3 * k * fstride, 3 * fstride);
                }

                typename simd_vec<C, 8>::f s0r = simd_fmsub_ps<C>(r1, t1r, _mm256_mul_ps(i1, t1i));
                typename simd_vec<C, 8>::f s0i = simd_fmadd_ps<C>(r1, t1i, _mm256_mul_ps(i1, t1r));

                typename simd_vec<C, 8>::f s1r = simd_fmsub_ps<C>(r2, t2r, _mm256_mul_ps(i2, t2i));
                typename simd_vec<C, 8>::f s1i = simd_fmadd_ps<C>(r2, t2i, _mm256_mul_ps(i2, t2r));

                typename simd_vec<C, 8>::f s2r = simd_fmsub_ps<C>(r3, t3r, _mm256_mul_ps(i3, t3i));
                typename simd_vec<C, 8>::f s2i = simd_fmadd_ps<C>(r3, t3i, _mm256_mul_ps(i3, t3r));

                typename simd_vec<C, 8>::f tmp_r = _mm256_sub_ps(r0, s1r);
                typename simd_vec<C, 8>::f tmp_i = _mm256_sub_ps(i0, s1i);

                r0 = _mm256_add_ps(r0, s1r);
                i0 = _mm256_add_ps(i0, s1i);

                typename simd_vec<C, 8>::f s3r = _mm256_add_ps(s0r, s2r);
                typename simd_vec<C, 8>::f s3i = _mm256_add_ps(s0i, s2i);

                typename simd_vec<C, 8>::f s4r = _mm256_sub_ps(s0r, s2r);
                typename simd_vec<C, 8>::f s4i = _mm256_sub_ps(s0i, s2i);

                _mm256_storeu_ps(p2r, _mm256_sub_ps(r0, s3r));
                _mm256_storeu_ps(p2i, _mm256_sub_ps(i0, s3i));

                _mm256_storeu_ps(p0r, _mm256_add_ps(r0, s3r));
                _mm256_storeu_ps(p0i, _mm256_add_ps(i0, s3i));

                _mm256_storeu_ps(p1r, _mm256_add_ps(tmp_r, s4i));
                _mm256_storeu_ps(p1i, _mm256_sub_ps(tmp_i, s4r));

                _mm256_storeu_ps(p3r, _mm256_sub_ps(tmp_r, s4i));
                _mm256_storeu_ps(p3i, _mm256_add_ps(tmp_i, s4r));
            }
        }
#endif
#if defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx10_1_256)
        {
        if (k < m)
        {
            const typename simd_vec<C, 8>::mask mask = simd_tail_mask8<C>(m - k);
            const size_t rem = static_cast<size_t>(m - k);
            float* p0r = Foutr + k;
            float* p0i = Fouti + k;
            float* p1r = Foutr + k + m;
            float* p1i = Fouti + k + m;
            float* p2r = Foutr + k + m2;
            float* p2i = Fouti + k + m2;
            float* p3r = Foutr + k + m3;
            float* p3i = Fouti + k + m3;

            typename simd_vec<C, 8>::f r0 = _mm256_maskz_loadu_ps(mask, p0r);
            typename simd_vec<C, 8>::f i0 = _mm256_maskz_loadu_ps(mask, p0i);
            typename simd_vec<C, 8>::f r1 = _mm256_maskz_loadu_ps(mask, p1r);
            typename simd_vec<C, 8>::f i1 = _mm256_maskz_loadu_ps(mask, p1i);
            typename simd_vec<C, 8>::f r2 = _mm256_maskz_loadu_ps(mask, p2r);
            typename simd_vec<C, 8>::f i2 = _mm256_maskz_loadu_ps(mask, p2i);
            typename simd_vec<C, 8>::f r3 = _mm256_maskz_loadu_ps(mask, p3r);
            typename simd_vec<C, 8>::f i3 = _mm256_maskz_loadu_ps(mask, p3i);

            typename simd_vec<C, 8>::f t1r = simd_tail_load8_strided_ps<C>(twr + k * fstride, fstride, rem);
            typename simd_vec<C, 8>::f t1i = simd_tail_load8_strided_ps<C>(twi + k * fstride, fstride, rem);
            typename simd_vec<C, 8>::f t2r = simd_tail_load8_strided_ps<C>(twr + 2 * k * fstride, 2 * fstride, rem);
            typename simd_vec<C, 8>::f t2i = simd_tail_load8_strided_ps<C>(twi + 2 * k * fstride, 2 * fstride, rem);
            typename simd_vec<C, 8>::f t3r = simd_tail_load8_strided_ps<C>(twr + 3 * k * fstride, 3 * fstride, rem);
            typename simd_vec<C, 8>::f t3i = simd_tail_load8_strided_ps<C>(twi + 3 * k * fstride, 3 * fstride, rem);

            typename simd_vec<C, 8>::f s0r = simd_fmsub_ps<C>(r1, t1r, _mm256_mul_ps(i1, t1i));
            typename simd_vec<C, 8>::f s0i = simd_fmadd_ps<C>(r1, t1i, _mm256_mul_ps(i1, t1r));
            typename simd_vec<C, 8>::f s1r = simd_fmsub_ps<C>(r2, t2r, _mm256_mul_ps(i2, t2i));
            typename simd_vec<C, 8>::f s1i = simd_fmadd_ps<C>(r2, t2i, _mm256_mul_ps(i2, t2r));
            typename simd_vec<C, 8>::f s2r = simd_fmsub_ps<C>(r3, t3r, _mm256_mul_ps(i3, t3i));
            typename simd_vec<C, 8>::f s2i = simd_fmadd_ps<C>(r3, t3i, _mm256_mul_ps(i3, t3r));

            typename simd_vec<C, 8>::f tmp_r = _mm256_sub_ps(r0, s1r);
            typename simd_vec<C, 8>::f tmp_i = _mm256_sub_ps(i0, s1i);

            r0 = _mm256_add_ps(r0, s1r);
            i0 = _mm256_add_ps(i0, s1i);

            typename simd_vec<C, 8>::f s3r = _mm256_add_ps(s0r, s2r);
            typename simd_vec<C, 8>::f s3i = _mm256_add_ps(s0i, s2i);
            typename simd_vec<C, 8>::f s4r = _mm256_sub_ps(s0r, s2r);
            typename simd_vec<C, 8>::f s4i = _mm256_sub_ps(s0i, s2i);

            _mm256_mask_storeu_ps(p2r, mask, _mm256_sub_ps(r0, s3r));
            _mm256_mask_storeu_ps(p2i, mask, _mm256_sub_ps(i0, s3i));
            _mm256_mask_storeu_ps(p0r, mask, _mm256_add_ps(r0, s3r));
            _mm256_mask_storeu_ps(p0i, mask, _mm256_add_ps(i0, s3i));
            _mm256_mask_storeu_ps(p1r, mask, _mm256_add_ps(tmp_r, s4i));
            _mm256_mask_storeu_ps(p1i, mask, _mm256_sub_ps(tmp_i, s4r));
            _mm256_mask_storeu_ps(p3r, mask, _mm256_sub_ps(tmp_r, s4i));
            _mm256_mask_storeu_ps(p3i, mask, _mm256_add_ps(tmp_i, s4r));
        }
            return;
        }
#endif
#if defined(COSYVOICE_HAS_SSE42) || defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2)
        if constexpr (C.sse42)
        {
            for (; k + 3 < m; k += 4)
            {
                float* p0r = Foutr + k;
                float* p0i = Fouti + k;

                float* p1r = Foutr + k + m;
                float* p1i = Fouti + k + m;

                float* p2r = Foutr + k + m2;
                float* p2i = Fouti + k + m2;

                float* p3r = Foutr + k + m3;
                float* p3i = Fouti + k + m3;

                typename simd_vec<C, 4>::f r0 = _mm_loadu_ps(p0r);
                typename simd_vec<C, 4>::f i0 = _mm_loadu_ps(p0i);

                typename simd_vec<C, 4>::f r1 = _mm_loadu_ps(p1r);
                typename simd_vec<C, 4>::f i1 = _mm_loadu_ps(p1i);

                typename simd_vec<C, 4>::f r2 = _mm_loadu_ps(p2r);
                typename simd_vec<C, 4>::f i2 = _mm_loadu_ps(p2i);

                typename simd_vec<C, 4>::f r3 = _mm_loadu_ps(p3r);
                typename simd_vec<C, 4>::f i3 = _mm_loadu_ps(p3i);

                typename simd_vec<C, 4>::f t1r, t1i, t2r, t2i, t3r, t3i;
#if defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
                if constexpr (C.avx2 || C.avx10_1_256)
                {
                    const typename simd_vec<C, 4>::i vidx0 = _mm_setr_epi32(0, fstride, 2 * fstride, 3 * fstride);
                    const typename simd_vec<C, 4>::i vidx1 = _mm_mullo_epi32(vidx0, _mm_set1_epi32(2));
                    const typename simd_vec<C, 4>::i vidx2 = _mm_mullo_epi32(vidx0, _mm_set1_epi32(3));

                    t1r = _mm_i32gather_ps(twr + k * fstride, vidx0, 4);
                    t1i = _mm_i32gather_ps(twi + k * fstride, vidx0, 4);

                    t2r = _mm_i32gather_ps(twr + 2 * k * fstride, vidx1, 4);
                    t2i = _mm_i32gather_ps(twi + 2 * k * fstride, vidx1, 4);

                    t3r = _mm_i32gather_ps(twr + 3 * k * fstride, vidx2, 4);
                    t3i = _mm_i32gather_ps(twi + 3 * k * fstride, vidx2, 4);
                }
                else
#endif
                {
                    t1r = simd_load4_strided_ps<C>(twr + k * fstride, fstride);
                    t1i = simd_load4_strided_ps<C>(twi + k * fstride, fstride);

                    t2r = simd_load4_strided_ps<C>(twr + 2 * k * fstride, 2 * fstride);
                    t2i = simd_load4_strided_ps<C>(twi + 2 * k * fstride, 2 * fstride);

                    t3r = simd_load4_strided_ps<C>(twr + 3 * k * fstride, 3 * fstride);
                    t3i = simd_load4_strided_ps<C>(twi + 3 * k * fstride, 3 * fstride);
                }

                typename simd_vec<C, 4>::f s0r = simd_fmsub_ps<C>(r1, t1r, _mm_mul_ps(i1, t1i));
                typename simd_vec<C, 4>::f s0i = simd_fmadd_ps<C>(r1, t1i, _mm_mul_ps(i1, t1r));

                typename simd_vec<C, 4>::f s1r = simd_fmsub_ps<C>(r2, t2r, _mm_mul_ps(i2, t2i));
                typename simd_vec<C, 4>::f s1i = simd_fmadd_ps<C>(r2, t2i, _mm_mul_ps(i2, t2r));

                typename simd_vec<C, 4>::f s2r = simd_fmsub_ps<C>(r3, t3r, _mm_mul_ps(i3, t3i));
                typename simd_vec<C, 4>::f s2i = simd_fmadd_ps<C>(r3, t3i, _mm_mul_ps(i3, t3r));

                typename simd_vec<C, 4>::f tmp_r = _mm_sub_ps(r0, s1r);
                typename simd_vec<C, 4>::f tmp_i = _mm_sub_ps(i0, s1i);

                r0 = _mm_add_ps(r0, s1r);
                i0 = _mm_add_ps(i0, s1i);

                typename simd_vec<C, 4>::f s3r = _mm_add_ps(s0r, s2r);
                typename simd_vec<C, 4>::f s3i = _mm_add_ps(s0i, s2i);

                typename simd_vec<C, 4>::f s4r = _mm_sub_ps(s0r, s2r);
                typename simd_vec<C, 4>::f s4i = _mm_sub_ps(s0i, s2i);

                _mm_storeu_ps(p2r, _mm_sub_ps(r0, s3r));
                _mm_storeu_ps(p2i, _mm_sub_ps(i0, s3i));

                _mm_storeu_ps(p0r, _mm_add_ps(r0, s3r));
                _mm_storeu_ps(p0i, _mm_add_ps(i0, s3i));

                _mm_storeu_ps(p1r, _mm_add_ps(tmp_r, s4i));
                _mm_storeu_ps(p1i, _mm_sub_ps(tmp_i, s4r));

                _mm_storeu_ps(p3r, _mm_sub_ps(tmp_r, s4i));
                _mm_storeu_ps(p3i, _mm_add_ps(tmp_i, s4r));
            }
        }
#endif

        for (Foutr += k, Fouti += k; k < m; ++k)
        {
            float scratchr[6];
            float scratchi[6];

            scratchr[0] = Foutr[m] * twr[k * fstride] - Fouti[m] * twi[k * fstride];
            scratchi[0] = Foutr[m] * twi[k * fstride] + Fouti[m] * twr[k * fstride];
            scratchr[1] = Foutr[m2] * twr[k * 2 * fstride] - Fouti[m2] * twi[k * 2 * fstride];
            scratchi[1] = Foutr[m2] * twi[k * 2 * fstride] + Fouti[m2] * twr[k * 2 * fstride];
            scratchr[2] = Foutr[m3] * twr[k * 3 * fstride] - Fouti[m3] * twi[k * 3 * fstride];
            scratchi[2] = Foutr[m3] * twi[k * 3 * fstride] + Fouti[m3] * twr[k * 3 * fstride];

            scratchr[5] = *Foutr - scratchr[1];
            scratchi[5] = *Fouti - scratchi[1];
            *Foutr += scratchr[1];
            *Fouti += scratchi[1];

            scratchr[3] = scratchr[0] + scratchr[2];
            scratchi[3] = scratchi[0] + scratchi[2];
            scratchr[4] = scratchr[0] - scratchr[2];
            scratchi[4] = scratchi[0] - scratchi[2];

            Foutr[m2] = *Foutr - scratchr[3];
            Fouti[m2] = *Fouti - scratchi[3];

            *Foutr += scratchr[3];
            *Fouti += scratchi[3];

            Foutr[m] = scratchr[5] + scratchi[4];
            Fouti[m] = scratchi[5] - scratchr[4];
            Foutr[m3] = scratchr[5] - scratchi[4];
            Fouti[m3] = scratchi[5] + scratchr[4];

            ++Foutr;
            ++Fouti;
        }
    }
}

template<simd_caps C>
void fft_bfly3_kernel<C>::run(float* Foutr, float* Fouti, int fstride, float* tw1_r, float* tw1_i, int m)
{
    const int m2 = 2 * m;
    const float epi3_i = tw1_i[fstride * m];

    int k = 0;
#if defined(COSYVOICE_HAS_AVX512)
    if constexpr (C.avx512)
    {
        const typename simd_vec<C, 16>::f v_half = _mm512_set1_ps(0.5f), v_epi3 = _mm512_set1_ps(epi3_i);
        while (k < m)
        {
            const typename simd_vec<C, 16>::mask mask = (k + 16 <= m) ? static_cast<typename simd_vec<C, 16>::mask>(0xFFFFu) : simd_tail_mask16<C>(m - k);
            const size_t rem = static_cast<size_t>(m - k) < 16 ? static_cast<size_t>(m - k) : 16;

            typename simd_vec<C, 16>::f fr0 = _mm512_maskz_loadu_ps(mask, Foutr + k);
            typename simd_vec<C, 16>::f fi0 = _mm512_maskz_loadu_ps(mask, Fouti + k);
            typename simd_vec<C, 16>::f fr1 = _mm512_maskz_loadu_ps(mask, Foutr + m + k);
            typename simd_vec<C, 16>::f fi1 = _mm512_maskz_loadu_ps(mask, Fouti + m + k);
            typename simd_vec<C, 16>::f fr2 = _mm512_maskz_loadu_ps(mask, Foutr + m2 + k);
            typename simd_vec<C, 16>::f fi2 = _mm512_maskz_loadu_ps(mask, Fouti + m2 + k);

            typename simd_vec<C, 16>::f tw1r = simd_tail_load16_strided_ps<C>(tw1_r + k * fstride, fstride, rem);
            typename simd_vec<C, 16>::f tw1i = simd_tail_load16_strided_ps<C>(tw1_i + k * fstride, fstride, rem);
            typename simd_vec<C, 16>::f tw2r = simd_tail_load16_strided_ps<C>(tw1_r + 2 * k * fstride, 2 * fstride, rem);
            typename simd_vec<C, 16>::f tw2i = simd_tail_load16_strided_ps<C>(tw1_i + 2 * k * fstride, 2 * fstride, rem);

            typename simd_vec<C, 16>::f sr1 = simd_fmsub_ps<C>(fr1, tw1r, _mm512_mul_ps(fi1, tw1i));
            typename simd_vec<C, 16>::f si1 = simd_fmadd_ps<C>(fr1, tw1i, _mm512_mul_ps(fi1, tw1r));
            typename simd_vec<C, 16>::f sr2 = simd_fmsub_ps<C>(fr2, tw2r, _mm512_mul_ps(fi2, tw2i));
            typename simd_vec<C, 16>::f si2 = simd_fmadd_ps<C>(fr2, tw2i, _mm512_mul_ps(fi2, tw2r));

            typename simd_vec<C, 16>::f sr3 = _mm512_add_ps(sr1, sr2);
            typename simd_vec<C, 16>::f si3 = _mm512_add_ps(si1, si2);
            typename simd_vec<C, 16>::f sr0 = _mm512_sub_ps(sr1, sr2);
            typename simd_vec<C, 16>::f si0 = _mm512_sub_ps(si1, si2);

            typename simd_vec<C, 16>::f fr1o = _mm512_sub_ps(fr0, _mm512_mul_ps(sr3, v_half));
            typename simd_vec<C, 16>::f fi1o = _mm512_sub_ps(fi0, _mm512_mul_ps(si3, v_half));

            sr0 = _mm512_mul_ps(sr0, v_epi3);
            si0 = _mm512_mul_ps(si0, v_epi3);

            typename simd_vec<C, 16>::f fr0o = _mm512_add_ps(fr0, sr3);
            typename simd_vec<C, 16>::f fi0o = _mm512_add_ps(fi0, si3);

            typename simd_vec<C, 16>::f fr2o = _mm512_add_ps(fr1o, si0);
            typename simd_vec<C, 16>::f fi2o = _mm512_sub_ps(fi1o, sr0);

            fr1o = _mm512_sub_ps(fr1o, si0);
            fi1o = _mm512_add_ps(fi1o, sr0);

            _mm512_mask_storeu_ps(Foutr + k, mask, fr0o);
            _mm512_mask_storeu_ps(Fouti + k, mask, fi0o);
            _mm512_mask_storeu_ps(Foutr + m + k, mask, fr1o);
            _mm512_mask_storeu_ps(Fouti + m + k, mask, fi1o);
            _mm512_mask_storeu_ps(Foutr + m2 + k, mask, fr2o);
            _mm512_mask_storeu_ps(Fouti + m2 + k, mask, fi2o);

            k += 16;
        }
        return;
    }
    else
#endif
    {
#if defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx || C.avx10_1_256)
        {
            for (const typename simd_vec<C, 8>::f v_half = _mm256_set1_ps(0.5f), v_epi3 = _mm256_set1_ps(epi3_i); k + 7 < m; k += 8)
            {
                typename simd_vec<C, 8>::f fr0 = _mm256_loadu_ps(Foutr + k);
                typename simd_vec<C, 8>::f fi0 = _mm256_loadu_ps(Fouti + k);
                typename simd_vec<C, 8>::f fr1 = _mm256_loadu_ps(Foutr + m + k);
                typename simd_vec<C, 8>::f fi1 = _mm256_loadu_ps(Fouti + m + k);
                typename simd_vec<C, 8>::f fr2 = _mm256_loadu_ps(Foutr + m2 + k);
                typename simd_vec<C, 8>::f fi2 = _mm256_loadu_ps(Fouti + m2 + k);

                typename simd_vec<C, 8>::f tw1r, tw1i, tw2r, tw2i;
#if defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
                if constexpr (C.avx2 || C.avx10_1_256)
                {
                    typename simd_vec<C, 8>::i idx1 = _mm256_setr_epi32(
                        k * fstride, (k + 1) * fstride, (k + 2) * fstride, (k + 3) * fstride,
                        (k + 4) * fstride, (k + 5) * fstride, (k + 6) * fstride, (k + 7) * fstride);
                    typename simd_vec<C, 8>::i idx2 = _mm256_mullo_epi32(idx1, _mm256_set1_epi32(2));

                    tw1r = _mm256_i32gather_ps(tw1_r, idx1, 4);
                    tw1i = _mm256_i32gather_ps(tw1_i, idx1, 4);
                    tw2r = _mm256_i32gather_ps(tw1_r, idx2, 4);
                    tw2i = _mm256_i32gather_ps(tw1_i, idx2, 4);
                }
                else
#endif
                {
                    if (fstride == 1)
                    {
                        tw1r = _mm256_loadu_ps(tw1_r + k);
                        tw1i = _mm256_loadu_ps(tw1_i + k);
                    }
                    else
                    {
                        tw1r = simd_load8_strided_ps<C>(tw1_r + k * fstride, fstride);
                        tw1i = simd_load8_strided_ps<C>(tw1_i + k * fstride, fstride);
                    }
                    tw2r = simd_load8_strided_ps<C>(tw1_r + 2 * k * fstride, 2 * fstride);
                    tw2i = simd_load8_strided_ps<C>(tw1_i + 2 * k * fstride, 2 * fstride);
                }

                typename simd_vec<C, 8>::f sr1 = simd_fmsub_ps<C>(fr1, tw1r, _mm256_mul_ps(fi1, tw1i));
                typename simd_vec<C, 8>::f si1 = simd_fmadd_ps<C>(fr1, tw1i, _mm256_mul_ps(fi1, tw1r));
                typename simd_vec<C, 8>::f sr2 = simd_fmsub_ps<C>(fr2, tw2r, _mm256_mul_ps(fi2, tw2i));
                typename simd_vec<C, 8>::f si2 = simd_fmadd_ps<C>(fr2, tw2i, _mm256_mul_ps(fi2, tw2r));

                typename simd_vec<C, 8>::f sr3 = _mm256_add_ps(sr1, sr2);
                typename simd_vec<C, 8>::f si3 = _mm256_add_ps(si1, si2);
                typename simd_vec<C, 8>::f sr0 = _mm256_sub_ps(sr1, sr2);
                typename simd_vec<C, 8>::f si0 = _mm256_sub_ps(si1, si2);

                typename simd_vec<C, 8>::f fr1o = _mm256_sub_ps(fr0, _mm256_mul_ps(sr3, v_half));
                typename simd_vec<C, 8>::f fi1o = _mm256_sub_ps(fi0, _mm256_mul_ps(si3, v_half));

                sr0 = _mm256_mul_ps(sr0, v_epi3);
                si0 = _mm256_mul_ps(si0, v_epi3);

                typename simd_vec<C, 8>::f fr0o = _mm256_add_ps(fr0, sr3);
                typename simd_vec<C, 8>::f fi0o = _mm256_add_ps(fi0, si3);

                typename simd_vec<C, 8>::f fr2o = _mm256_add_ps(fr1o, si0);
                typename simd_vec<C, 8>::f fi2o = _mm256_sub_ps(fi1o, sr0);

                fr1o = _mm256_sub_ps(fr1o, si0);
                fi1o = _mm256_add_ps(fi1o, sr0);

                _mm256_storeu_ps(Foutr + k, fr0o);
                _mm256_storeu_ps(Fouti + k, fi0o);
                _mm256_storeu_ps(Foutr + m + k, fr1o);
                _mm256_storeu_ps(Fouti + m + k, fi1o);
                _mm256_storeu_ps(Foutr + m2 + k, fr2o);
                _mm256_storeu_ps(Fouti + m2 + k, fi2o);
            }
        }
#endif
#if defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx10_1_256)
        {
        const typename simd_vec<C, 8>::f v_half = _mm256_set1_ps(0.5f), v_epi3 = _mm256_set1_ps(epi3_i);
            if (k < m)
            {
                const typename simd_vec<C, 8>::mask mask = simd_tail_mask8<C>(m - k);
                const size_t rem = static_cast<size_t>(m - k);
                typename simd_vec<C, 8>::f fr0 = _mm256_maskz_loadu_ps(mask, Foutr + k);
                typename simd_vec<C, 8>::f fi0 = _mm256_maskz_loadu_ps(mask, Fouti + k);
                typename simd_vec<C, 8>::f fr1 = _mm256_maskz_loadu_ps(mask, Foutr + m + k);
                typename simd_vec<C, 8>::f fi1 = _mm256_maskz_loadu_ps(mask, Fouti + m + k);
                typename simd_vec<C, 8>::f fr2 = _mm256_maskz_loadu_ps(mask, Foutr + m2 + k);
                typename simd_vec<C, 8>::f fi2 = _mm256_maskz_loadu_ps(mask, Fouti + m2 + k);

                typename simd_vec<C, 8>::f tw1r = simd_tail_load8_strided_ps<C>(tw1_r + k * fstride, fstride, rem);
                typename simd_vec<C, 8>::f tw1i = simd_tail_load8_strided_ps<C>(tw1_i + k * fstride, fstride, rem);
                typename simd_vec<C, 8>::f tw2r = simd_tail_load8_strided_ps<C>(tw1_r + 2 * k * fstride, 2 * fstride, rem);
                typename simd_vec<C, 8>::f tw2i = simd_tail_load8_strided_ps<C>(tw1_i + 2 * k * fstride, 2 * fstride, rem);

                typename simd_vec<C, 8>::f sr1 = simd_fmsub_ps<C>(fr1, tw1r, _mm256_mul_ps(fi1, tw1i));
                typename simd_vec<C, 8>::f si1 = simd_fmadd_ps<C>(fr1, tw1i, _mm256_mul_ps(fi1, tw1r));
                typename simd_vec<C, 8>::f sr2 = simd_fmsub_ps<C>(fr2, tw2r, _mm256_mul_ps(fi2, tw2i));
                typename simd_vec<C, 8>::f si2 = simd_fmadd_ps<C>(fr2, tw2i, _mm256_mul_ps(fi2, tw2r));

                typename simd_vec<C, 8>::f sr3 = _mm256_add_ps(sr1, sr2);
                typename simd_vec<C, 8>::f si3 = _mm256_add_ps(si1, si2);
                typename simd_vec<C, 8>::f sr0 = _mm256_sub_ps(sr1, sr2);
                typename simd_vec<C, 8>::f si0 = _mm256_sub_ps(si1, si2);

                typename simd_vec<C, 8>::f fr1o = _mm256_sub_ps(fr0, _mm256_mul_ps(sr3, v_half));
                typename simd_vec<C, 8>::f fi1o = _mm256_sub_ps(fi0, _mm256_mul_ps(si3, v_half));

                sr0 = _mm256_mul_ps(sr0, v_epi3);
                si0 = _mm256_mul_ps(si0, v_epi3);

                typename simd_vec<C, 8>::f fr0o = _mm256_add_ps(fr0, sr3);
                typename simd_vec<C, 8>::f fi0o = _mm256_add_ps(fi0, si3);

                typename simd_vec<C, 8>::f fr2o = _mm256_add_ps(fr1o, si0);
                typename simd_vec<C, 8>::f fi2o = _mm256_sub_ps(fi1o, sr0);

                fr1o = _mm256_sub_ps(fr1o, si0);
                fi1o = _mm256_add_ps(fi1o, sr0);

                _mm256_mask_storeu_ps(Foutr + k, mask, fr0o);
                _mm256_mask_storeu_ps(Fouti + k, mask, fi0o);
                _mm256_mask_storeu_ps(Foutr + m + k, mask, fr1o);
                _mm256_mask_storeu_ps(Fouti + m + k, mask, fi1o);
                _mm256_mask_storeu_ps(Foutr + m2 + k, mask, fr2o);
                _mm256_mask_storeu_ps(Fouti + m2 + k, mask, fi2o);
            }
            return;
        }
#endif
#if defined(COSYVOICE_HAS_SSE42) || defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2)
        if constexpr (C.sse42)
        {
            for (const typename simd_vec<C, 4>::f v_half_128 = _mm_set_ps1(0.5f), v_epi3_128 = _mm_set_ps1(epi3_i);
                k + 3 < m; k += 4)
            {
                typename simd_vec<C, 4>::f fr0 = _mm_loadu_ps(Foutr + k);
                typename simd_vec<C, 4>::f fi0 = _mm_loadu_ps(Fouti + k);
                typename simd_vec<C, 4>::f fr1 = _mm_loadu_ps(Foutr + m + k);
                typename simd_vec<C, 4>::f fi1 = _mm_loadu_ps(Fouti + m + k);
                typename simd_vec<C, 4>::f fr2 = _mm_loadu_ps(Foutr + m2 + k);
                typename simd_vec<C, 4>::f fi2 = _mm_loadu_ps(Fouti + m2 + k);

                typename simd_vec<C, 4>::f tw1r;
                typename simd_vec<C, 4>::f tw1i;
                typename simd_vec<C, 4>::f tw2r;
                typename simd_vec<C, 4>::f tw2i;
                if (fstride == 1)
                {
                    tw1r = _mm_loadu_ps(tw1_r + k);
                    tw1i = _mm_loadu_ps(tw1_i + k);

#if defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
                    if constexpr (C.avx2 || C.avx10_1_256)
                    {
                        const typename simd_vec<C, 4>::i idx2 = _mm_setr_epi32(k * 2, (k + 1) * 2, (k + 2) * 2, (k + 3) * 2);
                        tw2r = _mm_i32gather_ps(tw1_r, idx2, 4);
                        tw2i = _mm_i32gather_ps(tw1_i, idx2, 4);
                    }
                    else
#endif
                    {
                        tw2r = simd_load4_strided_ps<C>(tw1_r + 2 * k, 2);
                        tw2i = simd_load4_strided_ps<C>(tw1_i + 2 * k, 2);
                    }
                }
                else
                {
#if defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
                    if constexpr (C.avx2 || C.avx10_1_256)
                    {
                        const typename simd_vec<C, 4>::i idx1 = _mm_setr_epi32(
                            k * fstride, (k + 1) * fstride, (k + 2) * fstride, (k + 3) * fstride);
                        const typename simd_vec<C, 4>::i idx2 = _mm_mullo_epi32(idx1, _mm_set1_epi32(2));

                        tw1r = _mm_i32gather_ps(tw1_r, idx1, 4);
                        tw1i = _mm_i32gather_ps(tw1_i, idx1, 4);
                        tw2r = _mm_i32gather_ps(tw1_r, idx2, 4);
                        tw2i = _mm_i32gather_ps(tw1_i, idx2, 4);
                    }
                    else
#endif
                    {
                        tw1r = simd_load4_strided_ps<C>(tw1_r + k * fstride, fstride);
                        tw1i = simd_load4_strided_ps<C>(tw1_i + k * fstride, fstride);
                        tw2r = simd_load4_strided_ps<C>(tw1_r + 2 * k * fstride, 2 * fstride);
                        tw2i = simd_load4_strided_ps<C>(tw1_i + 2 * k * fstride, 2 * fstride);
                    }
                }

                typename simd_vec<C, 4>::f sr1 = simd_fmsub_ps<C>(fr1, tw1r, _mm_mul_ps(fi1, tw1i));
                typename simd_vec<C, 4>::f si1 = simd_fmadd_ps<C>(fr1, tw1i, _mm_mul_ps(fi1, tw1r));
                typename simd_vec<C, 4>::f sr2 = simd_fmsub_ps<C>(fr2, tw2r, _mm_mul_ps(fi2, tw2i));
                typename simd_vec<C, 4>::f si2 = simd_fmadd_ps<C>(fr2, tw2i, _mm_mul_ps(fi2, tw2r));

                typename simd_vec<C, 4>::f sr3 = _mm_add_ps(sr1, sr2);
                typename simd_vec<C, 4>::f si3 = _mm_add_ps(si1, si2);
                typename simd_vec<C, 4>::f sr0 = _mm_sub_ps(sr1, sr2);
                typename simd_vec<C, 4>::f si0 = _mm_sub_ps(si1, si2);

                typename simd_vec<C, 4>::f fr1o = _mm_sub_ps(fr0, _mm_mul_ps(sr3, v_half_128));
                typename simd_vec<C, 4>::f fi1o = _mm_sub_ps(fi0, _mm_mul_ps(si3, v_half_128));

                sr0 = _mm_mul_ps(sr0, v_epi3_128);
                si0 = _mm_mul_ps(si0, v_epi3_128);

                typename simd_vec<C, 4>::f fr0o = _mm_add_ps(fr0, sr3);
                typename simd_vec<C, 4>::f fi0o = _mm_add_ps(fi0, si3);

                typename simd_vec<C, 4>::f fr2o = _mm_add_ps(fr1o, si0);
                typename simd_vec<C, 4>::f fi2o = _mm_sub_ps(fi1o, sr0);

                fr1o = _mm_sub_ps(fr1o, si0);
                fi1o = _mm_add_ps(fi1o, sr0);

                _mm_storeu_ps(Foutr + k, fr0o);
                _mm_storeu_ps(Fouti + k, fi0o);
                _mm_storeu_ps(Foutr + m + k, fr1o);
                _mm_storeu_ps(Fouti + m + k, fi1o);
                _mm_storeu_ps(Foutr + m2 + k, fr2o);
                _mm_storeu_ps(Fouti + m2 + k, fi2o);
            }
        }
#endif

        for (; k < m; ++k)
        {
            float tw1r = tw1_r[k * fstride];
            float tw1i = tw1_i[k * fstride];
            float tw2r = tw1_r[k * fstride * 2];
            float tw2i = tw1_i[k * fstride * 2];

            float fr0 = Foutr[k];
            float fi0 = Fouti[k];
            float fr1 = Foutr[m + k];
            float fi1 = Fouti[m + k];
            float fr2 = Foutr[m2 + k];
            float fi2 = Fouti[m2 + k];

            float sr1 = fr1 * tw1r - fi1 * tw1i;
            float si1 = fr1 * tw1i + fi1 * tw1r;
            float sr2 = fr2 * tw2r - fi2 * tw2i;
            float si2 = fr2 * tw2i + fi2 * tw2r;

            float sr3 = sr1 + sr2;
            float si3 = si1 + si2;
            float sr0 = sr1 - sr2;
            float si0 = si1 - si2;

            float fr1o = fr0 - sr3 * 0.5f;
            float fi1o = fi0 - si3 * 0.5f;

            sr0 *= epi3_i;
            si0 *= epi3_i;

            Foutr[k] = fr0 + sr3;
            Fouti[k] = fi0 + si3;
            Foutr[m + k] = fr1o - si0;
            Fouti[m + k] = fi1o + sr0;
            Foutr[m2 + k] = fr1o + si0;
            Fouti[m2 + k] = fi1o - sr0;
        }
    }
}

template<simd_caps C>
void fft_bfly5_kernel<C>::run(float* Foutr, float* Fouti, int fstride, const float* tw_r, const float* tw_i, int m)
{
    int u = 0;

    float yar = tw_r[fstride * m];
    float yai = tw_i[fstride * m];
    float ybr = tw_r[fstride * 2 * m];
    float ybi = tw_i[fstride * 2 * m];

    float* F0r = Foutr, * F1r = Foutr + m, * F2r = Foutr + 2 * m,
        * F3r = Foutr + 3 * m, * F4r = Foutr + 4 * m;
    float* F0i = Fouti, * F1i = Fouti + m, * F2i = Fouti + 2 * m,
        * F3i = Fouti + 3 * m, * F4i = Fouti + 4 * m;

#if defined(COSYVOICE_HAS_AVX512)
    if constexpr (C.avx512)
    {
        const typename simd_vec<C, 16>::f v_yar = _mm512_set1_ps(yar),
            v_yai = _mm512_set1_ps(yai),
            v_ybr = _mm512_set1_ps(ybr),
            v_ybi = _mm512_set1_ps(ybi);
        while (u < m)
        {
            const typename simd_vec<C, 16>::mask mask = (u + 16 <= m) ? static_cast<typename simd_vec<C, 16>::mask>(0xFFFFu) : simd_tail_mask16<C>(m - u);
            const size_t rem = static_cast<size_t>(m - u) < 16 ? static_cast<size_t>(m - u) : 16;

            typename simd_vec<C, 16>::f tw1r = simd_tail_load16_strided_ps<C>(tw_r + u * fstride, fstride, rem);
            typename simd_vec<C, 16>::f tw1i = simd_tail_load16_strided_ps<C>(tw_i + u * fstride, fstride, rem);
            typename simd_vec<C, 16>::f tw2r = simd_tail_load16_strided_ps<C>(tw_r + 2 * u * fstride, 2 * fstride, rem);
            typename simd_vec<C, 16>::f tw2i = simd_tail_load16_strided_ps<C>(tw_i + 2 * u * fstride, 2 * fstride, rem);
            typename simd_vec<C, 16>::f tw3r = simd_tail_load16_strided_ps<C>(tw_r + 3 * u * fstride, 3 * fstride, rem);
            typename simd_vec<C, 16>::f tw3i = simd_tail_load16_strided_ps<C>(tw_i + 3 * u * fstride, 3 * fstride, rem);
            typename simd_vec<C, 16>::f tw4r = simd_tail_load16_strided_ps<C>(tw_r + 4 * u * fstride, 4 * fstride, rem);
            typename simd_vec<C, 16>::f tw4i = simd_tail_load16_strided_ps<C>(tw_i + 4 * u * fstride, 4 * fstride, rem);

            typename simd_vec<C, 16>::f f0r = _mm512_maskz_loadu_ps(mask, F0r + u);
            typename simd_vec<C, 16>::f f0i = _mm512_maskz_loadu_ps(mask, F0i + u);
            typename simd_vec<C, 16>::f f1r = _mm512_maskz_loadu_ps(mask, F1r + u);
            typename simd_vec<C, 16>::f f1i = _mm512_maskz_loadu_ps(mask, F1i + u);
            typename simd_vec<C, 16>::f f2r = _mm512_maskz_loadu_ps(mask, F2r + u);
            typename simd_vec<C, 16>::f f2i = _mm512_maskz_loadu_ps(mask, F2i + u);
            typename simd_vec<C, 16>::f f3r = _mm512_maskz_loadu_ps(mask, F3r + u);
            typename simd_vec<C, 16>::f f3i = _mm512_maskz_loadu_ps(mask, F3i + u);
            typename simd_vec<C, 16>::f f4r = _mm512_maskz_loadu_ps(mask, F4r + u);
            typename simd_vec<C, 16>::f f4i = _mm512_maskz_loadu_ps(mask, F4i + u);

            CMUL<C>(f1r, f1i, tw1r, tw1i);
            CMUL<C>(f2r, f2i, tw2r, tw2i);
            CMUL<C>(f3r, f3i, tw3r, tw3i);
            CMUL<C>(f4r, f4i, tw4r, tw4i);

            typename simd_vec<C, 16>::f s7r = _mm512_add_ps(f1r, f4r);
            typename simd_vec<C, 16>::f s7i = _mm512_add_ps(f1i, f4i);
            typename simd_vec<C, 16>::f s10r = _mm512_sub_ps(f1r, f4r);
            typename simd_vec<C, 16>::f s10i = _mm512_sub_ps(f1i, f4i);
            typename simd_vec<C, 16>::f s8r = _mm512_add_ps(f2r, f3r);
            typename simd_vec<C, 16>::f s8i = _mm512_add_ps(f2i, f3i);
            typename simd_vec<C, 16>::f s9r = _mm512_sub_ps(f2r, f3r);
            typename simd_vec<C, 16>::f s9i = _mm512_sub_ps(f2i, f3i);

            typename simd_vec<C, 16>::f s5r = simd_fmadd_ps<C>(s7r, v_yar, f0r);
            s5r = simd_fmadd_ps<C>(s8r, v_ybr, s5r);
            typename simd_vec<C, 16>::f s5i = simd_fmadd_ps<C>(s7i, v_yar, f0i);
            s5i = simd_fmadd_ps<C>(s8i, v_ybr, s5i);

            typename simd_vec<C, 16>::f s6r = simd_fmadd_ps<C>(s10i, v_yai, _mm512_mul_ps(s9i, v_ybi));
            typename simd_vec<C, 16>::f s6i = _mm512_sub_ps(_mm512_setzero_ps(), simd_fmadd_ps<C>(s10r, v_yai, _mm512_mul_ps(s9r, v_ybi)));

            typename simd_vec<C, 16>::f t1r = _mm512_sub_ps(s5r, s6r);
            typename simd_vec<C, 16>::f t1i = _mm512_sub_ps(s5i, s6i);
            typename simd_vec<C, 16>::f t4r = _mm512_add_ps(s5r, s6r);
            typename simd_vec<C, 16>::f t4i = _mm512_add_ps(s5i, s6i);

            typename simd_vec<C, 16>::f s11r = simd_fmadd_ps<C>(s7r, v_ybr, f0r);
            s11r = simd_fmadd_ps<C>(s8r, v_yar, s11r);
            typename simd_vec<C, 16>::f s11i = simd_fmadd_ps<C>(s7i, v_ybr, f0i);
            s11i = simd_fmadd_ps<C>(s8i, v_yar, s11i);

            typename simd_vec<C, 16>::f tmp = _mm512_add_ps(s7r, s8r);
            f0r = _mm512_add_ps(f0r, tmp);
            tmp = _mm512_add_ps(s7i, s8i);
            f0i = _mm512_add_ps(f0i, tmp);

            typename simd_vec<C, 16>::f s12r = simd_fmsub_ps<C>(s9i, v_yai, _mm512_mul_ps(s10i, v_ybi));
            typename simd_vec<C, 16>::f s12i = simd_fmsub_ps<C>(s10r, v_ybi, _mm512_mul_ps(s9r, v_yai));

            typename simd_vec<C, 16>::f t2r = _mm512_add_ps(s11r, s12r);
            typename simd_vec<C, 16>::f t2i = _mm512_add_ps(s11i, s12i);
            typename simd_vec<C, 16>::f t3r = _mm512_sub_ps(s11r, s12r);
            typename simd_vec<C, 16>::f t3i = _mm512_sub_ps(s11i, s12i);

            _mm512_mask_storeu_ps(F0r + u, mask, f0r);
            _mm512_mask_storeu_ps(F0i + u, mask, f0i);
            _mm512_mask_storeu_ps(F1r + u, mask, t1r);
            _mm512_mask_storeu_ps(F1i + u, mask, t1i);
            _mm512_mask_storeu_ps(F2r + u, mask, t2r);
            _mm512_mask_storeu_ps(F2i + u, mask, t2i);
            _mm512_mask_storeu_ps(F3r + u, mask, t3r);
            _mm512_mask_storeu_ps(F3i + u, mask, t3i);
            _mm512_mask_storeu_ps(F4r + u, mask, t4r);
            _mm512_mask_storeu_ps(F4i + u, mask, t4i);

            u += 16;
        }
        return;
    }
    else
#endif
    {
#if defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx || C.avx10_1_256)
        {
            for (const typename simd_vec<C, 8>::f v_yar = _mm256_set1_ps(yar),
                v_yai = _mm256_set1_ps(yai),
                v_ybr = _mm256_set1_ps(ybr),
                v_ybi = _mm256_set1_ps(ybi); u + 7 < m; u += 8) {
                typename simd_vec<C, 8>::f tw1r, tw1i, tw2r, tw2i, tw3r, tw3i, tw4r, tw4i;
#if defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
                if constexpr (C.avx2 || C.avx10_1_256)
                {
                    typename simd_vec<C, 8>::i idx1 = _mm256_setr_epi32(
                        u * fstride, (u + 1) * fstride, (u + 2) * fstride, (u + 3) * fstride,
                        (u + 4) * fstride, (u + 5) * fstride, (u + 6) * fstride, (u + 7) * fstride
                    );
                    typename simd_vec<C, 8>::i idx2 = _mm256_mullo_epi32(idx1, _mm256_set1_epi32(2));
                    typename simd_vec<C, 8>::i idx3 = _mm256_mullo_epi32(idx1, _mm256_set1_epi32(3));
                    typename simd_vec<C, 8>::i idx4 = _mm256_mullo_epi32(idx1, _mm256_set1_epi32(4));

                    tw1r = _mm256_i32gather_ps(tw_r, idx1, 4);
                    tw1i = _mm256_i32gather_ps(tw_i, idx1, 4);
                    tw2r = _mm256_i32gather_ps(tw_r, idx2, 4);
                    tw2i = _mm256_i32gather_ps(tw_i, idx2, 4);
                    tw3r = _mm256_i32gather_ps(tw_r, idx3, 4);
                    tw3i = _mm256_i32gather_ps(tw_i, idx3, 4);
                    tw4r = _mm256_i32gather_ps(tw_r, idx4, 4);
                    tw4i = _mm256_i32gather_ps(tw_i, idx4, 4);
                }
                else
#endif
                {
                    if (fstride == 1)
                    {
                        tw1r = _mm256_loadu_ps(tw_r + u);
                        tw1i = _mm256_loadu_ps(tw_i + u);
                    }
                    else
                    {
                        tw1r = simd_load8_strided_ps<C>(tw_r + u * fstride, fstride);
                        tw1i = simd_load8_strided_ps<C>(tw_i + u * fstride, fstride);
                    }
                    tw2r = simd_load8_strided_ps<C>(tw_r + 2 * u * fstride, 2 * fstride);
                    tw2i = simd_load8_strided_ps<C>(tw_i + 2 * u * fstride, 2 * fstride);
                    tw3r = simd_load8_strided_ps<C>(tw_r + 3 * u * fstride, 3 * fstride);
                    tw3i = simd_load8_strided_ps<C>(tw_i + 3 * u * fstride, 3 * fstride);
                    tw4r = simd_load8_strided_ps<C>(tw_r + 4 * u * fstride, 4 * fstride);
                    tw4i = simd_load8_strided_ps<C>(tw_i + 4 * u * fstride, 4 * fstride);
                }

                typename simd_vec<C, 8>::f f0r = _mm256_loadu_ps(F0r + u);
                typename simd_vec<C, 8>::f f0i = _mm256_loadu_ps(F0i + u);
                typename simd_vec<C, 8>::f f1r = _mm256_loadu_ps(F1r + u);
                typename simd_vec<C, 8>::f f1i = _mm256_loadu_ps(F1i + u);
                typename simd_vec<C, 8>::f f2r = _mm256_loadu_ps(F2r + u);
                typename simd_vec<C, 8>::f f2i = _mm256_loadu_ps(F2i + u);
                typename simd_vec<C, 8>::f f3r = _mm256_loadu_ps(F3r + u);
                typename simd_vec<C, 8>::f f3i = _mm256_loadu_ps(F3i + u);
                typename simd_vec<C, 8>::f f4r = _mm256_loadu_ps(F4r + u);
                typename simd_vec<C, 8>::f f4i = _mm256_loadu_ps(F4i + u);

                CMUL<C>(f1r, f1i, tw1r, tw1i);
                CMUL<C>(f2r, f2i, tw2r, tw2i);
                CMUL<C>(f3r, f3i, tw3r, tw3i);
                CMUL<C>(f4r, f4i, tw4r, tw4i);

                typename simd_vec<C, 8>::f s7r = _mm256_add_ps(f1r, f4r);
                typename simd_vec<C, 8>::f s7i = _mm256_add_ps(f1i, f4i);
                typename simd_vec<C, 8>::f s10r = _mm256_sub_ps(f1r, f4r);
                typename simd_vec<C, 8>::f s10i = _mm256_sub_ps(f1i, f4i);
                typename simd_vec<C, 8>::f s8r = _mm256_add_ps(f2r, f3r);
                typename simd_vec<C, 8>::f s8i = _mm256_add_ps(f2i, f3i);
                typename simd_vec<C, 8>::f s9r = _mm256_sub_ps(f2r, f3r);
                typename simd_vec<C, 8>::f s9i = _mm256_sub_ps(f2i, f3i);

                typename simd_vec<C, 8>::f s5r = simd_fmadd_ps<C>(s7r, v_yar, f0r);
                s5r = simd_fmadd_ps<C>(s8r, v_ybr, s5r);
                typename simd_vec<C, 8>::f s5i = simd_fmadd_ps<C>(s7i, v_yar, f0i);
                s5i = simd_fmadd_ps<C>(s8i, v_ybr, s5i);

                typename simd_vec<C, 8>::f s6r = simd_fmadd_ps<C>(s10i, v_yai, _mm256_mul_ps(s9i, v_ybi));
                typename simd_vec<C, 8>::f s6i = _mm256_sub_ps(_mm256_setzero_ps(), simd_fmadd_ps<C>(s10r, v_yai, _mm256_mul_ps(s9r, v_ybi)));

                typename simd_vec<C, 8>::f t1r = _mm256_sub_ps(s5r, s6r);
                typename simd_vec<C, 8>::f t1i = _mm256_sub_ps(s5i, s6i);
                typename simd_vec<C, 8>::f t4r = _mm256_add_ps(s5r, s6r);
                typename simd_vec<C, 8>::f t4i = _mm256_add_ps(s5i, s6i);

                typename simd_vec<C, 8>::f s11r = simd_fmadd_ps<C>(s7r, v_ybr, f0r);
                s11r = simd_fmadd_ps<C>(s8r, v_yar, s11r);
                typename simd_vec<C, 8>::f s11i = simd_fmadd_ps<C>(s7i, v_ybr, f0i);
                s11i = simd_fmadd_ps<C>(s8i, v_yar, s11i);

                typename simd_vec<C, 8>::f tmp = _mm256_add_ps(s7r, s8r);
                f0r = _mm256_add_ps(f0r, tmp);
                tmp = _mm256_add_ps(s7i, s8i);
                f0i = _mm256_add_ps(f0i, tmp);

                typename simd_vec<C, 8>::f s12r = simd_fmsub_ps<C>(s9i, v_yai, _mm256_mul_ps(s10i, v_ybi));
                typename simd_vec<C, 8>::f s12i = simd_fmsub_ps<C>(s10r, v_ybi, _mm256_mul_ps(s9r, v_yai));

                typename simd_vec<C, 8>::f t2r = _mm256_add_ps(s11r, s12r);
                typename simd_vec<C, 8>::f t2i = _mm256_add_ps(s11i, s12i);
                typename simd_vec<C, 8>::f t3r = _mm256_sub_ps(s11r, s12r);
                typename simd_vec<C, 8>::f t3i = _mm256_sub_ps(s11i, s12i);

                _mm256_storeu_ps(F0r + u, f0r);
                _mm256_storeu_ps(F0i + u, f0i);
                _mm256_storeu_ps(F1r + u, t1r);
                _mm256_storeu_ps(F1i + u, t1i);
                _mm256_storeu_ps(F2r + u, t2r);
                _mm256_storeu_ps(F2i + u, t2i);
                _mm256_storeu_ps(F3r + u, t3r);
                _mm256_storeu_ps(F3i + u, t3i);
                _mm256_storeu_ps(F4r + u, t4r);
                _mm256_storeu_ps(F4i + u, t4i);
            }
        }
#endif
#if defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx10_1_256)
        {
        const typename simd_vec<C, 8>::f v_yar = _mm256_set1_ps(yar),
            v_yai = _mm256_set1_ps(yai),
            v_ybr = _mm256_set1_ps(ybr),
            v_ybi = _mm256_set1_ps(ybi);
            if (u < m)
            {
                const typename simd_vec<C, 8>::mask mask = simd_tail_mask8<C>(m - u);
                const size_t rem = static_cast<size_t>(m - u);
                typename simd_vec<C, 8>::f tw1r = simd_tail_load8_strided_ps<C>(tw_r + u * fstride, fstride, rem);
                typename simd_vec<C, 8>::f tw1i = simd_tail_load8_strided_ps<C>(tw_i + u * fstride, fstride, rem);
                typename simd_vec<C, 8>::f tw2r = simd_tail_load8_strided_ps<C>(tw_r + 2 * u * fstride, 2 * fstride, rem);
                typename simd_vec<C, 8>::f tw2i = simd_tail_load8_strided_ps<C>(tw_i + 2 * u * fstride, 2 * fstride, rem);
                typename simd_vec<C, 8>::f tw3r = simd_tail_load8_strided_ps<C>(tw_r + 3 * u * fstride, 3 * fstride, rem);
                typename simd_vec<C, 8>::f tw3i = simd_tail_load8_strided_ps<C>(tw_i + 3 * u * fstride, 3 * fstride, rem);
                typename simd_vec<C, 8>::f tw4r = simd_tail_load8_strided_ps<C>(tw_r + 4 * u * fstride, 4 * fstride, rem);
                typename simd_vec<C, 8>::f tw4i = simd_tail_load8_strided_ps<C>(tw_i + 4 * u * fstride, 4 * fstride, rem);

                typename simd_vec<C, 8>::f f0r = _mm256_maskz_loadu_ps(mask, F0r + u);
                typename simd_vec<C, 8>::f f0i = _mm256_maskz_loadu_ps(mask, F0i + u);
                typename simd_vec<C, 8>::f f1r = _mm256_maskz_loadu_ps(mask, F1r + u);
                typename simd_vec<C, 8>::f f1i = _mm256_maskz_loadu_ps(mask, F1i + u);
                typename simd_vec<C, 8>::f f2r = _mm256_maskz_loadu_ps(mask, F2r + u);
                typename simd_vec<C, 8>::f f2i = _mm256_maskz_loadu_ps(mask, F2i + u);
                typename simd_vec<C, 8>::f f3r = _mm256_maskz_loadu_ps(mask, F3r + u);
                typename simd_vec<C, 8>::f f3i = _mm256_maskz_loadu_ps(mask, F3i + u);
                typename simd_vec<C, 8>::f f4r = _mm256_maskz_loadu_ps(mask, F4r + u);
                typename simd_vec<C, 8>::f f4i = _mm256_maskz_loadu_ps(mask, F4i + u);

                CMUL<C>(f1r, f1i, tw1r, tw1i);
                CMUL<C>(f2r, f2i, tw2r, tw2i);
                CMUL<C>(f3r, f3i, tw3r, tw3i);
                CMUL<C>(f4r, f4i, tw4r, tw4i);

                typename simd_vec<C, 8>::f s7r = _mm256_add_ps(f1r, f4r);
                typename simd_vec<C, 8>::f s7i = _mm256_add_ps(f1i, f4i);
                typename simd_vec<C, 8>::f s10r = _mm256_sub_ps(f1r, f4r);
                typename simd_vec<C, 8>::f s10i = _mm256_sub_ps(f1i, f4i);
                typename simd_vec<C, 8>::f s8r = _mm256_add_ps(f2r, f3r);
                typename simd_vec<C, 8>::f s8i = _mm256_add_ps(f2i, f3i);
                typename simd_vec<C, 8>::f s9r = _mm256_sub_ps(f2r, f3r);
                typename simd_vec<C, 8>::f s9i = _mm256_sub_ps(f2i, f3i);

                typename simd_vec<C, 8>::f s5r = simd_fmadd_ps<C>(s7r, v_yar, f0r);
                s5r = simd_fmadd_ps<C>(s8r, v_ybr, s5r);
                typename simd_vec<C, 8>::f s5i = simd_fmadd_ps<C>(s7i, v_yar, f0i);
                s5i = simd_fmadd_ps<C>(s8i, v_ybr, s5i);

                typename simd_vec<C, 8>::f s6r = simd_fmadd_ps<C>(s10i, v_yai, _mm256_mul_ps(s9i, v_ybi));
                typename simd_vec<C, 8>::f s6i = _mm256_sub_ps(_mm256_setzero_ps(), simd_fmadd_ps<C>(s10r, v_yai, _mm256_mul_ps(s9r, v_ybi)));

                typename simd_vec<C, 8>::f t1r = _mm256_sub_ps(s5r, s6r);
                typename simd_vec<C, 8>::f t1i = _mm256_sub_ps(s5i, s6i);
                typename simd_vec<C, 8>::f t4r = _mm256_add_ps(s5r, s6r);
                typename simd_vec<C, 8>::f t4i = _mm256_add_ps(s5i, s6i);

                typename simd_vec<C, 8>::f s11r = simd_fmadd_ps<C>(s7r, v_ybr, f0r);
                s11r = simd_fmadd_ps<C>(s8r, v_yar, s11r);
                typename simd_vec<C, 8>::f s11i = simd_fmadd_ps<C>(s7i, v_ybr, f0i);
                s11i = simd_fmadd_ps<C>(s8i, v_yar, s11i);

                typename simd_vec<C, 8>::f tmp = _mm256_add_ps(s7r, s8r);
                f0r = _mm256_add_ps(f0r, tmp);
                tmp = _mm256_add_ps(s7i, s8i);
                f0i = _mm256_add_ps(f0i, tmp);

                typename simd_vec<C, 8>::f s12r = simd_fmsub_ps<C>(s9i, v_yai, _mm256_mul_ps(s10i, v_ybi));
                typename simd_vec<C, 8>::f s12i = simd_fmsub_ps<C>(s10r, v_ybi, _mm256_mul_ps(s9r, v_yai));

                typename simd_vec<C, 8>::f t2r = _mm256_add_ps(s11r, s12r);
                typename simd_vec<C, 8>::f t2i = _mm256_add_ps(s11i, s12i);
                typename simd_vec<C, 8>::f t3r = _mm256_sub_ps(s11r, s12r);
                typename simd_vec<C, 8>::f t3i = _mm256_sub_ps(s11i, s12i);

                _mm256_mask_storeu_ps(F0r + u, mask, f0r);
                _mm256_mask_storeu_ps(F0i + u, mask, f0i);
                _mm256_mask_storeu_ps(F1r + u, mask, t1r);
                _mm256_mask_storeu_ps(F1i + u, mask, t1i);
                _mm256_mask_storeu_ps(F2r + u, mask, t2r);
                _mm256_mask_storeu_ps(F2i + u, mask, t2i);
                _mm256_mask_storeu_ps(F3r + u, mask, t3r);
                _mm256_mask_storeu_ps(F3i + u, mask, t3i);
                _mm256_mask_storeu_ps(F4r + u, mask, t4r);
                _mm256_mask_storeu_ps(F4i + u, mask, t4i);
            }
            return;
        }
#endif
#if defined(COSYVOICE_HAS_SSE42) || defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2)
        if constexpr (C.sse42)
        {
            for (const typename simd_vec<C, 4>::f v_yar = _mm_set_ps1(yar),
                v_yai = _mm_set_ps1(yai),
                v_ybr = _mm_set_ps1(ybr),
                v_ybi = _mm_set_ps1(ybi); u + 3 < m; u += 4)
            {
                typename simd_vec<C, 4>::f tw1r, tw1i, tw2r, tw2i, tw3r, tw3i, tw4r, tw4i;
#if defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
                if constexpr (C.avx2 || C.avx10_1_256)
                {
                    typename simd_vec<C, 4>::i idx1 = _mm_setr_epi32(
                        u * fstride, (u + 1) * fstride, (u + 2) * fstride, (u + 3) * fstride);
                    tw1r = _mm_i32gather_ps(tw_r, idx1, 4);
                    tw1i = _mm_i32gather_ps(tw_i, idx1, 4);

                    typename simd_vec<C, 4>::i idx2 = _mm_mullo_epi32(idx1, _mm_set1_epi32(2));
                    tw2r = _mm_i32gather_ps(tw_r, idx2, 4);
                    tw2i = _mm_i32gather_ps(tw_i, idx2, 4);

                    typename simd_vec<C, 4>::i idx3 = _mm_mullo_epi32(idx1, _mm_set1_epi32(3));
                    tw3r = _mm_i32gather_ps(tw_r, idx3, 4);
                    tw3i = _mm_i32gather_ps(tw_i, idx3, 4);

                    typename simd_vec<C, 4>::i idx4 = _mm_mullo_epi32(idx1, _mm_set1_epi32(4));
                    tw4r = _mm_i32gather_ps(tw_r, idx4, 4);
                    tw4i = _mm_i32gather_ps(tw_i, idx4, 4);
                }
                else
#endif
                {
                    tw1r = simd_load4_strided_ps<C>(tw_r + u * fstride, fstride);
                    tw1i = simd_load4_strided_ps<C>(tw_i + u * fstride, fstride);

                    tw2r = simd_load4_strided_ps<C>(tw_r + 2 * u * fstride, 2 * fstride);
                    tw2i = simd_load4_strided_ps<C>(tw_i + 2 * u * fstride, 2 * fstride);

                    tw3r = simd_load4_strided_ps<C>(tw_r + 3 * u * fstride, 3 * fstride);
                    tw3i = simd_load4_strided_ps<C>(tw_i + 3 * u * fstride, 3 * fstride);

                    tw4r = simd_load4_strided_ps<C>(tw_r + 4 * u * fstride, 4 * fstride);
                    tw4i = simd_load4_strided_ps<C>(tw_i + 4 * u * fstride, 4 * fstride);
                }

                typename simd_vec<C, 4>::f f0r = _mm_loadu_ps(F0r + u);
                typename simd_vec<C, 4>::f f0i = _mm_loadu_ps(F0i + u);
                typename simd_vec<C, 4>::f f1r = _mm_loadu_ps(F1r + u);
                typename simd_vec<C, 4>::f f1i = _mm_loadu_ps(F1i + u);
                typename simd_vec<C, 4>::f f2r = _mm_loadu_ps(F2r + u);
                typename simd_vec<C, 4>::f f2i = _mm_loadu_ps(F2i + u);
                typename simd_vec<C, 4>::f f3r = _mm_loadu_ps(F3r + u);
                typename simd_vec<C, 4>::f f3i = _mm_loadu_ps(F3i + u);
                typename simd_vec<C, 4>::f f4r = _mm_loadu_ps(F4r + u);
                typename simd_vec<C, 4>::f f4i = _mm_loadu_ps(F4i + u);

                CMUL<C>(f1r, f1i, tw1r, tw1i);
                CMUL<C>(f2r, f2i, tw2r, tw2i);
                CMUL<C>(f3r, f3i, tw3r, tw3i);
                CMUL<C>(f4r, f4i, tw4r, tw4i);

                typename simd_vec<C, 4>::f s7r = _mm_add_ps(f1r, f4r);
                typename simd_vec<C, 4>::f s7i = _mm_add_ps(f1i, f4i);
                typename simd_vec<C, 4>::f s10r = _mm_sub_ps(f1r, f4r);
                typename simd_vec<C, 4>::f s10i = _mm_sub_ps(f1i, f4i);
                typename simd_vec<C, 4>::f s8r = _mm_add_ps(f2r, f3r);
                typename simd_vec<C, 4>::f s8i = _mm_add_ps(f2i, f3i);
                typename simd_vec<C, 4>::f s9r = _mm_sub_ps(f2r, f3r);
                typename simd_vec<C, 4>::f s9i = _mm_sub_ps(f2i, f3i);

                typename simd_vec<C, 4>::f s5r = simd_fmadd_ps<C>(s7r, v_yar, f0r);
                s5r = simd_fmadd_ps<C>(s8r, v_ybr, s5r);
                typename simd_vec<C, 4>::f s5i = simd_fmadd_ps<C>(s7i, v_yar, f0i);
                s5i = simd_fmadd_ps<C>(s8i, v_ybr, s5i);

                typename simd_vec<C, 4>::f s6r = simd_fmadd_ps<C>(s10i, v_yai, _mm_mul_ps(s9i, v_ybi));
                typename simd_vec<C, 4>::f s6i = _mm_sub_ps(_mm_setzero_ps(), simd_fmadd_ps<C>(s10r, v_yai, _mm_mul_ps(s9r, v_ybi)));

                typename simd_vec<C, 4>::f t1r = _mm_sub_ps(s5r, s6r);
                typename simd_vec<C, 4>::f t1i = _mm_sub_ps(s5i, s6i);
                typename simd_vec<C, 4>::f t4r = _mm_add_ps(s5r, s6r);
                typename simd_vec<C, 4>::f t4i = _mm_add_ps(s5i, s6i);

                typename simd_vec<C, 4>::f s11r = simd_fmadd_ps<C>(s7r, v_ybr, f0r);
                s11r = simd_fmadd_ps<C>(s8r, v_yar, s11r);
                typename simd_vec<C, 4>::f s11i = simd_fmadd_ps<C>(s7i, v_ybr, f0i);
                s11i = simd_fmadd_ps<C>(s8i, v_yar, s11i);

                typename simd_vec<C, 4>::f tmp = _mm_add_ps(s7r, s8r);
                f0r = _mm_add_ps(f0r, tmp);
                tmp = _mm_add_ps(s7i, s8i);
                f0i = _mm_add_ps(f0i, tmp);

                typename simd_vec<C, 4>::f s12r = simd_fmsub_ps<C>(s9i, v_yai, _mm_mul_ps(s10i, v_ybi));
                typename simd_vec<C, 4>::f s12i = simd_fmsub_ps<C>(s10r, v_ybi, _mm_mul_ps(s9r, v_yai));

                typename simd_vec<C, 4>::f t2r = _mm_add_ps(s11r, s12r);
                typename simd_vec<C, 4>::f t2i = _mm_add_ps(s11i, s12i);
                typename simd_vec<C, 4>::f t3r = _mm_sub_ps(s11r, s12r);
                typename simd_vec<C, 4>::f t3i = _mm_sub_ps(s11i, s12i);

                _mm_storeu_ps(F0r + u, f0r);
                _mm_storeu_ps(F0i + u, f0i);
                _mm_storeu_ps(F1r + u, t1r);
                _mm_storeu_ps(F1i + u, t1i);
                _mm_storeu_ps(F2r + u, t2r);
                _mm_storeu_ps(F2i + u, t2i);
                _mm_storeu_ps(F3r + u, t3r);
                _mm_storeu_ps(F3i + u, t3i);
                _mm_storeu_ps(F4r + u, t4r);
                _mm_storeu_ps(F4i + u, t4i);
            }
        }
#endif

        for (; u < m; ++u) {
            float* Fout0r = Foutr + u,
                * Fout1r = Foutr + m + u,
                * Fout2r = Foutr + 2 * m + u,
                * Fout3r = Foutr + 3 * m + u,
                * Fout4r = Foutr + 4 * m + u;
            float* Fout0i = Fouti + u,
                * Fout1i = Fouti + m + u,
                * Fout2i = Fouti + 2 * m + u,
                * Fout3i = Fouti + 3 * m + u,
                * Fout4i = Fouti + 4 * m + u;

            float scratchr[13], scratchi[13];

            scratchr[0] = *Fout0r;
            scratchi[0] = *Fout0i;

            scratchr[1] = *Fout1r * tw_r[u * fstride] - *Fout1i * tw_i[u * fstride];
            scratchi[1] = *Fout1r * tw_i[u * fstride] + *Fout1i * tw_r[u * fstride];
            scratchr[2] = *Fout2r * tw_r[2 * u * fstride] - *Fout2i * tw_i[2 * u * fstride];
            scratchi[2] = *Fout2r * tw_i[2 * u * fstride] + *Fout2i * tw_r[2 * u * fstride];
            scratchr[3] = *Fout3r * tw_r[3 * u * fstride] - *Fout3i * tw_i[3 * u * fstride];
            scratchi[3] = *Fout3r * tw_i[3 * u * fstride] + *Fout3i * tw_r[3 * u * fstride];
            scratchr[4] = *Fout4r * tw_r[4 * u * fstride] - *Fout4i * tw_i[4 * u * fstride];
            scratchi[4] = *Fout4r * tw_i[4 * u * fstride] + *Fout4i * tw_r[4 * u * fstride];

            scratchr[7] = scratchr[1] + scratchr[4];
            scratchi[7] = scratchi[1] + scratchi[4];
            scratchr[10] = scratchr[1] - scratchr[4];
            scratchi[10] = scratchi[1] - scratchi[4];
            scratchr[8] = scratchr[2] + scratchr[3];
            scratchi[8] = scratchi[2] + scratchi[3];
            scratchr[9] = scratchr[2] - scratchr[3];
            scratchi[9] = scratchi[2] - scratchi[3];

            *Fout0r += scratchr[7] + scratchr[8];
            *Fout0i += scratchi[7] + scratchi[8];

            scratchr[5] = scratchr[0] + scratchr[7] * yar + scratchr[8] * ybr;
            scratchi[5] = scratchi[0] + scratchi[7] * yar + scratchi[8] * ybr;

            scratchr[6] = scratchi[10] * yai + scratchi[9] * ybi;
            scratchi[6] = -(scratchr[10] * yai) - scratchr[9] * ybi;

            *Fout1r = scratchr[5] - scratchr[6];
            *Fout1i = scratchi[5] - scratchi[6];
            *Fout4r = scratchr[5] + scratchr[6];
            *Fout4i = scratchi[5] + scratchi[6];

            scratchr[11] = scratchr[0] + scratchr[7] * ybr + scratchr[8] * yar;
            scratchi[11] = scratchi[0] + scratchi[7] * ybr + scratchi[8] * yar;
            scratchr[12] = -(scratchi[10] * ybi) + scratchi[9] * yai;
            scratchi[12] = scratchr[10] * ybi - scratchr[9] * yai;

            *Fout2r = scratchr[11] + scratchr[12];
            *Fout2i = scratchi[11] + scratchi[12];
            *Fout3r = scratchr[11] - scratchr[12];
            *Fout3i = scratchi[11] - scratchi[12];
        }
    }
}

template<simd_caps C>
void fft_bfly_generic_kernel<C>::run(float* Foutr, float* Fouti, int                    fstride, const float* twiddles_r, const float* twiddles_i, int                    Norig, int                    m, int                    p, float* scratchr, float* scratchi)
{
#if defined(COSYVOICE_HAS_AVX512)
    if constexpr (C.avx512)
    {
        for (int u = 0; u < m; ++u) {
            int k = u;
            for (int q = 0; q < p; ++q) {
                scratchr[q] = Foutr[k];
                scratchi[q] = Fouti[k];
                k += m;
            }

            k = u;
            for (int q1 = 0; q1 < p; ++q1) {
                const int step = k * fstride % Norig;

                float acc_r = scratchr[0];
                float acc_i = scratchi[0];

                for (int q = 1; q < p; q += 16)
                {
                    const typename simd_vec<C, 16>::mask mask = (q + 16 <= p) ? static_cast<typename simd_vec<C, 16>::mask>(0xFFFFu) : simd_tail_mask16<C>(p - q);

                    alignas(64) int idx[16];
                    int qlocal = q;
                    for (int t = 0; t != 16; ++t, ++qlocal) {
                        int idxv = qlocal * step;
                        if (idxv >= Norig)
                            idxv %= Norig;
                        idx[t] = idxv;
                    }

                    typename simd_vec<C, 16>::f twr = simd_load16_indexed_ps<C>(twiddles_r, idx);
                    typename simd_vec<C, 16>::f twi = simd_load16_indexed_ps<C>(twiddles_i, idx);

                    typename simd_vec<C, 16>::f sr = _mm512_maskz_loadu_ps(mask, scratchr + q);
                    typename simd_vec<C, 16>::f si = _mm512_maskz_loadu_ps(mask, scratchi + q);

                    typename simd_vec<C, 16>::f tr = simd_fmsub_ps<C>(sr, twr, _mm512_mul_ps(si, twi));
                    typename simd_vec<C, 16>::f ti = simd_fmadd_ps<C>(sr, twi, _mm512_mul_ps(si, twr));

                    acc_r += simd_hsum512_ps(tr);
                    acc_i += simd_hsum512_ps(ti);
                }

                Foutr[k] = acc_r;
                Fouti[k] = acc_i;

                k += m;
            }
        }
        return;
    }
    else
#endif
    {
        for (int u = 0; u < m; ++u) {
            int k = u;
            for (int q = 0; q < p; ++q) {
                scratchr[q] = Foutr[k];
                scratchi[q] = Fouti[k];
                k += m;
            }

            k = u;
            for (int q1 = 0; q1 < p; ++q1) {

                const int step = k * fstride % Norig;

                float acc_r = scratchr[0];
                float acc_i = scratchi[0];

                int q = 1;
#if defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
                if constexpr (C.avx || C.avx10_1_256)
                {
                    for (; q + 7 < p; q += 8) {
                        int qlocal = q;
                        alignas(32) int idx[8];
                        for (int i = 0; i != 8; ++i, ++qlocal) {
                            int idxv = qlocal * step;
                            if (idxv >= Norig)
                                idxv %= Norig;
                            idx[i] = idxv;
                        }

                        typename simd_vec<C, 8>::f twr;
                        typename simd_vec<C, 8>::f twi;
#if defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
                        if constexpr (C.avx2 || C.avx10_1_256)
                        {
                            typename simd_vec<C, 8>::i vindex = _mm256_loadu_si256((reinterpret_cast<typename simd_vec<C, 8>::i*>(idx)));
                            twr = _mm256_i32gather_ps(twiddles_r, vindex, 4);
                            twi = _mm256_i32gather_ps(twiddles_i, vindex, 4);
                        }
                        else
#endif
                        {
                            twr = simd_load8_indexed_ps<C>(twiddles_r, idx);
                            twi = simd_load8_indexed_ps<C>(twiddles_i, idx);
                        }

                        typename simd_vec<C, 8>::f sr = _mm256_loadu_ps(scratchr + q);
                        typename simd_vec<C, 8>::f si = _mm256_loadu_ps(scratchi + q);

                        typename simd_vec<C, 8>::f tr = simd_fmsub_ps<C>(sr, twr, _mm256_mul_ps(si, twi));
                        typename simd_vec<C, 8>::f ti = simd_fmadd_ps<C>(sr, twi, _mm256_mul_ps(si, twr));

                        acc_r += hsum256_ps(tr);
                        acc_i += hsum256_ps(ti);
                    }
                }
#endif
                #if defined(COSYVOICE_HAS_AVX10_1_256)
                if constexpr (C.avx10_1_256)
                {
                if (q < p)
                {
                    const typename simd_vec<C, 8>::mask mask = simd_tail_mask8<C>(p - q);
                    int qlocal = q;
                    alignas(32) int idx[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
                    for (int t = 0; t != static_cast<int>(p - q); ++t, ++qlocal)
                    {
                        int idxv = qlocal * step;
                        if (idxv >= Norig)
                            idxv %= Norig;
                        idx[t] = idxv;
                    }
                    typename simd_vec<C, 8>::f twr = simd_tail_load8_indexed_ps<C>(twiddles_r, idx, p - q);
                    typename simd_vec<C, 8>::f twi = simd_tail_load8_indexed_ps<C>(twiddles_i, idx, p - q);
                    typename simd_vec<C, 8>::f sr = _mm256_maskz_loadu_ps(mask, scratchr + q);
                    typename simd_vec<C, 8>::f si = _mm256_maskz_loadu_ps(mask, scratchi + q);
                    typename simd_vec<C, 8>::f tr = simd_fmsub_ps<C>(sr, twr, _mm256_mul_ps(si, twi));
                    typename simd_vec<C, 8>::f ti = simd_fmadd_ps<C>(sr, twi, _mm256_mul_ps(si, twr));
                    acc_r += hsum256_ps(tr);
                    acc_i += hsum256_ps(ti);
                    q = p;
                }
                }
                else
                #endif
#if defined(COSYVOICE_HAS_SSE42) || defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2)
                if constexpr (C.sse42)
                {
                    for (; q + 3 < p; q += 4)
                    {
                        int qlocal = q;
                        alignas(16) int idx[4];
                        for (int i = 0; i != 4; ++i, ++qlocal) {
                            int idxv = qlocal * step;
                            if (idxv >= Norig)
                                idxv %= Norig;
                            idx[i] = idxv;
                        }

                        typename simd_vec<C, 4>::f twr;
                        typename simd_vec<C, 4>::f twi;
#if defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
                        if constexpr (C.avx2 || C.avx10_1_256)
                        {
                            typename simd_vec<C, 4>::i vindex = _mm_loadu_si128(reinterpret_cast<typename simd_vec<C, 4>::i*>(idx));
                            twr = _mm_i32gather_ps(twiddles_r, vindex, 4);
                            twi = _mm_i32gather_ps(twiddles_i, vindex, 4);
                        }
                        else
#endif
                        {
                            twr = simd_load4_indexed_ps<C>(twiddles_r, idx);
                            twi = simd_load4_indexed_ps<C>(twiddles_i, idx);
                        }

                        typename simd_vec<C, 4>::f sr = _mm_loadu_ps(scratchr + q);
                        typename simd_vec<C, 4>::f si = _mm_loadu_ps(scratchi + q);

                        typename simd_vec<C, 4>::f tr = simd_fmsub_ps<C>(sr, twr, _mm_mul_ps(si, twi));
                        typename simd_vec<C, 4>::f ti = simd_fmadd_ps<C>(sr, twi, _mm_mul_ps(si, twr));

                        acc_r += hsum128_ps(tr);
                        acc_i += hsum128_ps(ti);
                    }
                }
#endif

                for (; q < p; ++q) {
                    int twidx = (q * step) % Norig;
                    float tr = scratchr[q] * twiddles_r[twidx] - scratchi[q] * twiddles_i[twidx];
                    float ti = scratchr[q] * twiddles_i[twidx] + scratchi[q] * twiddles_r[twidx];
                    acc_r += tr;
                    acc_i += ti;
                }

                Foutr[k] = acc_r;
                Fouti[k] = acc_i;

                k += m;
            }
        }
    }
}

template<simd_caps C>
void fft_work_kernel<C>::run(float* Foutr, float* Fouti, const float* f, int fstride, int in_stride,
    int* factors, float* scratchr, float* scratchi, float* tw_r, float* tw_i, int Norig)
{
    int Fout_beg = 0;
    const int p = *factors++; /* the radix  */
    const int m = *factors++; /* stage's fft length/p */
    const int Fout_end = p * m;

    if (m == 1) {
        do {
            Foutr[Fout_beg] = *f;
            f += fstride * in_stride;
        } while (++Fout_beg != Fout_end);
    }
    else {
        do {
            // recursive call:
            // DFT of size m*p performed by doing
            // p instances of smaller DFTs of size m,
            // each one takes a decimated version of the input
            run(Foutr + Fout_beg, Fouti + Fout_beg, f, fstride * p, in_stride, factors, scratchr, scratchi, tw_r, tw_i, Norig);
            f += fstride * in_stride;
        } while ((Fout_beg += m) != Fout_end);
    }

    // recombine the p smaller DFTs; direct intra-TU calls (one dispatch at the
    // caller's tree root only)
    switch (p) {
    case 2: fft_bfly2_kernel<C>::run(Foutr, Fouti, fstride, tw_r, tw_i, m); break;
    case 3: fft_bfly3_kernel<C>::run(Foutr, Fouti, fstride, tw_r, tw_i, m); break;
    case 4: fft_bfly4_kernel<C>::run(Foutr, Fouti, fstride, tw_r, tw_i, m); break;
    case 5: fft_bfly5_kernel<C>::run(Foutr, Fouti, fstride, tw_r, tw_i, m); break;
    default: fft_bfly_generic_kernel<C>::run(Foutr, Fouti, fstride, tw_r, tw_i, Norig, m, p, scratchr, scratchi);
    }
}

template<simd_caps C>
void fft_abs_kernel<C>::run(const float* foutr, const float* fouti, float* buffer, int n)
{
    int i = 0;
#if defined(COSYVOICE_HAS_AVX512)
    if constexpr (C.avx512)
    {
        for (; i + 15 < n; i += 16)
        {
            typename simd_vec<C, 16>::f tr = _mm512_loadu_ps(foutr + i);
            typename simd_vec<C, 16>::f ti = _mm512_loadu_ps(fouti + i);
            _mm512_storeu_ps(buffer + i, _mm512_sqrt_ps(simd_fmadd_ps<C>(tr, tr, _mm512_mul_ps(ti, ti))));
        }
        if (i < n)
        {
            const typename simd_vec<C, 16>::mask mask = simd_tail_mask16<C>(n - i);
            typename simd_vec<C, 16>::f tr = _mm512_maskz_loadu_ps(mask, foutr + i);
            typename simd_vec<C, 16>::f ti = _mm512_maskz_loadu_ps(mask, fouti + i);
            _mm512_mask_storeu_ps(buffer + i, mask, _mm512_sqrt_ps(simd_fmadd_ps<C>(tr, tr, _mm512_mul_ps(ti, ti))));
        }
        return;
    }
    else
#endif
    {
#if defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx || C.avx10_1_256)
        {
            while (i + 7 < n)
            {
                typename simd_vec<C, 8>::f tr = _mm256_loadu_ps(foutr + i);
                typename simd_vec<C, 8>::f ti = _mm256_loadu_ps(fouti + i);
                auto abs = _mm256_sqrt_ps(simd_fmadd_ps<C>(tr, tr, _mm256_mul_ps(ti, ti)));
                _mm256_storeu_ps(buffer + i, abs);
                i += 8;
            }
        }
#endif
#if defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx10_1_256)
        {
            if (i < n)
            {
                const typename simd_vec<C, 8>::mask mask = simd_tail_mask8<C>(n - i);
                typename simd_vec<C, 8>::f tr = _mm256_maskz_loadu_ps(mask, foutr + i);
                typename simd_vec<C, 8>::f ti = _mm256_maskz_loadu_ps(mask, fouti + i);
                _mm256_mask_storeu_ps(buffer + i, mask, _mm256_sqrt_ps(simd_fmadd_ps<C>(tr, tr, _mm256_mul_ps(ti, ti))));
            }
            return;
        }
#endif
#if defined(COSYVOICE_HAS_SSE42) || defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2)
        if constexpr (C.sse42)
        {
            while (i + 3 < n)
            {
                typename simd_vec<C, 4>::f tr = _mm_loadu_ps(foutr + i);
                typename simd_vec<C, 4>::f ti = _mm_loadu_ps(fouti + i);
                auto abs = _mm_sqrt_ps(simd_fmadd_ps<C>(tr, tr, _mm_mul_ps(ti, ti)));
                _mm_storeu_ps(buffer + i, abs);
                i += 4;
            }
        }
#endif
        for (; i < n; ++i)
            buffer[i] = sqrtf(foutr[i] * foutr[i] + fouti[i] * fouti[i]);
    }
}

template<simd_caps C>
void init_twiddles_kernel<C>::run(float* twiddles_r, float* twiddles_i, int n, float factor)
{
    int i = 0;
#if defined(COSYVOICE_HAS_AVX512)
    if constexpr (C.avx512)
    {
        const typename simd_vec<C, 16>::f factor512 = _mm512_set1_ps(factor);
        const typename simd_vec<C, 16>::f _16 = _mm512_set1_ps(16.f);
        typename simd_vec<C, 16>::f base = _mm512_setr_ps(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        for (; i + 15 < n; i += 16)
        {
            typename simd_vec<C, 16>::f phase = _mm512_mul_ps(base, factor512);
            base = _mm512_add_ps(base, _16);
            typename simd_vec<C, 16>::f sin512, cos512;
            simd_sincos_ps(phase, &sin512, &cos512);
            _mm512_storeu_ps(twiddles_r + i, cos512);
            _mm512_storeu_ps(twiddles_i + i, sin512);
        }
        if (i < n)
        {
            const typename simd_vec<C, 16>::mask mask = simd_tail_mask16<C>(n - i);
            typename simd_vec<C, 16>::f phase = _mm512_mul_ps(base, factor512);
            typename simd_vec<C, 16>::f sin512, cos512;
            simd_sincos_ps(phase, &sin512, &cos512);
            _mm512_mask_storeu_ps(twiddles_r + i, mask, cos512);
            _mm512_mask_storeu_ps(twiddles_i + i, mask, sin512);
        }
        return;
    }
    else
#endif
    {
#if defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx || C.avx10_1_256)
        {
            const typename simd_vec<C, 8>::f factor256 = _mm256_set1_ps(factor);
            typename simd_vec<C, 8>::f _8 = _mm256_set1_ps(8.f);
            typename simd_vec<C, 8>::f base = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
            for (; i + 7 < n; i += 8)
            {
                typename simd_vec<C, 8>::f phase = _mm256_mul_ps(base, factor256);
                base = _mm256_add_ps(base, _8);
                typename simd_vec<C, 8>::f sin256, cos256;
                simd_sincos_ps<C>(phase, &sin256, &cos256);
                _mm256_storeu_ps(twiddles_r + i, cos256);
                _mm256_storeu_ps(twiddles_i + i, sin256);
            }
        }
#endif
#if defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx10_1_256)
        {
            if (i < n)
            {
                const typename simd_vec<C, 8>::mask mask = simd_tail_mask8<C>(n - i);
                const typename simd_vec<C, 8>::f factor256t = _mm256_set1_ps(factor);
                const typename simd_vec<C, 8>::f base8 = _mm256_setr_ps(static_cast<float>(i), static_cast<float>(i) + 1, static_cast<float>(i) + 2, static_cast<float>(i) + 3,
                    static_cast<float>(i) + 4, static_cast<float>(i) + 5, static_cast<float>(i) + 6, static_cast<float>(i) + 7);
                typename simd_vec<C, 8>::f phase = _mm256_mul_ps(base8, factor256t);
                typename simd_vec<C, 8>::f sin256t, cos256t;
                simd_sincos_ps<C>(phase, &sin256t, &cos256t);
                _mm256_mask_storeu_ps(twiddles_r + i, mask, cos256t);
                _mm256_mask_storeu_ps(twiddles_i + i, mask, sin256t);
            }
            return;
        }
#endif
#if defined(COSYVOICE_HAS_SSE42) || defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2)
        if constexpr (C.sse42)
        {
            const typename simd_vec<C, 4>::f factor128 = _mm_set_ps1(factor);
            typename simd_vec<C, 4>::f _4 = _mm_set_ps1(4.f);
            typename simd_vec<C, 4>::f base128 = _mm_setr_ps(static_cast<float>(i), static_cast<float>(i) + 1, static_cast<float>(i) + 2, static_cast<float>(i) + 3);
            for (; i + 3 < n; i += 4)
            {
                typename simd_vec<C, 4>::f phase = _mm_mul_ps(base128, factor128);
                base128 = _mm_add_ps(base128, _4);
                typename simd_vec<C, 4>::f sin128, cos128;
                simd_sincos_ps(phase, &sin128, &cos128);
                _mm_storeu_ps(twiddles_r + i, cos128);
                _mm_storeu_ps(twiddles_i + i, sin128);
            }
        }
#endif
        for (; i < n; ++i)
        {
            const float phase = factor * i;
            twiddles_r[i] = cosf(phase);
            twiddles_i[i] = sinf(phase);
        }
    }
}

template<simd_caps C>
void safe_divide_kernel<C>::run(float* data, const float* denom, size_t n, float min_denom)
{
    size_t i = 0;
#if defined(COSYVOICE_HAS_AVX512)
    if constexpr (C.avx512)
    {
        const typename simd_vec<C, 16>::f eps512 = _mm512_set1_ps(min_denom);
        for (; i + 15 < n; i += 16)
        {
            typename simd_vec<C, 16>::f d = _mm512_max_ps(_mm512_loadu_ps(denom + i), eps512);
            _mm512_storeu_ps(data + i, _mm512_div_ps(_mm512_loadu_ps(data + i), d));
        }
        if (i < n)
        {
            const typename simd_vec<C, 16>::mask mask = simd_tail_mask16<C>(n - i);
            typename simd_vec<C, 16>::f d = _mm512_max_ps(_mm512_maskz_loadu_ps(mask, denom + i), eps512);
            _mm512_mask_storeu_ps(data + i, mask, _mm512_div_ps(_mm512_maskz_loadu_ps(mask, data + i), d));
        }
        return;
    }
    else
#endif
    {
#if defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx || C.avx10_1_256)
        {
            const typename simd_vec<C, 8>::f eps_v = _mm256_set1_ps(min_denom);
            for (; i + 7 < n; i += 8)
            {
                typename simd_vec<C, 8>::f d = _mm256_loadu_ps(denom + i);
                typename simd_vec<C, 8>::f y = _mm256_loadu_ps(data + i);
                d = _mm256_max_ps(d, eps_v);
                _mm256_storeu_ps(data + i, _mm256_div_ps(y, d));
            }
        }
#endif
#if defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx10_1_256)
        {
            if (i < n)
            {
                const typename simd_vec<C, 8>::mask mask = simd_tail_mask8<C>(n - i);
                typename simd_vec<C, 8>::f d = _mm256_max_ps(_mm256_maskz_loadu_ps(mask, denom + i), _mm256_set1_ps(min_denom));
                _mm256_mask_storeu_ps(data + i, mask, _mm256_div_ps(_mm256_maskz_loadu_ps(mask, data + i), d));
            }
            return;
        }
#endif
#if defined(COSYVOICE_HAS_SSE42) || defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2)
        if constexpr (C.sse42)
        {
            const typename simd_vec<C, 4>::f eps_128 = _mm_set_ps1(min_denom);
            for (; i + 3 < n; i += 4)
            {
                typename simd_vec<C, 4>::f d = _mm_loadu_ps(denom + i);
                typename simd_vec<C, 4>::f y = _mm_loadu_ps(data + i);
                d = _mm_max_ps(d, eps_128);
                _mm_storeu_ps(data + i, _mm_div_ps(y, d));
            }
        }
#endif
        for (; i < n; ++i)
            data[i] /= std::max(denom[i], min_denom);
    }
}

template<simd_caps C>
void apply_window_kernel<C>::run(const float* src, const float* window, float* dst, size_t n)
{
    size_t i = 0;
#if defined(COSYVOICE_HAS_AVX512)
    if constexpr (C.avx512)
    {
        for (; i + 15 < n; i += 16)
            _mm512_storeu_ps(dst + i, _mm512_mul_ps(_mm512_loadu_ps(src + i), _mm512_loadu_ps(window + i)));
        if (i < n)
        {
            const typename simd_vec<C, 16>::mask mask = simd_tail_mask16<C>(n - i);
            _mm512_mask_storeu_ps(dst + i, mask,
                _mm512_mul_ps(_mm512_maskz_loadu_ps(mask, src + i), _mm512_maskz_loadu_ps(mask, window + i)));
        }
        return;
    }
    else
#endif
    {
#if defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx || C.avx10_1_256)
        {
            for (; i + 7 < n; i += 8)
                _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(src + i), _mm256_loadu_ps(window + i)));
        }
#endif
#if defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx10_1_256)
        {
            if (i < n)
            {
                const typename simd_vec<C, 8>::mask mask = simd_tail_mask8<C>(n - i);
                _mm256_mask_storeu_ps(dst + i, mask,
                    _mm256_mul_ps(_mm256_maskz_loadu_ps(mask, src + i), _mm256_maskz_loadu_ps(mask, window + i)));
            }
            return;
        }
#endif
#if defined(COSYVOICE_HAS_SSE42) || defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2)
        if constexpr (C.sse42)
        {
            for (; i + 3 < n; i += 4)
                _mm_storeu_ps(dst + i, _mm_mul_ps(_mm_loadu_ps(src + i), _mm_loadu_ps(window + i)));
        }
#endif
        for (; i < n; ++i)
            dst[i] = src[i] * window[i];
    }
}

template<simd_caps C>
float mel_dot_kernel<C>::run(const float* a, const float* b, size_t n)
{
    size_t i = 0;
    float value = 0.f;
#if defined(COSYVOICE_HAS_AVX512)
    if constexpr (C.avx512)
    {
        typename simd_vec<C, 16>::f sum512 = _mm512_setzero_ps();
        for (; i + 15 < n; i += 16)
            sum512 = simd_fmadd_ps<C>(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), sum512);
        if (i < n)
        {
            const typename simd_vec<C, 16>::mask mask = simd_tail_mask16<C>(n - i);
            sum512 = simd_fmadd_ps<C>(_mm512_maskz_loadu_ps(mask, a + i), _mm512_maskz_loadu_ps(mask, b + i), sum512);
        }
        return simd_hsum512_ps(sum512);
    }
    else
#endif
    {
#if defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx || C.avx10_1_256)
        {
            typename simd_vec<C, 8>::f sum256 = _mm256_setzero_ps();
            for (; i + 7 < n; i += 8)
                sum256 = simd_fmadd_ps<C>(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum256);
            value = simd_hsum_ps(_mm_add_ps(_mm256_castps256_ps128(sum256), _mm256_extractf128_ps(sum256, 1)));
        }
#endif
#if defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx10_1_256)
        {
            if (i < n)
            {
                const typename simd_vec<C, 8>::mask mask = simd_tail_mask8<C>(n - i);
                typename simd_vec<C, 8>::f acc8 = simd_fmadd_ps<C>(_mm256_maskz_loadu_ps(mask, a + i), _mm256_maskz_loadu_ps(mask, b + i), _mm256_setzero_ps());
                value += hsum256_ps(acc8);
            }
            return value;
        }
#endif
#if defined(COSYVOICE_HAS_SSE42) || defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2)
        if constexpr (C.sse42)
        {
            typename simd_vec<C, 4>::f sum128 = _mm_setzero_ps();
            for (; i + 3 < n; i += 4)
                sum128 = simd_fmadd_ps<C>(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i), sum128);
            value += simd_hsum_ps(sum128);
        }
#endif
        for (; i < n; ++i)
            value += a[i] * b[i];
        return value;
    }
}

template<simd_caps C>
float mel_dot_sq_kernel<C>::run(const float* a, const float* b, size_t n)
{
    size_t i = 0;
    float value = 0.f;
#if defined(COSYVOICE_HAS_AVX512)
    if constexpr (C.avx512)
    {
        typename simd_vec<C, 16>::f sum512 = _mm512_setzero_ps();
        for (; i + 15 < n; i += 16)
        {
            typename simd_vec<C, 16>::f va = _mm512_loadu_ps(a + i);
            va = _mm512_mul_ps(va, va);
            sum512 = simd_fmadd_ps<C>(va, _mm512_loadu_ps(b + i), sum512);
        }
        if (i < n)
        {
            const typename simd_vec<C, 16>::mask mask = simd_tail_mask16<C>(n - i);
            typename simd_vec<C, 16>::f va = _mm512_maskz_loadu_ps(mask, a + i);
            va = _mm512_mul_ps(va, va);
            sum512 = simd_fmadd_ps<C>(va, _mm512_maskz_loadu_ps(mask, b + i), sum512);
        }
        return simd_hsum512_ps(sum512);
    }
    else
#endif
    {
#if defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx || C.avx10_1_256)
        {
            typename simd_vec<C, 8>::f sum256 = _mm256_setzero_ps();
            for (; i + 7 < n; i += 8)
            {
                typename simd_vec<C, 8>::f va = _mm256_loadu_ps(a + i);
                va = _mm256_mul_ps(va, va);
                sum256 = simd_fmadd_ps<C>(va, _mm256_loadu_ps(b + i), sum256);
            }
            value = simd_hsum_ps(_mm_add_ps(_mm256_castps256_ps128(sum256), _mm256_extractf128_ps(sum256, 1)));
        }
#endif
#if defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx10_1_256)
        {
            if (i < n)
            {
                const typename simd_vec<C, 8>::mask mask = simd_tail_mask8<C>(n - i);
                typename simd_vec<C, 8>::f va = _mm256_maskz_loadu_ps(mask, a + i);
                va = _mm256_mul_ps(va, va);
                typename simd_vec<C, 8>::f acc8 = simd_fmadd_ps<C>(va, _mm256_maskz_loadu_ps(mask, b + i), _mm256_setzero_ps());
                value += hsum256_ps(acc8);
            }
            return value;
        }
#endif
#if defined(COSYVOICE_HAS_SSE42) || defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2)
        if constexpr (C.sse42)
        {
            typename simd_vec<C, 4>::f sum128 = _mm_setzero_ps();
            for (; i + 3 < n; i += 4)
            {
                typename simd_vec<C, 4>::f va = _mm_loadu_ps(a + i);
                va = _mm_mul_ps(va, va);
                sum128 = simd_fmadd_ps<C>(va, _mm_loadu_ps(b + i), sum128);
            }
            value += simd_hsum_ps(sum128);
        }
#endif
        for (; i < n; ++i)
            value += a[i] * a[i] * b[i];
        return value;
    }
}

template<simd_caps C>
float sum_kernel<C>::run(const float* data, size_t n)
{
    size_t i = 0;
    float sum = 0.f;
#if defined(COSYVOICE_HAS_AVX512)
    if constexpr (C.avx512)
    {
        typename simd_vec<C, 16>::f sum512 = _mm512_setzero_ps();
        for (; i + 15 < n; i += 16)
            sum512 = _mm512_add_ps(sum512, _mm512_loadu_ps(data + i));
        if (i < n)
            sum512 = _mm512_add_ps(sum512, _mm512_maskz_loadu_ps(simd_tail_mask16<C>(n - i), data + i));
        return simd_hsum512_ps(sum512);
    }
    else
#endif
    {
#if defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx || C.avx10_1_256)
        {
            typename simd_vec<C, 8>::f sum256 = _mm256_setzero_ps();
            for (; i + 7 < n; i += 8)
                sum256 = _mm256_add_ps(sum256, _mm256_loadu_ps(data + i));
            sum = simd_hsum_ps(_mm_add_ps(_mm256_castps256_ps128(sum256), _mm256_extractf128_ps(sum256, 1)));
        }
#endif
#if defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx10_1_256)
        {
            if (i < n)
                sum += hsum256_ps(_mm256_maskz_loadu_ps(simd_tail_mask8<C>(n - i), data + i));
            return sum;
        }
#endif
#if defined(COSYVOICE_HAS_SSE42) || defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2)
        if constexpr (C.sse42)
        {
            typename simd_vec<C, 4>::f sum128 = _mm_setzero_ps();
            for (; i + 3 < n; i += 4)
                sum128 = _mm_add_ps(sum128, _mm_loadu_ps(data + i));
            sum += simd_hsum_ps(sum128);
        }
#endif
        for (; i < n; ++i)
            sum += data[i];
        return sum;
    }
}

template<simd_caps C>
void log_map_kernel<C>::run(float* data, size_t n, float min_level)
{
    size_t i = 0;
#if defined(COSYVOICE_HAS_AVX512)
    if constexpr (C.avx512)
    {
        const typename simd_vec<C, 16>::f level512 = _mm512_set1_ps(min_level);
        for (; i + 15 < n; i += 16)
        {
            typename simd_vec<C, 16>::f values = _mm512_max_ps(_mm512_loadu_ps(data + i), level512);
            _mm512_storeu_ps(data + i, simd_log_ps(values));
        }
        if (i < n)
        {
            const typename simd_vec<C, 16>::mask mask = simd_tail_mask16<C>(n - i);
            typename simd_vec<C, 16>::f values = _mm512_max_ps(_mm512_maskz_loadu_ps(mask, data + i), level512);
            _mm512_mask_storeu_ps(data + i, mask, simd_log_ps(values));
        }
        return;
    }
    else
#endif
    {
#if defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx || C.avx10_1_256)
        {
            const typename simd_vec<C, 8>::f level256 = _mm256_set1_ps(min_level);
            for (; i + 7 < n; i += 8)
            {
                typename simd_vec<C, 8>::f values = _mm256_loadu_ps(data + i);
                values = _mm256_max_ps(values, level256);
                _mm256_storeu_ps(data + i, simd_log_ps<C>(values));
            }
        }
#endif
#if defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx10_1_256)
        {
            if (i < n)
            {
                const typename simd_vec<C, 8>::mask mask = simd_tail_mask8<C>(n - i);
                typename simd_vec<C, 8>::f values = _mm256_max_ps(_mm256_maskz_loadu_ps(mask, data + i), _mm256_set1_ps(min_level));
                _mm256_mask_storeu_ps(data + i, mask, simd_log_ps<C>(values));
            }
            return;
        }
#endif
#if defined(COSYVOICE_HAS_SSE42) || defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2)
        if constexpr (C.sse42)
        {
            const typename simd_vec<C, 4>::f level128 = _mm_set_ps1(min_level);
            for (; i + 3 < n; i += 4)
            {
                typename simd_vec<C, 4>::f values = _mm_loadu_ps(data + i);
                values = _mm_max_ps(values, level128);
                _mm_storeu_ps(data + i, simd_log_ps(values));
            }
        }
#endif
        for (; i < n; ++i)
            data[i] = std::log(std::max(min_level, data[i]));
    }
}

template<simd_caps C>
float log10_map_kernel<C>::run(float* data, size_t n, float min_level)
{
    size_t i = 0;
    float max_value = min_level;
#if defined(COSYVOICE_HAS_AVX512)
    if constexpr (C.avx512)
    {
        const typename simd_vec<C, 16>::f level512 = _mm512_set1_ps(min_level);
        typename simd_vec<C, 16>::f maximum_512 = _mm512_set1_ps(min_level);
        for (; i + 15 < n; i += 16)
        {
            typename simd_vec<C, 16>::f values = simd_log10_ps(_mm512_max_ps(_mm512_loadu_ps(data + i), level512));
            _mm512_storeu_ps(data + i, values);
            maximum_512 = _mm512_max_ps(maximum_512, values);
        }
        if (i < n)
        {
            const typename simd_vec<C, 16>::mask mask = simd_tail_mask16<C>(n - i);
            typename simd_vec<C, 16>::f values = simd_log10_ps(_mm512_max_ps(_mm512_maskz_loadu_ps(mask, data + i), level512));
            _mm512_mask_storeu_ps(data + i, mask, values);
            maximum_512 = _mm512_max_ps(maximum_512, values); // invalid lanes: log10(min_level) below any running max
        }
        return std::max(max_value, simd_hmax512_ps(maximum_512));
    }
    else
#endif
    {
#if defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx || C.avx10_1_256)
        {
            const typename simd_vec<C, 8>::f level256 = _mm256_set1_ps(min_level);
            typename simd_vec<C, 8>::f maximum_256 = _mm256_set1_ps(min_level);
            for (; i + 7 < n; i += 8)
            {
                typename simd_vec<C, 8>::f values = _mm256_loadu_ps(data + i);
                values = _mm256_max_ps(values, level256);
                values = simd_log10_ps<C>(values);
                _mm256_storeu_ps(data + i, values);
                maximum_256 = _mm256_max_ps(maximum_256, values);
            }
            max_value = std::max(max_value, hmax_ps(maximum_256));
        }
#endif
#if defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx10_1_256)
        {
            if (i < n)
            {
                const typename simd_vec<C, 8>::mask mask = simd_tail_mask8<C>(n - i);
                typename simd_vec<C, 8>::f values = simd_log10_ps<C>(_mm256_max_ps(_mm256_maskz_loadu_ps(mask, data + i), _mm256_set1_ps(min_level)));
                _mm256_mask_storeu_ps(data + i, mask, values);
                max_value = std::max(max_value, hmax_ps(values)); // invalid lanes: log10(min_level) below any running max
            }
            return max_value;
        }
#endif
#if defined(COSYVOICE_HAS_SSE42) || defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2)
        if constexpr (C.sse42)
        {
            const typename simd_vec<C, 4>::f level128 = _mm_set_ps1(min_level);
            typename simd_vec<C, 4>::f maximum_128 = _mm_set_ps1(max_value);
            for (; i + 3 < n; i += 4)
            {
                typename simd_vec<C, 4>::f values = _mm_loadu_ps(data + i);
                values = _mm_max_ps(values, level128);
                values = simd_log10_ps(values);
                _mm_storeu_ps(data + i, values);
                maximum_128 = _mm_max_ps(maximum_128, values);
            }
            max_value = std::max(max_value, hmax_ps(maximum_128));
        }
#endif
        for (; i < n; ++i)
        {
            const float value = std::log10(std::max(min_level, data[i]));
            data[i] = value;
            if (value > max_value)
                max_value = value;
        }
        return max_value;
    }
}

template<simd_caps C>
void spec_normalize_kernel<C>::run(float* data, size_t n, float base)
{
    size_t i = 0;
#if defined(COSYVOICE_HAS_AVX512)
    if constexpr (C.avx512)
    {
        const typename simd_vec<C, 16>::f base512 = _mm512_set1_ps(base);
        const typename simd_vec<C, 16>::f four512 = _mm512_set1_ps(4.f);
        for (; i + 15 < n; i += 16)
        {
            typename simd_vec<C, 16>::f values = _mm512_add_ps(_mm512_max_ps(_mm512_loadu_ps(data + i), base512), four512);
            _mm512_storeu_ps(data + i, _mm512_div_ps(values, four512));
        }
        if (i < n)
        {
            const typename simd_vec<C, 16>::mask mask = simd_tail_mask16<C>(n - i);
            typename simd_vec<C, 16>::f values = _mm512_add_ps(_mm512_max_ps(_mm512_maskz_loadu_ps(mask, data + i), base512), four512);
            _mm512_mask_storeu_ps(data + i, mask, _mm512_div_ps(values, four512));
        }
        return;
    }
    else
#endif
    {
#if defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx || C.avx10_1_256)
        {
            const typename simd_vec<C, 8>::f base256 = _mm256_set1_ps(base);
            const typename simd_vec<C, 8>::f four256 = _mm256_set1_ps(4.f);
            for (; i + 7 < n; i += 8)
            {
                typename simd_vec<C, 8>::f values = _mm256_loadu_ps(data + i);
                values = _mm256_max_ps(values, base256);
                values = _mm256_add_ps(values, four256);
                _mm256_storeu_ps(data + i, _mm256_div_ps(values, four256));
            }
        }
#endif
#if defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx10_1_256)
        {
            if (i < n)
            {
                const typename simd_vec<C, 8>::mask mask = simd_tail_mask8<C>(n - i);
                typename simd_vec<C, 8>::f values = _mm256_add_ps(_mm256_max_ps(_mm256_maskz_loadu_ps(mask, data + i), _mm256_set1_ps(base)), _mm256_set1_ps(4.f));
                _mm256_mask_storeu_ps(data + i, mask, _mm256_div_ps(values, _mm256_set1_ps(4.f)));
            }
            return;
        }
#endif
#if defined(COSYVOICE_HAS_SSE42) || defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2)
        if constexpr (C.sse42)
        {
            const typename simd_vec<C, 4>::f base128 = _mm_set_ps1(base);
            const typename simd_vec<C, 4>::f four128 = _mm_set_ps1(4.f);
            for (; i + 3 < n; i += 4)
            {
                typename simd_vec<C, 4>::f values = _mm_loadu_ps(data + i);
                values = _mm_max_ps(values, base128);
                values = _mm_add_ps(values, four128);
                _mm_storeu_ps(data + i, _mm_div_ps(values, four128));
            }
        }
#endif
        for (; i < n; ++i)
        {
            if (base > data[i])
                data[i] = (base + 4.f) / 4.f;
            else
                data[i] = (data[i] + 4.f) / 4.f;
        }
    }
}

template<simd_caps C>
void sub_mean_kernel<C>::run(float* dst, const float* src, size_t n, float mean)
{
    size_t i = 0;
#if defined(COSYVOICE_HAS_AVX512)
    if constexpr (C.avx512)
    {
        const typename simd_vec<C, 16>::f mean512 = _mm512_set1_ps(mean);
        for (; i + 15 < n; i += 16)
            _mm512_storeu_ps(dst + i, _mm512_sub_ps(_mm512_loadu_ps(src + i), mean512));
        if (i < n)
        {
            const typename simd_vec<C, 16>::mask mask = simd_tail_mask16<C>(n - i);
            _mm512_mask_storeu_ps(dst + i, mask, _mm512_sub_ps(_mm512_maskz_loadu_ps(mask, src + i), mean512));
        }
        return;
    }
    else
#endif
    {
#if defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx || C.avx10_1_256)
        {
            const typename simd_vec<C, 8>::f mean256 = _mm256_set1_ps(mean);
            for (; i + 7 < n; i += 8)
                _mm256_storeu_ps(dst + i, _mm256_sub_ps(_mm256_loadu_ps(src + i), mean256));
        }
#endif
#if defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx10_1_256)
        {
            if (i < n)
            {
                const typename simd_vec<C, 8>::mask mask = simd_tail_mask8<C>(n - i);
                _mm256_mask_storeu_ps(dst + i, mask, _mm256_sub_ps(_mm256_maskz_loadu_ps(mask, src + i), _mm256_set1_ps(mean)));
            }
            return;
        }
#endif
#if defined(COSYVOICE_HAS_SSE42) || defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2)
        if constexpr (C.sse42)
        {
            const typename simd_vec<C, 4>::f mean128 = _mm_set_ps1(mean);
            for (; i + 3 < n; i += 4)
                _mm_storeu_ps(dst + i, _mm_sub_ps(_mm_loadu_ps(src + i), mean128));
        }
#endif
        for (; i < n; ++i)
            dst[i] = src[i] - mean;
    }
}

template<simd_caps C>
void preemph_gain_kernel<C>::run(float* frames, const float* prev, const float* window, size_t n, size_t win_size)
{
    size_t i = 0;
#if defined(COSYVOICE_HAS_AVX512)
    if constexpr (C.avx512)
    {
        const typename simd_vec<C, 16>::f factor512 = _mm512_set1_ps(0.97f);
        for (; i + 15 < n; i += 16)
        {
            typename simd_vec<C, 16>::f vframes = _mm512_loadu_ps(frames + i);
            typename simd_vec<C, 16>::f vprev = _mm512_loadu_ps(prev + i);
            vframes = _mm512_sub_ps(vframes, _mm512_mul_ps(vprev, factor512));
            vframes = _mm512_mul_ps(vframes, _mm512_loadu_ps(window + (i % win_size)));
            _mm512_storeu_ps(frames + i, vframes);
        }
        if (i < n)
        {
            const typename simd_vec<C, 16>::mask mask = simd_tail_mask16<C>(n - i);
            typename simd_vec<C, 16>::f vframes = _mm512_maskz_loadu_ps(mask, frames + i);
            typename simd_vec<C, 16>::f vprev = _mm512_maskz_loadu_ps(mask, prev + i);
            vframes = _mm512_sub_ps(vframes, _mm512_mul_ps(vprev, factor512));
            vframes = _mm512_mul_ps(vframes, _mm512_maskz_loadu_ps(mask, window + (i % win_size)));
            _mm512_mask_storeu_ps(frames + i, mask, vframes);
        }
        return;
    }
    else
#endif
    {
#if defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx || C.avx10_1_256)
        {
            const typename simd_vec<C, 8>::f factor256 = _mm256_set1_ps(0.97f);
            for (; i + 7 < n; i += 8)
            {
                typename simd_vec<C, 8>::f vframes = _mm256_loadu_ps(frames + i);
                typename simd_vec<C, 8>::f vprev = _mm256_loadu_ps(prev + i);
                vframes = _mm256_sub_ps(vframes, _mm256_mul_ps(vprev, factor256));
                vframes = _mm256_mul_ps(vframes, _mm256_loadu_ps(window + (i % win_size)));
                _mm256_storeu_ps(frames + i, vframes);
            }
        }
#endif
#if defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx10_1_256)
        {
            if (i < n)
            {
                const typename simd_vec<C, 8>::mask mask = simd_tail_mask8<C>(n - i);
                typename simd_vec<C, 8>::f vframes = _mm256_maskz_loadu_ps(mask, frames + i);
                typename simd_vec<C, 8>::f vprev = _mm256_maskz_loadu_ps(mask, prev + i);
                vframes = _mm256_sub_ps(vframes, _mm256_mul_ps(vprev, _mm256_set1_ps(0.97f)));
                vframes = _mm256_mul_ps(vframes, _mm256_maskz_loadu_ps(mask, window + (i % win_size)));
                _mm256_mask_storeu_ps(frames + i, mask, vframes);
            }
            return;
        }
#endif
#if defined(COSYVOICE_HAS_SSE42) || defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2)
        if constexpr (C.sse42)
        {
            const typename simd_vec<C, 4>::f factor128 = _mm_set_ps1(0.97f);
            for (; i + 3 < n; i += 4)
            {
                typename simd_vec<C, 4>::f vframes = _mm_loadu_ps(frames + i);
                typename simd_vec<C, 4>::f vprev = _mm_loadu_ps(prev + i);
                vframes = _mm_sub_ps(vframes, _mm_mul_ps(vprev, factor128));
                vframes = _mm_mul_ps(vframes, _mm_loadu_ps(window + (i % win_size)));
                _mm_storeu_ps(frames + i, vframes);
            }
        }
#endif
        for (; i < n; ++i)
            frames[i] = (frames[i] - 0.97f * prev[i]) * window[i % win_size];
    }
}

template<simd_caps C>
void square_store_kernel<C>::run(const float* src, float* dst, size_t n)
{
    size_t i = 0;
#if defined(COSYVOICE_HAS_AVX512)
    if constexpr (C.avx512)
    {
        for (; i + 15 < n; i += 16)
        {
            typename simd_vec<C, 16>::f v = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, _mm512_mul_ps(v, v));
        }
        if (i < n)
        {
            const typename simd_vec<C, 16>::mask mask = simd_tail_mask16<C>(n - i);
            typename simd_vec<C, 16>::f v = _mm512_maskz_loadu_ps(mask, src + i);
            _mm512_mask_storeu_ps(dst + i, mask, _mm512_mul_ps(v, v));
        }
        return;
    }
    else
#endif
    {
#if defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx || C.avx10_1_256)
        {
            for (; i + 7 < n; i += 8)
            {
                typename simd_vec<C, 8>::f v = _mm256_loadu_ps(src + i);
                _mm256_storeu_ps(dst + i, _mm256_mul_ps(v, v));
            }
        }
#endif
#if defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx10_1_256)
        {
            if (i < n)
            {
                const typename simd_vec<C, 8>::mask mask = simd_tail_mask8<C>(n - i);
                typename simd_vec<C, 8>::f v = _mm256_maskz_loadu_ps(mask, src + i);
                _mm256_mask_storeu_ps(dst + i, mask, _mm256_mul_ps(v, v));
            }
            return;
        }
#endif
#if defined(COSYVOICE_HAS_SSE42) || defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2)
        if constexpr (C.sse42)
        {
            for (; i + 3 < n; i += 4)
            {
                typename simd_vec<C, 4>::f v = _mm_loadu_ps(src + i);
                _mm_storeu_ps(dst + i, _mm_mul_ps(v, v));
            }
        }
#endif
        for (; i < n; ++i)
            dst[i] = src[i] * src[i];
    }
}

template<simd_caps C>
void column_mean_sub_kernel<C>::run(float* column, size_t rows, size_t stride)
{
    size_t i = 0;
    float sum = 0.f;
#if defined(COSYVOICE_HAS_AVX512)
    if constexpr (C.avx512)
    {
        typename simd_vec<C, 16>::f sum512 = _mm512_setzero_ps();
        for (; i + 15 < rows; i += 16)
            sum512 = _mm512_add_ps(sum512, simd_load16_strided_ps<C>(column + i * stride, static_cast<int>(stride)));
        if (i < rows)
            sum512 = _mm512_add_ps(sum512, simd_tail_load16_strided_ps<C>(column + i * stride, static_cast<int>(stride), rows - i));
        sum = simd_hsum512_ps(sum512);

        const float mean = sum / static_cast<float>(rows);
        const typename simd_vec<C, 16>::f mean512 = _mm512_set1_ps(mean);

        i = 0;
        for (; i + 15 < rows; i += 16)
        {
            typename simd_vec<C, 16>::f v = _mm512_sub_ps(simd_load16_strided_ps<C>(column + i * stride, static_cast<int>(stride)), mean512);
            alignas(64) float values[16];
            _mm512_store_ps(values, v);
            float* row = column + i * stride;
            for (int k = 0; k != 16; ++k)
                row[k * stride] = values[k];
        }
        if (i < rows)
        {
            typename simd_vec<C, 16>::f v = _mm512_sub_ps(simd_tail_load16_strided_ps<C>(column + i * stride, static_cast<int>(stride), rows - i), mean512);
            alignas(64) float values[16];
            _mm512_store_ps(values, v);
            float* row = column + i * stride;
            for (size_t k = 0; i + k < rows; ++k)
                row[k * stride] = values[k];
        }
        return;
    }
    else
#endif
    {
#if defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx || C.avx10_1_256)
        {
            typename simd_vec<C, 8>::f sum256 = _mm256_setzero_ps();
#if defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
            if constexpr (C.avx2 || C.avx10_1_256)
            {
                typename simd_vec<C, 8>::i stride256 = _mm256_set1_epi32(static_cast<int>(stride));
                typename simd_vec<C, 8>::i stride8v = _mm256_set1_epi32(static_cast<int>(8 * stride));
                typename simd_vec<C, 8>::i idx256 = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), stride256);
                for (; i + 7 < rows; i += 8)
                {
                    sum256 = _mm256_add_ps(sum256, _mm256_i32gather_ps(column, idx256, 4));
                    idx256 = _mm256_add_epi32(idx256, stride8v);
                }
            }
            else
#endif
                for (; i + 7 < rows; i += 8)
                    sum256 = _mm256_add_ps(sum256, simd_load8_strided_ps<C>(column + i * stride, static_cast<int>(stride)));

            sum = simd_hsum_ps(_mm_add_ps(_mm256_castps256_ps128(sum256), _mm256_extractf128_ps(sum256, 1)));
        }
#endif
#if defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx10_1_256)
        {
            if (i < rows)
                sum += hsum256_ps(simd_tail_load8_strided_ps<C>(column + i * stride, static_cast<int>(stride), rows - i));
            i = rows;
        }
#endif
#if defined(COSYVOICE_HAS_SSE42) || defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2)
        if constexpr (C.sse42)
        {
            typename simd_vec<C, 4>::f sum128 = _mm_setzero_ps();
#if defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
            if constexpr (C.avx2 || C.avx10_1_256)
            {
                typename simd_vec<C, 4>::i stride128 = _mm_set1_epi32(static_cast<int>(stride));
                typename simd_vec<C, 4>::i stride4v = _mm_set1_epi32(static_cast<int>(4 * stride));
                typename simd_vec<C, 4>::i idx128 = _mm_mullo_epi32(
                    _mm_setr_epi32(static_cast<int>(i), static_cast<int>(i) + 1, static_cast<int>(i) + 2, static_cast<int>(i) + 3),
                    stride128);
                for (; i + 3 < rows; i += 4)
                {
                    sum128 = _mm_add_ps(sum128, _mm_i32gather_ps(column, idx128, 4));
                    idx128 = _mm_add_epi32(idx128, stride4v);
                }
            }
            else
#endif
                for (; i + 3 < rows; i += 4)
                    sum128 = _mm_add_ps(sum128, simd_load4_strided_ps<C>(column + i * stride, static_cast<int>(stride)));
            sum += simd_hsum_ps(sum128);
        }
#endif
        for (; i < rows; ++i)
            sum += column[i * stride];

        const float mean = sum / static_cast<float>(rows);

        i = 0;
#if defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx || C.avx10_1_256)
        {
#if defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
            if constexpr (C.avx2 || C.avx10_1_256)
            {
                typename simd_vec<C, 8>::i stride256 = _mm256_set1_epi32(static_cast<int>(stride));
                typename simd_vec<C, 8>::i stride8v = _mm256_set1_epi32(static_cast<int>(8 * stride));
                const typename simd_vec<C, 8>::f mean256 = _mm256_set1_ps(mean);
                typename simd_vec<C, 8>::i idx256 = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), stride256);
                for (; i + 7 < rows; i += 8)
                {
                    typename simd_vec<C, 8>::f v = _mm256_i32gather_ps(column, idx256, 4);
                    v = _mm256_sub_ps(v, mean256);
                    alignas(32) float values[8];
                    _mm256_store_ps(values, v);
                    float* row = column + i * stride;
                    for (int k = 0; k != 8; ++k)
                        row[k * stride] = values[k];
                    idx256 = _mm256_add_epi32(idx256, stride8v);
                }
            }
            else
#endif
            {
                const typename simd_vec<C, 8>::f mean256 = _mm256_set1_ps(mean);
                for (; i + 7 < rows; i += 8)
                {
                    typename simd_vec<C, 8>::f v = _mm256_sub_ps(simd_load8_strided_ps<C>(column + i * stride, static_cast<int>(stride)), mean256);
                    alignas(32) float values[8];
                    _mm256_store_ps(values, v);
                    float* row = column + i * stride;
                    for (int k = 0; k != 8; ++k)
                        row[k * stride] = values[k];
                }
            }
        }
#endif
#if defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx10_1_256)
        {
            if (i < rows)
            {
                typename simd_vec<C, 8>::f v = _mm256_sub_ps(simd_tail_load8_strided_ps<C>(column + i * stride, static_cast<int>(stride), rows - i), _mm256_set1_ps(mean));
                alignas(32) float values[8];
                _mm256_store_ps(values, v);
                float* row = column + i * stride;
                for (size_t k = 0; i + k < rows; ++k)
                    row[k * stride] = values[k];
            }
            i = rows;
        }
#endif
#if defined(COSYVOICE_HAS_SSE42) || defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2)
        if constexpr (C.sse42)
        {
            const typename simd_vec<C, 4>::f mean128 = _mm_set_ps1(mean);
#if defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
            if constexpr (C.avx2 || C.avx10_1_256)
            {
                typename simd_vec<C, 4>::i stride128 = _mm_set1_epi32(static_cast<int>(stride));
                typename simd_vec<C, 4>::i stride4v = _mm_set1_epi32(static_cast<int>(4 * stride));
                typename simd_vec<C, 4>::i idx128 = _mm_mullo_epi32(
                    _mm_setr_epi32(static_cast<int>(i), static_cast<int>(i) + 1, static_cast<int>(i) + 2, static_cast<int>(i) + 3),
                    stride128);
                for (; i + 3 < rows; i += 4)
                {
                    typename simd_vec<C, 4>::f v = _mm_i32gather_ps(column, idx128, 4);
                    v = _mm_sub_ps(v, mean128);
                    alignas(16) float values[4];
                    _mm_store_ps(values, v);
                    float* row = column + i * stride;
                    for (int k = 0; k != 4; ++k)
                        row[k * stride] = values[k];
                    idx128 = _mm_add_epi32(idx128, stride4v);
                }
            }
            else
#endif
            {
                for (; i + 3 < rows; i += 4)
                {
                    typename simd_vec<C, 4>::f v = _mm_sub_ps(simd_load4_strided_ps<C>(column + i * stride, static_cast<int>(stride)), mean128);
                    alignas(16) float values[4];
                    _mm_store_ps(values, v);
                    float* row = column + i * stride;
                    for (int k = 0; k != 4; ++k)
                        row[k * stride] = values[k];
                }
            }
        }
#endif
        for (; i < rows; ++i)
            column[i * stride] -= mean;
    }
}

template<simd_caps C>
void div_kernel<C>::run(float* data, size_t n, float divisor)
{
    size_t i = 0;
#if defined(COSYVOICE_HAS_AVX512)
    if constexpr (C.avx512)
    {
        const typename simd_vec<C, 16>::f div512 = _mm512_set1_ps(divisor);
        for (; i + 15 < n; i += 16)
            _mm512_storeu_ps(data + i, _mm512_div_ps(_mm512_loadu_ps(data + i), div512));
        if (i < n)
        {
            const typename simd_vec<C, 16>::mask mask = simd_tail_mask16<C>(n - i);
            _mm512_mask_storeu_ps(data + i, mask, _mm512_div_ps(_mm512_maskz_loadu_ps(mask, data + i), div512));
        }
        return;
    }
    else
#endif
    {
#if defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx || C.avx10_1_256)
        {
            const typename simd_vec<C, 8>::f div256 = _mm256_set1_ps(divisor);
            for (; i + 7 < n; i += 8)
            {
                typename simd_vec<C, 8>::f values = _mm256_loadu_ps(data + i);
                values = _mm256_div_ps(values, div256);
                _mm256_storeu_ps(data + i, values);
            }
        }
#endif
#if defined(COSYVOICE_HAS_AVX10_1_256)
        if constexpr (C.avx10_1_256)
        {
            if (i < n)
            {
                const typename simd_vec<C, 8>::mask mask = simd_tail_mask8<C>(n - i);
                _mm256_mask_storeu_ps(data + i, mask, _mm256_div_ps(_mm256_maskz_loadu_ps(mask, data + i), _mm256_set1_ps(divisor)));
            }
            return;
        }
#endif
#if defined(COSYVOICE_HAS_SSE42) || defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2)
        if constexpr (C.sse42)
        {
            const typename simd_vec<C, 4>::f div128 = _mm_set_ps1(divisor);
            for (; i + 3 < n; i += 4)
            {
                typename simd_vec<C, 4>::f values = _mm_loadu_ps(data + i);
                values = _mm_div_ps(values, div128);
                _mm_storeu_ps(data + i, values);
            }
        }
#endif
        for (; i < n; ++i)
            data[i] /= divisor;
    }
}
