#pragma once

// ---------------------------------------------------------------------------
// SIMD kernel class DECLARATIONS + per-tier explicit-instantiation plumbing.
//
// The kernel bodies live in simd-kernels-impl.h, which is included ONLY by the
// dedicated one-source-per-tier targets (src/simd/kernels_{scalar,sse42,avx,
// avx2,avx512}.cpp) that explicit-instantiate them for their preset.
// Callers compile against ONLY these declarations: gcc/clang/MSVC otherwise
// semantics-check an inline member body even under `extern template` (for
// inlining), which would require the -m<isa> flags only the matching tier
// object provides. With the bodies out-of-callers' sight, `extern template`
// turns every `Kernel<C>::run(...)` in the dispatch into an ordinary external
// call, so caller TUs compile at a plain SSE2 baseline and every tier object is
// instruction-bounded by its own flags (fixing the gcc/clang cross-tier #UD).
//
// Kernel vector values are named through the DEPENDENT trait simd_vec<C,N>
// (::f / ::i / ::mask); the concrete specializations live in the tier object
// sources, so e.g. the AVX2 object never needs __m512 while the discarded
// AVX-512 branches of its bodies are parsed. N = lane count (4/8/16).
//
// Scalar (simd_none) is also a tier object (it needs no vector types, so it is
// the COSYVOICE_NO_SIMD dispatch target). simd_sse42_fma is only reached on
// non-x86 (SIMDe) builds.
// ---------------------------------------------------------------------------

#include "simd-dispatch.h"

#include <algorithm>
#include <cmath>

// Dependent vector-type trait. Specializations live in the tier object sources
// (the only TUs where the concrete __m128/__m256/__m512 types exist); kernels
// never name the intrinsic types directly, so e.g. the AVX2 object never needs
// __m512 while parsing the discarded AVX-512 branches. N = lane count (4/8/16).
template <simd_caps C, int N> struct simd_vec;


// ---------------------------------------------------------------------------

// The single public FFT tree entry point. The caller dispatches once per
// (sub)transform; the recursive butterfly stages then run entirely inside the
// tier object. The fft_bfly{2,3,4,5}_kernel / fft_bfly_generic_kernel classes
// are internal to simd-kernels-impl.h: fft_work_kernel's body calls and
// transitively instantiates them (same TU, inlinable, no per-node dispatch).
template<simd_caps C>
struct fft_work_kernel
{
    static void run(float* Foutr, float* Fouti, const float* f, int fstride, int in_stride,
        int* factors, float* scratchr, float* scratchi, float* tw_r, float* tw_i, int Norig);
};

template<simd_caps C>
struct fft_abs_kernel
{
    static void run(const float* foutr, const float* fouti, float* buffer, int n);
};

template<simd_caps C>
struct init_twiddles_kernel
{
    static void run(float* twiddles_r, float* twiddles_i, int n, float factor);
};

template<simd_caps C>
struct safe_divide_kernel
{
    static void run(float* data, const float* denom, size_t n, float min_denom);
};

template<simd_caps C>
struct apply_window_kernel
{
    static void run(const float* src, const float* window, float* dst, size_t n);
};

template<simd_caps C>
struct mel_dot_kernel
{
    static float run(const float* a, const float* b, size_t n);
};

template<simd_caps C>
struct mel_dot_sq_kernel
{
    static float run(const float* a, const float* b, size_t n);
};

template<simd_caps C>
struct sum_kernel
{
    static float run(const float* data, size_t n);
};

template<simd_caps C>
struct log_map_kernel
{
    static void run(float* data, size_t n, float min_level);
};

template<simd_caps C>
struct log10_map_kernel
{
    static float run(float* data, size_t n, float min_level);
};

template<simd_caps C>
struct spec_normalize_kernel
{
    static void run(float* data, size_t n, float base);
};

template<simd_caps C>
struct sub_mean_kernel
{
    static void run(float* dst, const float* src, size_t n, float mean);
};

template<simd_caps C>
struct preemph_gain_kernel
{
    static void run(float* frames, const float* prev, const float* window, size_t n, size_t win_size);
};

template<simd_caps C>
struct square_store_kernel
{
    static void run(const float* src, float* dst, size_t n);
};

template<simd_caps C>
struct column_mean_sub_kernel
{
    static void run(float* column, size_t rows, size_t stride);
};

template<simd_caps C>
struct div_kernel
{
    static void run(float* data, size_t n, float divisor);
};

#ifdef SIMD_TIER_IMPL
#define SIMD_INSTANTIATE_KERNELS(SIMD_PRESET) \
    template struct fft_work_kernel<SIMD_PRESET>; \
    template struct fft_abs_kernel<SIMD_PRESET>; \
    template struct init_twiddles_kernel<SIMD_PRESET>; \
    template struct safe_divide_kernel<SIMD_PRESET>; \
    template struct apply_window_kernel<SIMD_PRESET>; \
    template struct mel_dot_kernel<SIMD_PRESET>; \
    template struct mel_dot_sq_kernel<SIMD_PRESET>; \
    template struct sum_kernel<SIMD_PRESET>; \
    template struct log_map_kernel<SIMD_PRESET>; \
    template struct log10_map_kernel<SIMD_PRESET>; \
    template struct spec_normalize_kernel<SIMD_PRESET>; \
    template struct sub_mean_kernel<SIMD_PRESET>; \
    template struct preemph_gain_kernel<SIMD_PRESET>; \
    template struct square_store_kernel<SIMD_PRESET>; \
    template struct column_mean_sub_kernel<SIMD_PRESET>; \
    template struct div_kernel<SIMD_PRESET>;
#else
#define SIMD_EXTERN_KERNELS(SIMD_PRESET) \
    extern template struct fft_work_kernel<SIMD_PRESET>; \
    extern template struct fft_abs_kernel<SIMD_PRESET>; \
    extern template struct init_twiddles_kernel<SIMD_PRESET>; \
    extern template struct safe_divide_kernel<SIMD_PRESET>; \
    extern template struct apply_window_kernel<SIMD_PRESET>; \
    extern template struct mel_dot_kernel<SIMD_PRESET>; \
    extern template struct mel_dot_sq_kernel<SIMD_PRESET>; \
    extern template struct sum_kernel<SIMD_PRESET>; \
    extern template struct log_map_kernel<SIMD_PRESET>; \
    extern template struct log10_map_kernel<SIMD_PRESET>; \
    extern template struct spec_normalize_kernel<SIMD_PRESET>; \
    extern template struct sub_mean_kernel<SIMD_PRESET>; \
    extern template struct preemph_gain_kernel<SIMD_PRESET>; \
    extern template struct square_store_kernel<SIMD_PRESET>; \
    extern template struct column_mean_sub_kernel<SIMD_PRESET>; \
    extern template struct div_kernel<SIMD_PRESET>;

SIMD_EXTERN_KERNELS(simd_none)
SIMD_EXTERN_KERNELS(simd_sse42)
SIMD_EXTERN_KERNELS(simd_sse42_fma)
SIMD_EXTERN_KERNELS(simd_avx)
SIMD_EXTERN_KERNELS(simd_avx2)
SIMD_EXTERN_KERNELS(simd_avx512)
SIMD_EXTERN_KERNELS(simd_avx10_1_256)
#endif
