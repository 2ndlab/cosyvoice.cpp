// AVX2-class tier object. Compiled with -mavx -mavx2 -mfma (no AVX-512):
// instantiates the kernels for the AVX2 preset with 256-bit + 128-bit types.
#define SIMD_TIER_IMPL
#include "../simd-kernels.h"
#include "../simd-math.h"

template <> struct simd_vec<simd_avx2, 8> { using f = __m256; using i = __m256i; };
template <> struct simd_vec<simd_avx2, 4> { using f = __m128; using i = __m128i; };

#include "../simd-kernels-impl.h"

SIMD_INSTANTIATE_KERNELS(simd_avx2)
