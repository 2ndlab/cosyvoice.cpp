// AVX-class tier object. Compiled with -mavx (no AVX2/AVX-512): instantiates
// the kernels for the AVX preset with 256-bit + 128-bit types only.
#define SIMD_TIER_IMPL
#include "../simd-kernels.h"
#include "../simd-math.h"

template <> struct simd_vec<simd_avx, 8> { using f = __m256; using i = __m256i; };
template <> struct simd_vec<simd_avx, 4> { using f = __m128; using i = __m128i; };

#include "../simd-kernels-impl.h"

SIMD_INSTANTIATE_KERNELS(simd_avx)
