// AVX10.1-256 tier object. Compiled with -mavx10.1-256 / /arch:AVX10.1 /vlen=256:
// shares the AVX 8-wide bodies and masks its own tail with
// __mmask8 instead of falling through to the 128-bit vector tail. No zmm
// encoding exists in this TU (the HAS_AVX512/HAS_AVX10_*_512 blocks are not
// compiled here).
#define SIMD_TIER_IMPL
#include "../simd-kernels.h"
#include "../simd-math.h"

template <> struct simd_vec<simd_avx10_1_256, 8> { using f = __m256; using i = __m256i; using mask = __mmask8; };
template <> struct simd_vec<simd_avx10_1_256, 4> { using f = __m128; using i = __m128i; };

#include "../simd-kernels-impl.h"

SIMD_INSTANTIATE_KERNELS(simd_avx10_1_256)
