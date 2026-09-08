// AVX-512 tier object. Compiled with
// -msse4.2 -mavx -mavx2 -mfma -mavx512f -mavx512bw -mavx512dq (the exact
// sub-set set the tier requires; see simd-dispatch.h). On MSVC the gather load
// path is still the 2x256 VEX split to dodge the EVEX-gather VSIB reserved-
// encoding #UD; the rest lowers to whatever the intrinsics demand.
#define SIMD_TIER_IMPL
#include "../simd-kernels.h"
#include "../simd-math.h"

template <> struct simd_vec<simd_avx512, 16> { using f = __m512; using i = __m512i; using mask = __mmask16; };
template <> struct simd_vec<simd_avx512, 8>  { using f = __m256; using i = __m256i; };
template <> struct simd_vec<simd_avx512, 4>  { using f = __m128; using i = __m128i; };

#include "../simd-kernels-impl.h"

SIMD_INSTANTIATE_KERNELS(simd_avx512)
