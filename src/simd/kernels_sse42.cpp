// SSE4.2-class tier object. Compiled with -msse4.2 (no AVX/AVX-512), so it
// instantiates the kernels for the SSE4.2 presets with only the 128-bit types.
#define SIMD_TIER_IMPL
#include "../simd-kernels.h"
#include "../simd-math.h"

// simd_sse42_fma (SSE4.2 + FMA3) exists only where SIMDe emulates FMA (non-x86):
// on x86 an SSE4.2-only object cannot legally emit the VEX `_mm_fmadd_ps` and the
// preset is never dispatched there anyway (FMA3 implies AVX, which sse42 lacks).
#ifdef SIMDE_ENABLE_NATIVE_ALIASES
template <> struct simd_vec<simd_sse42_fma, 4> { using f = __m128; using i = __m128i; };
#else
template <> struct simd_vec<simd_sse42, 4> { using f = __m128; using i = __m128i; };
#endif

#include "../simd-kernels-impl.h"

#ifdef SIMDE_ENABLE_NATIVE_ALIASES
SIMD_INSTANTIATE_KERNELS(simd_sse42_fma)
#else
SIMD_INSTANTIATE_KERNELS(simd_sse42)
#endif
