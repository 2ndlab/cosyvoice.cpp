// Scalar (simd_none) tier object. No SIMD flags: instantiates the kernels'
// scalar fallback bodies only. In the body-separation design the kernel bodies
// live in simd-kernels-impl.h, so even the scalar path is explicit-instantiated
// in its own TU and callers just emit external calls.
#define SIMD_TIER_IMPL
#include "../simd-kernels.h"
#include "../simd-kernels-impl.h"

SIMD_INSTANTIATE_KERNELS(simd_none)
