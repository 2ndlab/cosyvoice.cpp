# SIMD Tier Architecture (Developer Reference)

User-facing overview: [README.md — SIMD Acceleration](../README.md#simd-acceleration).
This document covers the internal layout: file structure, capability model,
dispatch rules, the AVX10 policy, and the build-time inclusion knobs.

## File layout

| File | Role |
|---|---|
| `src/simd-dispatch.h` | `simd_caps` struct, presets, `simd_dispatch<Kernel>` — no intrinsic types, safe for caller TUs |
| `src/simd_detect.cpp` | CPUID/XCR0 feature detection → `g_simd_caps` (x86-64 only) |
| `src/simd-kernels.h` | Kernel class declarations + `simd_vec<C, N>` vector typedefs + extern-template declarations |
| `src/simd-kernels-impl.h` | All kernel bodies + support helpers, templated on `simd_caps C` — included **only** by tier TUs |
| `src/simd-math.h` | Vector log/sincos helpers (128/256/512) — included **only** by tier TUs |
| `src/simd/kernels_<tier>.cpp` | One TU per tier: defines `SIMD_TIER_IMPL`, includes the headers above, instantiates its own preset |

Each tier TU is compiled with **only its own ISA flags** (GCC/Clang: `-msse4.2`,
`-msse4.2 -mavx`, ..., `-mavx512f -mavx512bw -mavx512dq`, `-mavx10.1-256`;
MSVC: `/arch:SSE4.2`, `/arch:AVX`, `/arch:AVX2`, `/arch:AVX512`,
`/arch:AVX10.1 /vlen=256`). This bounds every emitted instruction —
intrinsics *and* compiler auto-vectorization — to the tier's ISA, while all
caller TUs stay at the plain x86-64 baseline. The scalar tier deliberately
carries no flags on any toolchain.

Kernel call sites resolve at link time: the caller objects reference
`Kernel<simd_xxx>::run` (declared `extern template` in `simd-kernels.h`), and
the tier objects provide the definitions.

## Capability model

```cpp
struct simd_caps {          // one bit per class, bitfield-packed
    bool sse42, avx, fma3, avx2, avx512, avx10_1_256, avx10_1_512;
};
```

Presets are `constexpr simd_caps` values used as template arguments:
`simd_none`, `simd_sse42`, `simd_sse42_fma`, `simd_avx`, `simd_avx2`,
`simd_avx512`, `simd_avx10_1_256`.

**Presets enumerate exactly their own class** (no upward inheritance beyond
the legacy bits the class genuinely implies): the AVX10 presets carry only
`fma3` + their own bit. Shared code paths therefore list every consumer
explicitly in both layers — the macro gate and the matching
`if constexpr` condition, e.g.

```cpp
#if defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
    if constexpr (C.avx || C.avx10_1_256)
```

The `COSYVOICE_HAS_*` macros are **load-bearing**: they gate every intrinsic
block with the preprocessor, so a platform without the declarations (non-x86
without SIMDe) never even parses them. They must stay consistent with the
`if constexpr` conditions — a shared lower-class block IS the main loop /
vector tail of every class that reuses it.

The chain uses **independent `if constexpr`s, never `else`** across class
boundaries: a tier's 512 or masked-256 branch returns after its own tail, and
the presets' bits are disjoint, so fall-through is impossible. An `else` would
dangle onto whatever the next macro gate leaves behind and break
configurations like "AVX10-256 enabled, SSE4.2/AVX/AVX2 disabled".

## Detection (`simd_detect.cpp`)

- CPUID leaf 1 → `sse42`, `osxsave`, `avx`, `fma3`.
- `XGETBV(0)` gate: bits 1+2 (XMM/YMM state) required for any AVX-class bit.
- CPUID leaf 7 subleaf 0 → `avx2`; `avx512` lights only with **F+DQ+BW+VL**
  together *and* `XCR0 & 0xE6 == 0xE6` (opmask + ZMM state). VL is required
  even though the 512 branch never emits 128/256-bit EVEX itself, because the
  compilers' own spill forms use EVEX xmm16+/ymm16+. Partial parts (Knights
  Landing: F alone) fall through to AVX2.
- **AVX10** — CPUID leaf 0x24H (independent of the legacy bits, because
  AVX10-only parts do not enumerate them): `EAX[7:0]` = version (1 = AVX10.1,
  2 = AVX10.2), `EBX bit16` = 256-bit, `EBX bit17` = 512-bit.
  256 class also requires `XCR0 & 0x26 == 0x26` (opmask); 512 the full
  `0xE6`. Version ≥ 1 gates both classes; 10.2 lights the same bits as 10.1.
- The result is cached in `g_simd_caps` (one-time static init); Debug builds
  log the detected set at startup.

## Dispatch

`simd_dispatch<Kernel>(args...)` is one inline template shared by every
caller TU. Order: **512 (legacy ∨ AVX10) → AVX10.1-256 → AVX2 → AVX →
SSE4.2 → scalar**. A class the build disabled is compiled out of the chain
(`#if defined(COSYVOICE_HAS_*)` around each case), so the CPU always lands on
the best *available* tier; if none remains the call throws. Under
`COSYVOICE_NO_SIMD` the dispatch collapses to `Kernel<simd_none>`. On
non-x86 the dispatch is not used at all — it returns
`Kernel<simd_sse42_fma>` (SIMDe-emulated) directly, and the scalar tier object
is not even built on those targets.

## AVX10 policy

AVX10.1 **512-bit** parts are served by the **AVX-512 tier's kernels**;
there is no separate 512 tier object. Rationale (verified):

- MSVC: `/arch:AVX512` and `/arch:AVX10.1 /vlen=512` produce **byte-identical**
  tier objects (line-level diff of the disassembly is empty).
- Clang: `-mavx512f -mavx512bw -mavx512dq` output vs `-mavx10.1-512` differs
  only in a handful of VEX-vs-EVEX width *encoding* choices; the legacy tier's
  instruction space (VEX.128/256 + EVEX.512, F/BW/DQ/VL subsets, gather
  included) is a strict subset of AVX10.1-512. The exotic instructions AVX10
  removed (MASKMOVDQU, VPOPCNTDQ, ...) are not used.
- The XCR0 requirement is identical for both enumeration paths.

Side benefit: toolchains without any AVX10 flag support (e.g. gcc < 14) still
serve AVX10-512 CPUs, since only `-mavx512*` flags are needed to build the
tier.

AVX10.1 **256-bit** parts need their own tier: the k-masked 256-bit vector
tail (`_mm256_maskz_loadu_ps` / `_mm256_mask_storeu_ps`, `__mmask8`) and the
EVEX k-masked log/sincos helpers (`_mm256_cmp_ps_mask`,
`_mm256_cmp_epi32_mask`, `_mm256_maskz_mov_ps`, `_mm256_mask_blend_ps`,
`_mm256_mask_xor_epi32`) cannot be emitted by an AVX2 build, and a 256-only
part must never execute the 512 tier (no ZMM state). The class runs the
shared AVX 8-wide bodies and replaces only the tail.
`simd_vec<simd_avx10_1_256, 8>::mask = __mmask8` is the typedef that makes
the tail code shared.

**AVX10.2** gets no tier at all: it is a strict superset of 10.1 and no
kernel uses a 10.2-exclusive instruction (FP16 arithmetic, `VFPCLASS`,
bf16 helpers, `V4FMADD`, ...). If a kernel ever exploits one, re-add a
dedicated class: a new caps bit + preset + tier TU + dispatch case above the
10.1 case, keeping the "preset = exactly its own class" rule.

## Build-time inclusion knobs

| CMake cache entry | Compile definition | Gate |
|---|---|---|
| `COSYVOICE_NO_SIMD` | `COSYVOICE_NO_SIMD` | kills all SIMD code (auto-set on non-x86 without SIMDe) |
| `COSYVOICE_HAS_SCALAR` | `COSYVOICE_HAS_SCALAR` | scalar tier object; also built whenever `x86` (dispatch fallback) — on non-x86 + SIMDe it is skipped (never dispatched) |
| `COSYVOICE_HAS_SSE42` / `_AVX` / `_AVX2` / `_AVX512` | same names | each legacy tier |
| `COSYVOICE_HAS_AVX10_1` | `COSYVOICE_HAS_AVX10_1_256` | the AVX10-256 tier only (plus a compiler-flag probe) |

- Disabling SSE4.2 cascades AVX/AVX2/AVX-512 off (shared code paths); the
  AVX10-256 class sits outside the cascade (it consumes no SSE4.2-only blocks
  in its instantiated branches).
- Tier objects and all library targets receive the same uniform macro set;
  the per-tier *flags* are what specialize the codegen.
- The AVX10-256 probe (`check_cxx_source_compiles` with `-mavx10.1-256` /
  `/arch:AVX10.1 /vlen=256`) silently drops the class on toolchains that
  don't know the flags.
- LTO (`CMAKE_INTERPROCEDURAL_OPTIMIZATION`, ON by default) additionally
  deduplicates byte-identical tier instantiations at link time (MSVC ICF /
  GCC-Clang ICF). Verified safe with mixed `/arch` inputs: LTCG records the
  per-TU target and codegens each function at its own ISA.

## Non-x86

SIMDe (when found) emulates exactly the SSE4.2+FMA3 class; the SIMD section
in the README's Build chapter covers detection and options. Everything above
the 128-bit class is compile-out on those targets (the `COSYVOICE_HAS_*`
macros for AVX/AVX2/AVX-512/AVX10 are never defined there), so the
intrinsics-gated bodies never reach the parser.
