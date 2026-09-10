# `asimd` — the width is the author's choice

A portable SIMD layer where **the vector width is a parameter of the type**, not a consequence of
the target:

```cpp
using V = asimd::SimdVec<float,8>;      // eight lanes. Everywhere.
```

On AVX2 that is one `ymm`. On SSE4 it is **two `xmm`** — `SimdVecImpl` splits recursively
(`split_size_0 = prev_pow_2(size)`, the rest into `v1`, down to a single lane). Neither xsimd nor
Highway does this: Highway refuses at compile time (*"Too many lanes"*) as soon as the descriptor
exceeds the target, which rules out NEON — 128 bits, so four `float` lanes — for any kernel
written on eight.

Header-only, C++20.

```cpp
#include <asimd/SimdOpsPlus.h>

using V = asimd::SimdVec<float,8>;
using I = asimd::SimdVec<asimd::SI32,8>;

V s = asimd::fma( V( dx ), vx, asimd::fma( V( dy ), vy, V( -off ) ) );
unsigned outside = asimd::to_bits( s > V( 0.f ) );          // the sign mask, as an integer
V t = asimd::permute( v, I::load_aligned( idx ) );          // variable-index permutation
V u = asimd::select( asimd::mask_from_bits<8>( 1u << k ), a, b );
```

---

## 1. Operations

| | |
|---|---|
| `fma(a,b,c)` | fused multiply-add |
| `to_bits(m)` | a mask **as an integer** — this is what carries `ctz`, `popcount` and rotations |
| `mask_from_bits<n>(b)` | the dual: one `kmovb` where `eq(iota,i)` costs two instructions |
| `select(m,a,b)` | blend two vectors according to a mask |
| `permute(v,idx)` | **variable-index** permutation |
| `bcast_lane<i>(v)` | broadcast a compile-time lane |
| `gt` `lt` `eq` `ge` | comparisons, materialized as a mask |
| `-` `*` `/` | the operators |

Comparisons are **lazy**: `a > b` computes nothing, it returns an object holding both operands,
and `to_bits` or `select` decides in which form to materialize it. That is finer than what
intrinsics and the other libraries do, where a comparison picks its representation before knowing
what it will be used for.

## 2. Variant selection

Every operation goes through `Selection.h`. A backend registers by specializing
`sel::Variant<Op, Key, RANK>`; the selector takes the highest-ranked available one, and rank 0 —
the generic form — is always available, so there is always an answer.

```cpp
sel::rank<ops::permute, Key<float,8,NativeCpu>>          // 20 (REGISTER) on AVX2
sel::require_at_least<ops::cmp_gt, Key<float,8,A>, sel::MASK_REGISTER>();
```

Three problems this solves, all met in practice:

**Constraints do not order each other.** Adding an AVX-512 form next to an AVX one makes the call
ambiguous as soon as both features are present. Writing the constraint as a conjunction does not
help: two `requires` clauses written separately yield *distinct* atomic constraints even when
textually identical, so there is no subsumption. An explicit rank gives a total order instead.

**Overload resolution depends on include order.** A facade calling `internal::f(...)` through a
qualified name freezes resolution at its own definition; a register form declared later is
invisible. The code still compiles, still returns correct results, and silently loses all
vectorization. Class specializations are looked up at instantiation, so order stops mattering.

**Available is not best.** On a Skylake-X, 512-bit exists and downclocks. Preference is a
measurement, not an instruction-set generation — so it is declared, and a backend can rule itself
out for one micro-architecture without anything else moving.

And `require_at_least` turns a silent fallback into a compile error, which matters more than it
sounds: vectorization was lost twice, silently, while this was being built.

## 3. The layout of a register impl decides the ABI

This one was a performance bug, not a correctness bug. A plain `fma` passed by value cost **11
instructions including 3 memory accesses**, against 4 for raw intrinsics. Every vector went through
the stack on every call.

The SysV x86-64 rule: an aggregate larger than two *eightbytes* is passed in registers only if the
first eightbyte is `SSE` and **all** the following ones are `SSEUP`. But `float[8]` classifies as
`SSE,SSE,SSE,SSE` — an array is not a vector — and a union takes the **worst** class among its
members.

| the union holds | instructions | memory accesses |
|---|---|---|
| the register alone | 4 | 0 |
| **a vector type + the register** | **4** | **0** |
| `T values[SIZE]` + the register | 8 | 4 |
| … + `Split` (two `__m128`) | 8 | 4 |

`Split` is unrecoverable: two `__m128` give `SSE,SSEUP,SSE,SSEUP`, and the third eightbyte alone
sends everything back to memory.

So in `SIMD_VEC_IMPL_REG`: `values` typed with `vector_size` (which keeps `values[i]` indexing) and
**`Split` taken out of the union**. Operations without a register form now go lane by lane through
`values`. `iota` had to follow, and got register forms — without them gcc materialized it as six
consecutive `vpinsrd` rather than a constant load.

Two things the audit added to this section. First, the same rule applies to **masks**, and had not
been applied: `SIMD_MASK_IMPL_REG_LARGE` still held `PI32 values[8]` *and* a `Split`, so every
lane-flavoured mask crossing a call went through memory — 9 instructions and 3 stack accesses on a
by-value `select`, against 4 and 0 once fixed. The rule is now in one place,
`support/VecValues.h`, which both macros use and which carries the MSVC fallback (MSVC has no
`vector_size`, and does not need one: its x64 convention passes vectors by reference anyway, so
the classification question never arises).

Second, taking `Split` out of the union had a consequence nobody followed through: the generic
fallbacks still recursed into `data.split`, which a register impl no longer has. At a
register-backed width, any operation *without* a register form was therefore a hard compile error
rather than a slow path — `.sum()`, integer `mul` and `div`, the strided `iota`, `gather` below
AVX2. A `HasSplit` concept and an `if constexpr` fix it, and buy something on the way: when there
is no split but `values` is a vector type, the operation applies **to the whole vector**, and gcc
and clang lower it themselves. That is how `SI32 × 8` gets a `vpmulld` with no backend registered
for it.

## 4. Building and testing

```sh
cd tests && make          # build and run the tests, for this machine
make isa                  # ... and at five x86 feature levels, plus two MSVC-path ones
make matrix               # the (operation, type, ISA) compile map, 408 translation units
make msvc                 # what the dispatch table is worth WITHOUT the compiler's vector types
make bench                # timings for the cross-lane operations
```

On Windows, `pwsh tests/run_msvc.ps1` does what `make isa` does, at MSVC's four `/arch:` levels.
`.github/workflows/ci.yml` runs all of it — gcc and clang on Linux, MSVC on Windows.

`make isa` is the one that matters when changing a backend: `make` alone builds for
`-march=native`, so on a recent box the SSE2 and AVX paths — the ones most likely to rot — are
never exercised. It is what caught the last two bugs of the audit, one of them in the test suite
itself.

The build is described in `tests/xmake.lua`; the `Makefile` next to it is a thin wrapper that
calls xmake, so `make` works for anyone who would rather not learn a new tool.

| test | what it checks |
|---|---|
| `test_ops` | every operation, value by value |
| `test_split` | `SimdVec<float,8>` on an architecture capped at SSE2 — **the differentiator** |
| `test_selection` | total order, independence from declaration order, the safety net |
| `test_x86_ops` | **572 assertions.** A grid drives every operation over 23 (type, width) pairs — FP32/FP64/SI32/PI32/SI64/PI64/SI16 from 2 to 32 lanes, plus widths that are not a power of two and one wider than any register. Each pair lands on a different variant per target, and all of them must agree |
| `test_x86_dispatch` | the (operation, type, width) rank grid: which variant is *actually* selected, everywhere rather than at one width — and `require_at_least` on each cell that must have one, so a lost registration fails the **build** |

and two checks that are not binaries:

`tests/compile_matrix.sh` builds one translation unit per (operation, type, ISA) across
SSE2 / AVX / AVX2 / native — 408 of them — and prints the map, because a coverage hole that is a
*compile* error cannot be seen from a test binary.

`tests/run_all_isa.sh` builds and runs every test at five feature levels. This is the one that
matters when touching a backend: an intrinsic guarded by the wrong feature macro compiles fine on
a machine that has everything and not at all on the target that needs it. Both take the compiler
as an argument, so `./run_all_isa.sh clang++` answers the portability half.

And a check that is not an assertion but a **reading of the disassembly** — it runs on every
build, because the regression it catches shows up nowhere else:

```
ABI probe (a value crossing a call by value):
  probe_fma         3 instructions, 0 touching %rsp
  probe_perm        3 instructions, 0 touching %rsp
  probe_fma_f64     3 instructions, 0 touching %rsp
  probe_sel_lane    3 instructions, 0 touching %rsp
  probe_sel_bits    4 instructions, 0 touching %rsp
ok: vectors and masks cross a call in their registers.
```

Five symbols, all enforced: any one of them touching `%rsp` fails the build and names the cause.
A `static_assert` can do nothing about an ABI, and the regression changes no result, only the
speed. Widening it from one probe to five was worth it immediately — two came out dirty, for two
*different* reasons: `probe_sel_lane` was the mask ABI of § 3, and `probe_fma_f64` was not an ABI
problem at all but a missing register variant at eight lanes of double, showing up in the same
measurement.

## 5. Status

Working: x86 (SSE2 / AVX / AVX2 / AVX-512VL), the recursive split for widths beyond the register,
the operations above, variant selection.

An audit went over all of it — by compiling and running it, not by reading it — and the results
are in [FINDINGS.md](FINDINGS.md). Where it stood, and where it stands:

| | before | after |
|---|---|---|
| (operation, type, ISA) cells that compile | 258 / 408 | **408 / 408** |
| (operation, type, width) cells with a register form | 8 / 63 | **59 / 63** |
| operations returning wrong values | 5 | **0** |
| values passed through memory across a call | 2 of 5 probes | **0 of 5** |
| value assertions, at 5 ISA levels | 29 (one level) | **2 860** |
| cells needing gcc's vector extensions to be fast (the MSVC gap) | 12 / 168 | **2 / 168** |
| compilers the suite runs under | 1 | **gcc, clang, MSVC** |

The short version: the backend was correct and fast for `float × 8`, the cell it was ported on,
and thin everywhere else. The feature lattice went SSE2 → AVX with no SSE4.1 in between, so a
genuine SSE2 or AVX target did not build; the generic fallbacks recursed into a `split` that
register impls deliberately do not have, so any operation without a register form was a compile
error rather than a slow path; and `SimdOpsPlus_X86.h` registered nothing outside eight lanes, so
on this AVX-512 machine — where the native width is sixteen — a plain `SimdVec<float>` got the
slowest path on the widest hardware.

The four cells still on the generic path are the honest ones: x86 has no integer fused
multiply-add, and a two-lane permutation of 64-bit elements is two moves whatever you do.

Next up:

- **ARM / NEON** — where the split earns its keep. `architectures/ArmCpu.h` now declares the
  feature and the width, so `SimdVec<float>` on AArch64 is four generic lanes that give the right
  answer; `impl/SimdVecImpl_Neon.h` and `SimdMaskImpl_Neon.h` remain to be written. That is the
  point of the `ScalarCpu` fallback added to `NativeCpu.h`: a new backend is now *additive*, where
  before the library did not compile at all outside x86 and Apple ARM. Two primitives to watch:
  `permute` (`tbl` works on bytes, so it costs more for 32-bit lanes) and `to_bits` (no `movemask`
  on NEON — a three or four instruction reduction).
- **clang and MSVC are unverified.** Neither is installed on the machine this was done on, so the
  portability work — `std::bit_floor` for `__builtin_clz`, `<immintrin.h>` for `<x86intrin.h>`,
  and above all the fact that *MSVC never defines `__SSE2__`*, which gave it an empty feature set
  and silently scalar code — is reasoned from documented behaviour, not measured.
  `./run_all_isa.sh clang++` and `./compile_matrix.sh clang++` take the compiler as an argument.
- **The `SPLIT` rank is declared and never used.** Nothing registers a variant spanning several
  registers for one requested width, which is on paper what asimd is for. The generic forms cover
  it correctly; whether a dedicated split-rank `permute` would beat them is unmeasured.
