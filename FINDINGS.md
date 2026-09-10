# asimd — x86 audit, and what was done about it

State of the x86 backend, established by compiling and running it rather than by reading it, then
repaired. Everything below is reproduced by `make`, `make isa` or `make matrix` in `tests/`.

Measured on `g++ 15.2`, Xeon W-2145 (Skylake-X: SSE2 … AVX-512VL/BW/DQ).

## Where it stood, and where it stands

| | before | after |
|---|---|---|
| (operation, type, ISA) cells that compile | 258 / 408 | **408 / 408** |
| (operation, type, width) cells with a register form | 8 / 63 | **59 / 63** |
| operations returning wrong values | 5 | **0** |
| values passed through memory across a call | 2 of 5 probes | **0 of 5** |
| value assertions, at 5 ISA levels | 29 (one level) | **2 860** |
| public headers that compile standalone | 4 / 7 | **7 / 7** |
| cells that need gcc's vector extensions to be fast (the MSVC gap) | 12 / 168 | **2 / 168** |
| compilers the suite is run under | 1 | **2 locally, 3 in CI** |

The four remaining generic cells are legitimate: x86 has no integer fused multiply-add (three
cells), and a two-lane permutation of 64-bit elements is two moves whatever you do.

---

## 1. Wrong results, silently — all five fixed

**1.1 `all()` on an AVX-512 mask register was always `false`.**
`~mask.data.reg == 0` integer-*promotes* `__mmask8` to `int` before complementing, so a full mask
gave `~255 == -256`. Only `__mmask64` escaped. gcc had been printing
`warning: promoted bitwise complement of an unsigned value is always nonzero` on that exact line
all along — it was the only warning the library emitted, and nobody was reading it.
→ complement inside the mask type. `impl/SimdMaskImpl_AVX512.h`

**1.2 `min` / `max` on unsigned types used the SIGNED instruction.**
`PI32`/`PI64` went to `_mm*_min_epi32/64`, so `min( 0xFFFFFFFF, 9 )` returned `0xFFFFFFFF`.
→ `_epu` forms throughout, and the 64-bit ones moved to AVX-512VL where they actually exist.
`impl/SimdVecImpl_{SSE2,AVX2,AVX512}.h`

**1.3 `permute` was wrong at every width that is not a power of two.**
`tmp[ idx & ( N - 1 ) ]`. At N = 5 that is `& 4`: reversing five lanes gave `50 10 10 10 10`.
Arbitrary widths are the reason asimd exists, so this was the generic form failing on the
library's own headline case. → `% N`, which folds to the same `and` at a power of two.
`SimdOpsPlus.h`

**1.4 AVX comparisons on 32-bit integers compared 64-bit lanes.**
`SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, PI32, 8, 32, …, _mm256_cmp_epi64( … ) )` — the wrong lane
width, signed for an unsigned type, and calling an intrinsic that **does not exist**. Three errors
on one line, which is why it had never compiled. → real `vpcmpgtd`/`vpcmpgtq` forms in the AVX2
file, and the unsigned ones done properly by flipping both sign bits. `impl/SimdVecImpl_AVX.h`

**1.5 `prefetch` asked for write ownership on a read-only pointer.**
`_mm_prefetch( beg, _MM_HINT_ET0 )` on a `const void *`. `ET0` emits `prefetchw`, which needs the
PREFETCHW feature and takes the line in exclusive state — on a buffer several threads read, that
turns a shared line into a ping-pong. → `_MM_HINT_T0`. `impl/SimdVecImpl_SSE2.h`

---

## 2. What did not compile — 150 of 408 cells, now 0

### 2.1 The structural one: generic fallbacks recursed into a `split` register impls do not have

`SIMD_VEC_IMPL_REG` deliberately took `Split` out of the union — that is the ABI fix of README § 3,
and it is right. But only `iota( beg )` was moved over to `values`. Every other generic form still
wrote `a.data.split.v0`, so **at a register-backed width, any operation with no register form was a
hard compile error** rather than a slow path: `.sum()`, `mul` and `div` on every integer type, the
strided `iota`, `gather` and `scatter` below AVX2, half the comparison fallbacks.

The fix is a concept and three branches, in `impl/SimdVecImpl_Generic.h`:

```cpp
template<class I> concept HasSplit = requires ( I i ) { i.data.split.v0; };
```

- has a split → recurse, as before;
- no split but `values` is a vector type → **apply the operation to the whole vector**, which gcc
  and clang lower to the right instruction on their own. This is how `SI32 × 8` gets a `vpmulld`
  without anyone registering a `mul` backend for it;
- otherwise → lane by lane.

That one change took the map from 150 broken cells to 46.

`and` needed its own version: it is a *bitwise* operation, so on a floating point type it is
neither `a & b` (ill-formed) nor arithmetic — `SimdVec<float,5> & …` used to fail because the
one-lane tail of the split reached `float & float`. It now goes through `std::bit_cast`.

### 2.2 256-bit integer operations were registered under `AVX`, and are AVX2 instructions

`vpaddd`, `vpminsd`, `vpand` at 256 bits are AVX2 — what AVX adds is the floating point unit and
the wider register file. They sat under `Has<features::AVX>`, so on a genuine AVX target (Sandy
Bridge, Ivy Bridge) the **entire integer column** failed to compile. The register *layouts* are
right at AVX and stayed; the operations moved to `impl/SimdVecImpl_AVX2.h`.

### 2.3 Intrinsics guarded by the wrong feature macro

| intrinsic | needs | was guarded by |
|---|---|---|
| `_mm_stream_load_si128`, `_mm_testz_si128`, `_mm_min_epi32` | SSE4.1 | `__SSE2__` |
| `_mm_min_epi64` / `_mm_max_epi64` | AVX-512VL | `__SSE2__` |
| `_mm256_stream_load_si256`, `_mm256_xor_si256` | AVX2 | `__AVX__` |
| `_mm256_i32scatter_*`, `_mm_i32scatter_*` | AVX-512VL | `__AVX512F__` |
| `_mm512_cmp_ep{i,u}{16,8}_mask` | AVX-512BW | `__AVX512F__` |

Underneath all of it: **SSE3, SSSE3, SSE4.1 and SSE4.2 were not modelled as features at all.** The
lattice went SSE2 → AVX, and a good part of what the SSE2 backend called lives in SSE4.1.
`architectures/X86CpuFeatures.h` now carries the full ladder plus FMA and AVX-512VL/BW/DQ as
markers — features that unlock instructions at a width someone else already provides, and so
contribute nothing to `SimdSize`.

Two incidental fixes fell out: `_mm_testz_si128` + `xor` for `all()` became a single
`_mm_testc_si128`, which is the instruction that answers that exact question; and the 8- and
16-bit comparisons, which were registered against `SimdVecImpl<PI16,32,Arch>` **for which no
register impl existed**, got the impl the lines were already assuming — plus load, store and
arithmetic, so `SimdVec<SI16>` is now a real 512-bit vector on AVX-512BW instead of 32 scalars.

### 2.4 Type errors that had simply never been instantiated

`(__m128)_mm_stream_load_si128( … )` assigned to a `__m128d`, and the same at 256 and 512 bits.
`_mm256_i32gather_epi64( data, … )` with `data` a `const SI64 *` — `long` against the intrinsic's
`long long`, which is the same type only in the abstract.

### 2.5 `all()` / `any()` on a lane mask

Generic `all()` called `mask.data.values.all()`, but only the *bit* flavour stores a `BitVec` with
an `all()`; the lane flavour is a plain array. And `any()` was declared only for `item_size == 1`.
Both are now `if constexpr` over the two flavours.

### 2.6 Three public headers had not compiled since the `src/` reorganisation

`Ptr.h`, `SimdRange.h` and `SimdRangePtr.h` still included `internal/…` and bare `N.h`. That is
why `SimdVec.h` carried a commented-out `//#include "Ptr.h"` and four test files were excluded
from the build. Fixing the paths exposed two more:

- `N.h` and `Int.h` include each other, and entering through `Int.h` — which is what `Ptr.h` does
  — left `Int` undeclared where `N.h`'s operators named it. A forward declaration breaks the cycle.
- `SimdRangePtr.h` used `Ptrs::T` (the member is `pointed_type`) and `SimdAlig<T,Arch>` (it takes
  `<T, simd_size, Arch>`). Neither had ever been checked by a compiler.

`SimdVec.h` includes `Ptr.h` again, so the aligned-pointer `load( P )` / `store( P )` overloads
are live rather than aspirational.

---

## 3. Dispatch: what actually got selected

### 3.1 `SimdOpsPlus_X86.h` implemented one cell, and it was not the native one

The grid `tests/test_x86_dispatch.cpp` prints, before:

```
  op \ (type,lanes)   f,4   f,8  f,16   d,2   d,4   d,8   i,4   i,8  i,16
  fma                  0    20     0     0     0     0     0     0     0
  permute              0    20     0     0     0     0     0    20     0
  cmp_gt               0    30     0     0     0     0     0     0     0
```

55 of 63 cells on the scalar path. The native width here is **16** floats — so `SimdVec<float>`,
written without naming a width, got rank 0 for `fma`, `permute`, `select`, `to_bits` and every
comparison. The ABI probe measured the consequence: `fma` at 8 lanes is 3 instructions and 0 stack
accesses, at 16 lanes it was **23 and 12**.

After, at every level from SSE2 up:

```
  fma                 20    20    20    20    20    20     0     0     0
  permute             20    20    20     0    20    20    20    20    20
  cmp_gt              30    30    30    30    30    30    30    30    30
```

The file is now laid out by feature level — 128 bits (SSE2/SSE4.1/FMA), 256 (AVX/AVX2/FMA), 512
(AVX-512F), then AVX-512VL for the mask registers at the narrower widths, which is what turns a
`vblendvps` back into a masked move at eight lanes on an AVX-512 machine.

**The mechanism was never the problem.** `require_at_least` would have caught every one of these;
it was simply never called outside a demonstration. `test_x86_dispatch.cpp` now calls it at each
width the library claims to support, guarded by the features, so a lost registration stops the
*build* rather than a benchmark six months later.

### 3.2 The mask ABI was not fixed alongside the vector ABI

README § 3 removed the array and `Split` from `SIMD_VEC_IMPL_REG`. `SIMD_MASK_IMPL_REG_LARGE` kept
both, so a lane-flavoured mask classified `SSE,SSE,SSE,SSE` → MEMORY. Same rule, same cost:

```
  probe_sel_lane    9 instructions, 3 touching %rsp     ->  3 instructions, 0
  probe_fma_f64    23 instructions, 12 touching %rsp    ->  3 instructions, 0
```

The second one was not an ABI problem at all — it was § 3.1, the missing `fma` at eight lanes of
double, showing up in the same measurement. Which is the argument for probing five symbols instead
of one. `support/VecValues.h` now holds the vector-typed-`values` trick in one place, with the
MSVC fallback, and both impl macros use it.

### 3.3 Smaller ones, also fixed

- `cmp_gt` for `float` was inside `#ifdef __AVX2__` while its `requires` only asked for AVX: on an
  AVX-only build it fell to generic for nothing.
- no integer `mul` register form existed anywhere; `_mm_mullo_epi32` (SSE4.1),
  `_mm256_mullo_epi32`, `_mm512_mullo_epi32` and `_mm512_mullo_epi64` (AVX-512DQ) now do.
- `to_bits` returned `unsigned`, which cannot carry a 64-lane mask, and its generic form shifted
  by `i >= 32` — undefined behaviour on exactly the widths that need it. It returns `PI64` now.
- no integer comparison was registered at 128 bits at all, though `pcmpgtd` has been there since
  SSE2.

---

## 4. Portability: clang, MSVC, and the arches to come

| | what it was | what it is |
|---|---|---|
| `__builtin_clz` in `prev_pow_2` | absent on MSVC, and `__builtin_clz( 0 )` made `prev_pow_2( 1 )` undefined behaviour in a function evaluated for every vector width | `std::bit_floor`, C++20, `constexpr`, total |
| `__attribute__(( vector_size ))` | gcc/clang only, and it *is* the ABI fix | `support/VecValues.h`, with an array fallback; the `requires { values + values }` test then routes those operations lane by lane instead |
| `#include <x86intrin.h>` (9 files) | gcc/clang only | `<immintrin.h>` via `impl/x86_intrin.h`, one place that knows what each compiler offers |
| **MSVC never defines `__SSE2__`, `__SSE4_1__`, `__FMA__`** | so `NativeCpu` on MSVC x64 was `X86Cpu<64>` with an **empty feature set** — everything silently scalar, the exact failure `Selection.h` exists to prevent, on one of the three required compilers | `_M_X64` / `_M_IX86_FP` / `/arch:` mapped explicitly, in `NativeCpu.h` and `x86_intrin.h`, which are written to be read together |
| no `#else` in `NativeCpu.h` | on any target that was neither x86 nor Apple's `__arm64__`, `NativeCpu` was not declared and nothing compiled | `ScalarCpu` — one lane, everything generic. This is what makes each new backend *additive* |
| `#if defined( __arm64__ )` | Apple's spelling; the ARM branch was unreachable under gcc/clang on Linux (`__aarch64__`) or MSVC (`_M_ARM64`) | all three, plus 32-bit ARM, plus a `features::NEON` that declares the width so `SimdVec<float>` is four generic lanes rather than a compile error |
| `-march=native` hard-coded in `xmake.lua` | contradicted the cross-build recipe in that same file's header | an `arch_flags` option, and `make isa` |
| missing includes, duplicate `using PI32` | worked only through transitive includes | fixed |

**Not verified here:** clang and MSVC are not installed on this machine, so the portability work
above is reasoned from their documented behaviour and not measured. `./run_all_isa.sh clang++` and
`./compile_matrix.sh clang++` are written to take the compiler as an argument, and are the first
thing to run where one exists.

---

## 5. The tests

| file | what it does |
|---|---|
| `test_x86_ops.cpp` | 572 value assertions. A **grid** drives every operation over 23 (type, width) pairs — FP32/FP64/SI32/PI32/SI64/PI64/SI16 at 2…32 lanes, plus widths that are not a power of two and one wider than any register. Each lands on a different variant per target, and all must agree. |
| `test_x86_dispatch.cpp` | prints the (operation, type, width) rank grid, and asserts on it with `require_at_least` — static asserts, so a lost registration fails the build. |
| `compile_matrix.sh` | 408 translation units across SSE2 / AVX / AVX2 / native, printed as a map. A coverage hole that is a *compile* error cannot be seen from a test binary. Takes the compiler as an argument. |
| `run_all_isa.sh` | every test binary, built and run at five feature levels. `make` alone only exercises this machine, so the SSE2 and AVX paths — the ones most likely to rot — were never touched. This is what caught the two remaining bugs, one of them in the test suite itself. |
| `abi_probe.cpp` | five symbols instead of one, all enforced: a value passed through memory fails the build. |
| `check.h` | `KNOWN_BROKEN( why, cond )` — a diagnosed, unfixed bug prints without failing the build and turns into a loud `XPASS` once fixed. Every one it held is now a plain `CHECK`. |

`make` → 665 assertions, 0 failures. `make isa` → the same at five ISA levels, 2 860 assertions.
`make matrix` → 408 / 408.

---

## 6. What is left

- **clang and MSVC are unverified**, per § 4.
- **NEON.** `architectures/ArmCpu.h` declares the feature and the width, so `SimdVec<float>` on
  AArch64 is four generic lanes that give the right answer; `impl/SimdVecImpl_Neon.h` and
  `SimdMaskImpl_Neon.h` remain to be written. Two primitives to watch: `permute` (`tbl` works on
  bytes, so it costs more for 32-bit lanes) and `to_bits` (no `movemask` — a three or four
  instruction reduction).
- **8- and 16-bit types below AVX-512BW.** They work, entirely through the generic forms.
  `X86CpuFeatures.h` advertises them at every level, so the widths are right and only the
  backends are missing.
- **`SimdRangePtr`** compiles again but is still scaffolding: its loop body is commented out.
- ~~**The `SPLIT` rank (10) is declared and never used**~~ — **done**, see § 7.
  (kept below for the measurement that motivated it)

- **The `SPLIT` rank (10) was declared and never used, and it cost.** Nothing registers a variant
  spanning several registers for one requested width — which is, on paper, what asimd is for. The
  generic forms cover it correctly, but "correctly" is not "well". Measured on `to_bits( a > b )`
  for `float × 8`, against the same thing written with plain gcc vector extensions:

  | | vector extensions | asimd |
  |---|---|---|
  | SSE2 (8 lanes = 2 registers) | 48 instructions | **73** |
  | AVX2 | 12 | **4** |
  | AVX-512 | 12 | **4** |

  Where the dispatch table has an entry, asimd is 3× better. Where it does not, it is 1.5× *worse*
  than what the compiler does on its own — because the fallback goes lane by lane through a
  `BitVec` while gcc keeps the data in registers and reduces with a shift-or tree.
  `cmp_gt` and `to_bits` at `float × 8` under SSE2 both come out at rank 0.

  A split-rank `to_bits` is two `movmskps` and a shift-or; a split-rank comparison is two
  compares. Neither is hard, and both are exactly the case the library exists for. This is the
  most valuable thing left on the list.

- **`horizontal_sum` has no register form at any width**, so `.sum()` is a pairwise tree through
  `values` (14 instructions at 8 lanes). A `vhaddps` ladder, or `_mm512_reduce_add_ps`, would be
  three or four.


---

## 7. The MSVC pass: what the dispatch table is worth on its own

gcc and clang have `vector_size`; **MSVC has none**. `support/VecValues.h` already fell back to a
plain array there, so the library compiled — but the middle branch of the generic fallback, the
one that applies an operation to the whole `values` vector and lets the compiler lower it, went
away with it. Every hole in the dispatch table then became a scalar lane loop.

That also means gcc had been **hiding** the holes. So: `ASIMD_NO_COMPILER_VECTORS` forces the
array form on a compiler that has vectors, which reproduces MSVC's constraint exactly, and
`tests/no_vecext.sh` compiles each cell both ways and counts instructions. A cell that changes is
a cell gcc was carrying.

It found twelve, on `-march=native`, and every one was a real hole:

| what | was | cause |
|---|---|---|
| `fma` on every integer type | 7 → 24 | the generic `fma` looped over lanes instead of calling the library's own `mul` and `add`, which have register forms |
| `iota` at 16 lanes | 9 → 26 | register forms existed only at 8 |
| `mul` on `SI64 × 4` | 6 → 23 | `vpmullq` registered at 512 bits only; it needs AVX-512DQ **and** VL at 128/256 |
| `bcast_lane` on `PI32 × 8` | 5 → 20 | registered for `SI32`, not for its unsigned twin |
| integer `div` | 20 → 35 | **not a hole** — see below |

Fixed, plus one that the measurement did not flag because it was equally bad both ways:
**`horizontal_sum` had no register form at any width.** `.sum()` on eight floats was 44
instructions; it is 11 now. The split form also had to stop reducing each half and adding the two
scalars, and start adding the halves and reducing once.

**2 of 168 remain, and both are integer division.** x86 has no SIMD integer divide at all: both
paths scalarise, and gcc simply reaches for a reciprocal-multiply sequence it cannot apply to an
array. There is no instruction to dispatch to, so no table entry can fix it.

### The split rank, which is the same story one level down

The same question at a width that exceeds the register — `float × 8` under SSE2 — had a worse
answer, because the operations added by `SimdOpsPlus.h` never used the split at all. Their generic
forms walk `values` lane by lane, which is why asimd was **beaten by plain gcc vectors** at the one
thing it exists for. `SimdOpsPlus_Split.h` registers the missing rank for `cmp_*`, `to_bits`,
`select` and `fma`:

| `to_bits( a > b )` on `float × 8` | plain gcc vectors | asimd before | asimd after |
|---|---|---|---|
| SSE2 | 48 | 73 | **10** |
| AVX | 50 | 4 | **4** |
| AVX-512 | 12 | 4 | **4** |

A split form is the same three lines for every type and every width, because it delegates to
whatever the halves resolve to — a register form, a mask register form, or another split.
`permute` was left out at first, on the grounds that moving a lane from one half to the other is
precisely what two half-registers cannot do. That was the wrong call: an operation that opts out
of the split mechanism is a hole in the premise, not an exception to it. It is in now — see § 8.

### How to run it

```sh
make msvc                      # the with/without table, -march=native
make msvc ARCH=-msse2
make isa                       # includes two MSVC-path rows, built and RUN
```

`make isa` runs the whole suite under `ASIMD_NO_COMPILER_VECTORS` as well, so the MSVC
configuration is checked for correctness, not only for codegen.

**Still not verified on MSVC itself** — it is not installed here. What is verified is that the
library no longer *depends* on a compiler extension MSVC lacks, which was the actual risk.


---

## 8. `permute` at a split width, and the CI

### 8.1 The one operation that did not split

`permute` was the exception, and an exception is not something a library whose whole premise is
"the width is the author's choice" gets to have. It is four half-permutations and two blends:

```
r_k = select( i_k < n0, permute( v0, i_k ), permute( v1, i_k - n0 ) )
```

for each half `k`. Both halves are permuted and one is discarded — an index pointing into the
other half is out of range for the half it is handed to, which is harmless because the sub-permute
wraps it and the blend throws that lane away. Only when the two halves have the same width: at
`N = 5` they are 4 and 1, and there is no type in which to say "permute a one-lane vector with a
four-lane index". Those widths keep the generic form, which is correct.

Two register forms were missing underneath it, and without them the split had nothing to delegate
to:

- **`pshufb` at 128 bits (SSSE3).** A 32-bit lane index becomes four byte indices: multiply by
  four, broadcast the low byte across its lane, add 0,1,2,3. Four instructions where `vpermilps`
  needs one — but `vpermilps` is AVX, and without this an SSE machine had no variable permutation
  at all, hence nothing for the split to use.
- **A full 8-lane permutation on AVX without AVX2.** `vpermps` is AVX2; AVX has only `vpermilps`,
  which stays inside each 128-bit half. Permute in place, permute again with the halves swapped,
  blend on whether each index points at the half it started in — `( idx ^ lane_half ) & 4`.

Timed, `permute` on `float × 8`, against the same thing written with gcc's vector extensions:

| | asimd | gcc vectors | |
|---|---|---|---|
| SSE2 | 8.19 ns | 9.08 ns | no variable shuffle exists at all; both go through memory |
| SSE4.1 | **1.89** | 9.34 | the split form, **4.9×** |
| AVX | **1.36** | 3.50 | the new 8-lane form, **2.6×** |
| AVX2 | 0.93 | 0.93 | one `vpermps` either way |
| AVX-512 | 0.90 | 0.90 | idem |

### 8.2 A rank says "better if available". `available` has to mean it.

Under `-mssse3` the split form was selected and came out at **75 instructions against the generic
form's 51** — slower, and chosen anyway because its rank was higher. The cause: `blendv` is
SSE4.1, so the two blends fell back to lane loops and dominated. Availability now asks about
*every piece* of a composite form, not just the headline one.

Same shape, one level up: two backends registering the same `(Op, Key, RANK)` are as ambiguous as
two overloads — the rank orders levels, not variants inside a level. `pshufb` and `vpermilps` both
wanted `REGISTER` for `float × 4`. The fix is to say what is actually true, `Has<SSSE3> && !
Has<AVX>`: the older form is the fallback, not a rival. `Selection.h` documents both limits now.

### 8.3 Three more holes, found by measuring rather than reading

- **`init_sc`, the scalar broadcast, ignored the split.** `SimdVec<float,32>( x )` wrote 32 floats
  to memory one at a time — 25 instructions where two `vbroadcastss` do it. Six now.
- **`to_bits` at twice the native width**: 11.36 ns → 1.13.
- **`horizontal_sum` at twice the native width**: 20.05 ns → 2.41.

`tests/bench_ops.cpp` is what found them, and it carries two traps worth knowing about. Indices
the compiler can fold turn `__builtin_shuffle` into a *constant* shuffle, so the comparison stops
being between two implementations of the same operation. And whichever variant runs first pays the
frequency ramp: on this box that alone looked like a 1.9× regression on a loop the disassembler
said was identical instruction for instruction.

### 8.4 clang, and what it caught

The suite now runs under gcc **and** clang locally, and clang immediately found two things gcc
accepts:

- **`if constexpr` does not discard anything outside a template.** Every `require_at_least` in
  `test_x86_dispatch.cpp` sat inside `if constexpr ( A::Has<...> )` in `main`, so all of them
  fired regardless of the target. gcc let it pass; clang is right. They live in a function
  template now, where the guard actually guards.
- **`at()` returned `T &` into a `vector_size` lane.** clang rejects binding a reference to a
  vector element outright; gcc allows it as an extension. Reading a lane returns a value now, and
  `v[ i ] = x` goes through a small proxy — so `SimdVec::operator[]` worked on clang by accident
  only as long as nobody wrote to it.

Both are exactly the class of bug a second compiler exists to find, and neither would ever have
shown up on this machine.

### 8.5 CI

`.github/workflows/ci.yml`, three jobs:

| job | what runs |
|---|---|
| `linux` (gcc, clang) | `run_all_isa.sh` — every test at five ISA levels plus two MSVC-path ones; `compile_matrix.sh`, asserted at 0/408; `no_vecext.sh` |
| `windows` (MSVC) | `run_msvc.ps1` — every test at `/arch:` SSE2, AVX, AVX2 (built and run) and AVX512 (built only) |
| `probes` | `make`, for the ABI check, which reads the disassembly and needs objdump |

`tests/run_msvc.ps1` is a script and not workflow-inline on purpose: it is the thing to run on a
Windows box directly. Two flags in it are load-bearing — `/Zc:preprocessor`, because MSVC's default
preprocessor mishandles `__VA_ARGS__` and `check.h` is built on variadic macros; and the absence
of `/arch:` for the baseline, because SSE2 is architectural on x64 and MSVC rejects the flag.

AVX-512 is compiled but not run: GitHub's hosted runners do not guarantee the instruction set, and
a `/arch:AVX512` binary faults without it. Compiling still exercises every AVX-512 backend, which
is most of what CI is for. `pwsh tests/run_msvc.ps1 -RunAvx512` on a machine that has it.

**Still unverified: MSVC itself.** It is not installed here, so the Windows job is the first thing
that will actually run it. Given that clang found two real bugs the moment it was pointed at this
code, the honest expectation is that MSVC finds some too — which is the point of adding it.
