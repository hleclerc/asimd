# asimd — x86 audit, and what was done about it

State of the x86 backend, established by compiling and running it rather than by reading it, then
repaired. Everything below is reproduced by `make`, `make isa` or `make matrix` in `tests/`.

Measured on `g++ 15.2`, Xeon W-2145 (Skylake-X: SSE2 … AVX-512VL/BW/DQ).

> **§ 9 is the ARM port**, and four of its findings are in *this* code rather than in ARM's. A
> second backend turns out to be an audit instrument of its own: it reaches widths and lane types
> the x86 tests never asked for, and every generic form it exercises is shared. Those four are
> listed there, not here, because that is where they were found.

## Where it stood, and where it stands

| | before | after |
|---|---|---|
| (operation, type, ISA) cells that compile | 258 / 408 | **408 / 408** |
| (operation, type, width) cells with a register form | 8 / 63 | **59 / 63** |
| operations returning wrong values | 5 | **0** |
| values passed through memory across a call | 2 of 5 probes | **0 of 5** (and see § 9.4: three of those five were the wrong question) |
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
→ complement inside the mask type. `impl/SimdBoolImpl_AVX512.h`

**1.2 `min` / `max` on unsigned types used the SIGNED instruction.**
`PI32`/`PI64` went to `_mm*_min_epi32/64`, so `min( 0xFFFFFFFF, 9 )` returned `0xFFFFFFFF`.
→ `_epu` forms throughout, and the 64-bit ones moved to AVX-512VL where they actually exist.
`impl/SimdVecImpl_{SSE2,AVX2,AVX512}.h`

**1.3 `permute` was wrong at every width that is not a power of two.**
`tmp[ idx & ( N - 1 ) ]`. At N = 5 that is `& 4`: reversing five lanes gave `50 10 10 10 10`.
Arbitrary widths are the reason asimd exists, so this was the generic form failing on the
library's own headline case. → `% N`, which folds to the same `and` at a power of two.
`SimdOps.h`

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

### 3.1 `ops/X86.h` implemented one cell, and it was not the native one

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

README § 3 removed the array and `Split` from `SIMD_VEC_IMPL_REG`. `SIMD_BOOL_IMPL_REG_LARGE` kept
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
`make matrix` → 408 / 408. (After § 9: 1 833 assertions per level, and `make matrix` → 544 / 544.)

---

## 6. What is left

- **clang and MSVC are unverified**, per § 4. ~~clang~~ — **done**, see § 9: verified at five x86
  feature levels and six ARM ones. MSVC, including its ARM64 target, is still reasoned rather
  than measured.
- ~~**NEON**~~ — **done**, see § 9. Both primitives this entry flagged behaved as predicted:
  `permute` does go through `TBL` and does cost more than one instruction for 32-bit lanes (four,
  plus a constant — against `pshufb`'s five), and `to_bits` is indeed a reduction with no
  `movemask` to reach for. The prediction was "three or four instructions"; it is two, an `AND`
  with `{1,2,4,8}` and an `ADDV`.
- **8- and 16-bit types below AVX-512BW.** They still work entirely through the generic forms on
  x86; `X86CpuFeatures.h` advertises them at every level, so the widths are right and only the
  backends are missing. **On ARM they now have register forms at every operation**, because
  Advanced SIMD covers the lane widths uniformly where SSE does not — which is also what made the
  `iota` promotion bug of § 9.2 visible.
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
answer, because the operations added by `SimdOps.h` never used the split at all. Their generic
forms walk `values` lane by lane, which is why asimd was **beaten by plain gcc vectors** at the one
thing it exists for. `ops/Split.h` registers the missing rank for `cmp_*`, `to_bits`,
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

`.github/workflows/ci.yml`, four jobs since § 9:

| job | what runs |
|---|---|
| `linux` (gcc, clang) | `run_all_isa.sh` — every test at five x86 ISA levels plus two MSVC-path ones; `compile_matrix.sh`, asserted at 0/544; `no_vecext.sh` |
| `arm` (gcc, clang) | the same three scripts on a **native arm64 runner**, so the tests run rather than only build; plus `check_arm_lattice.sh`, `make` for the ABI probe, and an ARMv7 cross build |
| `windows` (MSVC) | `run_msvc.ps1` — every test at `/arch:` SSE2, AVX, AVX2 (built and run) and AVX512 (built only) |
| `probes` | `make`, for the ABI check, which reads the disassembly and needs objdump |

The ARM job runs the tests rather than only building them, and that is the point: this backend's
failure modes change *values*, not compilability — a byte index off by one in a `TBL` control, a
`to_bits` reduction that overflows a `uint8_t` and drops the top eight lanes.

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


---

## 9. The ARM backend, and what building it found in the x86 one

The README's § 4 describes the backend; this is the audit half — what the port broke, what it
found, and what it measured.

`architectures/ArmCpuFeatures.h`, `impl/arm_intrin.h`, `impl/SimdVecImpl_Neon.h`,
`impl/SimdBoolImpl_Neon.h`, `ops/Neon.h`, plus `ops/Shapes.h` factored out of
`ops/X86.h` so both backends share the variant *shapes*. Measured on `clang 17`, Apple M4
Pro; the x86 half of every claim below was re-run at `-msse2`, `-msse4.2`, `-mavx` and
`-mavx2 -mfma`, and compiled at AVX-512.

### 9.1 The lattice splits at A64, not at "NEON"

`__ARM_NEON` is defined on ARMv7-A **and** on AArch64, and `float64x2_t`, `vdivq_f32`, `vaddvq_*`,
`vqtbl1q_u8`, `vdupq_laneq_*` and the 64-bit compares are all A64 only — about a third of what a
128-bit backend wants to call. One feature would have repeated § 2.3 exactly: guarded by
`__SSE2__`, half the SSE2 backend was really SSE4.1, and a genuine SSE2 target did not build.

So `NEON` and `ASIMD` are two width-carrying features, and `tests/check_arm_lattice.sh` checks the
boundary **from both sides**:

```
  NEON-guarded intrinsics on ARMv7-A             yes  ok
  NEON-guarded intrinsics on AArch64             yes  ok
  ASIMD-guarded intrinsics on AArch64            yes  ok
  ASIMD-guarded intrinsics on ARMv7-A            no   ok
```

The negative row is the one that earns its place. If those intrinsics *were* available on ARMv7,
the `ASIMD` guard would be denying a 32-bit part instructions it has; and if a later edit moved
one of them under `NEON`, the first row would start failing and name it. Both probe files are C
and `arm_neon.h` is compiler-provided — so this runs with **no cross toolchain**, on any host with
clang. That matters more than it sounds: the ARMv7 lattice is the one claim nobody can check on
the machine they develop on, and this makes it the one claim that is checked everywhere.

**It did not run in CI on the first try, and it lied about why.** `arm_neon.h` includes
`<stdint.h>`; clang's own `stdint.h` does `#include_next` to the system one whenever
`__STDC_HOSTED__` is set, and on the aarch64 Linux runner that lands in the *host's* glibc and
dies on `bits/libc-header-start.h` — the armhf multiarch headers are not installed. The script
reported:

```
  NEON-guarded intrinsics on ARMv7-A             no   EXPECTED yes
      /usr/include/stdint.h:26:10: fatal error: 'bits/libc-header-start.h' file not found
THE LATTICE IS WRONG: a feature guard does not match the instruction set
```

A missing header, reported as a feature-lattice defect. The third false verdict of this port,
after `no_vecext.sh`'s clean sweep (§ 9.2) and the ABI probe's two-way misreading (§ 9.4), and the
worst of the three: the other two said "fine" when they knew nothing, this one sent the reader to
hunt a bug in `ArmCpuFeatures.h` that did not exist.

Two fixes, and the second matters more than the first:

- **`-ffreestanding`** clears `__STDC_HOSTED__`, so clang defines the integer types itself and no
  system header enters the picture — checked with `-H`: `arm_neon.h`, `arm_bf16.h`,
  `arm_vector_types.h` and `stdint.h`, all four from clang's own resource directory. On macOS
  clang's `stdint.h` is self-contained regardless, which is precisely why this was invisible on
  the machine the check was written on. The probe files' pointer parameters also moved to the
  `<stdint.h>` typedefs, since `int64_t` is `long` on LP64 and `long long` on ILP32 and hand-spelling
  it was wrong on one of the two.
- **A third verdict, `SKIPPED`**, for a host that genuinely cannot compile for the target: it names
  the file that was not found and does not blame the lattice. A missing intrinsic is told from a
  missing header by whether a *file* could not be found — not by `fatal error:` alone, because the
  ASIMD-on-ARMv7 row is *meant* to fail, fails with dozens of errors, and clang caps that with
  `fatal error: too many errors emitted`, which the first cut of the discriminator misread in turn.
  Hence also `-ferror-limit=0`.

A skip is honest but silent, and silence is how a check stops being one. So CI passes `--strict`,
under which a skipped row fails the job; `make lattice` stays permissive, because a contributor on
an odd host should not be blocked by a check that cannot run there. All three verdicts were
verified to have teeth: a real defect (an A64 intrinsic moved into the NEON file) exits 1, a
missing header exits 0 permissive and 1 strict, and the clean state exits 0.

`test_arm_ops.cpp` does the runtime half: it runs the entire value grid a second time over
`ArmCpu<64,NEON,FMA>` — NEON without ASIMD, on whatever host it is compiled on — and everything
A64-only has to disappear and leave *the same answers* behind. `test_arm_dispatch.cpp` prints both
rank grids side by side, and the ARMv7 column is visibly poorer (33 of 63 cells generic against 3),
which is what says the two levels are really separate rather than one level written twice.

### 9.2 Five bugs in the x86 code, found by five different mechanisms

None of these is ARM's. Each was found by a different part of the port, which is the argument for a
second backend that the timings do not make. The fifth -- `a - b` through array-to-pointer decay --
is in § 9.6, because MSVC is what found it.

**`any( a > b )` did not compile at 16 lanes** on any target with a register form at 8 — `-mavx`
and up. The split branch of `NAME##_as_a_simd_bool` assumed both halves returned the *bit* flavour
of mask; a register form returns the *lane* flavour, and `SimdBoolImpl<8,32>` does not assign to a
`SimdBoolImpl<8,1>`. A hard error, not a slow path — the same shape of hole as § 2.1, one
mechanism over.

It needed 16 lanes *and* a register form at 8, and it survived because nothing reached it:
`to_bits( a > b )` and `select( a > b, … )` both route through `ops::cmp_gt` and its ranked
variants, and only `any`/`all` on a lazy comparison use that function at all. Found by writing the
ARM grid, where the same shape occurs at `SimdVec<SI16,16>` — an ordinary width, because the ARM
register stops at 128 bits. Fixed with a `requires` in the `if constexpr`, which asks the only
question that matters — do the halves give something this mask can hold — and falls back to the
lane loop when they do not. Both grids now call `any( a > b )` and `all( a > b )` at every cell.

**`FP32 × 4` and `FP64 × 2` were registered twice.** `SimdVecImpl_SSE2.h` registers them with
`cmpps` and `SimdVecImpl_AVX.h` with `vcmpps` — three operands and a predicate. These are ordinary
function overloads, not ranked variants: on an AVX target both constraints held, neither subsumed
the other, and the call was **ambiguous**. Precisely the situation `Selection.h`'s KNOWN LIMITS
note describes ("two backends registering the same (Op, Key, RANK) are as ambiguous as two
overloads would be"), and fixed with the remedy it prescribes — a
`SIMD_VEC_IMPL_CMP_OP_SIMDVEC_EXCL` that says the older form is the *fallback*, not a rival.

**`iota( beg, mul )` did not compile on the 8- and 16-bit lane types** at any width that splits.
`beg + n * mul` is subject to the integer promotions, so on a 16-bit lane the recursive call became
`iota( int, short, S<…> )` and `T` could be deduced as both at once. Found by adding `SI16` and
`SI8` to `compile_matrix.sh` — 408 cells became 544, and 136 of the new ones failed. This is the
second time this one operation has been a compile error for a structural reason (§ 2.1 was the
first, for a different cause); it now has a cell in both value grids rather than a commented-out
line saying it cannot be called.

**`no_vecext.sh` had been measuring nothing.** Its `count()` ran `objdump -d --disassemble=probe`,
which is a GNU binutils spelling; LLVM's objdump rejects it, and Mach-O prefixes symbols with an
underscore besides. `grep -c` on empty output is 0, so every cell read `0->0`, the
"is it more than 25 % worse" test never fired, and the script reported **"0 of 168 cells
degrade"** — a clean sweep that was a broken tool. Nothing in the output said so.

That is the same failure mode the ABI probe would have had on ARM, and for the same reason: a
measurement that silently returns zero is worse than no measurement, because it reports success.
Both now try both spellings and both symbol names, and the ABI probe *raises* if it cannot
disassemble a symbol at all rather than counting zero stack accesses and passing.

With it fixed: four cells degrade on x86, all integer division; two on ARM, both `iota` with a
compile-time argument, where the register form is folded away in both builds and what is being
counted is how the compiler materialises sixteen bytes. And one real hole, which is what the script
is for: 64-bit integer `min`/`max` on ARM — no instruction at any level — was leaning on the
compiler's vector arithmetic. `CMGT` plus `BSL` now does it in two instructions, and the cell no
longer moves.

### 9.3 Two gaps in the `SPLIT` rank

§ 6 listed "the `SPLIT` rank is declared and never used" as outstanding, and § 8 gave it
comparisons, `to_bits`, `select`, `fma` and `permute`. On ARM it is not a refinement but the main
path — the register is 128 bits and never wider, so *every* width above four floats is a split —
and two operations turned out never to have been given one:

- **`bcast_lane`** was a lane loop at every width above the register. Its split form is *exact*,
  where `permute`'s is a compromise: the lane it reads lives in one half, and every lane of the
  result is that value, so it is one half-broadcast stored twice — no blend, no comparison, no
  index arithmetic. The index has to be rebased into the chosen half (`LANE - n0` for the upper
  one), and getting that wrong is silent: it would broadcast a neighbour and every value would
  still look plausible. Six of the nine cells then still generic on ARM.
- **`mask_from_bits`** likewise. `Key<void,N,Arch>` carries no item size, so one registration per
  width decides that width's mask *flavour* and every later `select` lives with it — which is why
  the ARM backend deliberately registers it at 4 and 2 lanes only. At 8 a 16-bit form would have
  been correct and would have driven `select` on `SimdVec<float,8>` into a lane loop.

Both improve an SSE-only x86 build by the same amount. The `mask_from_bits` split also came with a
lesson in its own right: written with `prev_pow_2( N )` directly rather than taken from the impl's
`split_size_0`, it self-recursed at `N = 1` — `prev_pow_2( 1 )` is 1 — and clang reported it as
"`is0` must be initialized by a constant expression", sixty instantiations deep. The other split
variants get the guard for free by asking the *impl* whether it splits; this one has no impl to ask,
so the guard is explicit.

### 9.4 The AArch64 ABI, and the check that was asking the wrong question

§ 3 is the longest section of this document because the SysV eightbyte classification cost a factor
of two. AAPCS64 does not have it: a Homogeneous Vector Aggregate — one to four members, all the
same vector type — travels in `v0`-`v7`, so **at one register** the union that broke SysV breaks
nothing here, and no layout work was needed.

**Above one register, neither convention can help, and the probe did not know that.** This is the
one thing the port got wrong and CI caught:

| | above one register |
|---|---|
| SysV x86-64 | on the **stack**. Two `__m256d` classify SSE,SSEUP,SSEUP,SSEUP then SSE again, and the fifth eightbyte alone is enough. § 3 already calls this "unrecoverable" about `Split` |
| AAPCS64 | **by reference** in `x0`-`x7`, with an indirect return through `x8` |

So `SimdVec<double,8>` — 64 bytes — crosses a call through memory on every target without a 64-byte
register. `probe_fma_f64` was in the *enforced* list, and it passed for one reason only: the machine
this probe was written on is a Skylake-X, where eight doubles are one `zmm`. On a CI runner with
AVX2 it is 18 instructions and 12 stack accesses, and the check failed the build for something no
code change can fix:

```
  probe_fma_f64    14 instructions, 6 touching %rsp
error: probe_fma_f64: this value is passed through MEMORY. Either the union in ...
```

That is a **false negative in the check, reported as a failure in the code** — the most expensive
kind, because it points at the wrong file. Worse, the same probe was reporting the *opposite* false
answer on ARM: AAPCS64's by-reference form means the callee reads its argument through a pointer
register and never touches `sp`, so those widths came out "0 touching sp" while the value sat in
memory the whole time. The README claim written from that reading — that the split crosses a call
in its registers on ARM — was wrong, and is corrected.

Two green lights that meant nothing, in one check, in opposite directions. The same failure mode as
`no_vecext.sh` in § 9.2, and the reason the probe now `raise`s when it cannot disassemble a symbol
instead of counting zero.

**The fix is to ask the question only where it has an answer.** The enforced probes are now written
at `SimdSize<T>` — one register by definition on every target, four lanes under SSE2 and on any ARM
part, eight under AVX2, sixteen under AVX-512 — and their mask flavour is *deduced* from
`decltype( gt( … ) )` rather than spelled out. That last part matters: `SimdBool<16,32>` asks
AVX-512 for a 64-byte **lane** mask, a flavour whose comparisons never produce it and which has no
register impl, so the first version of this fix failed on AVX-512 for a new reason of its own.
Nothing in the enforced set now names a width or a flavour.

The fixed-width 8-lane probes are kept and printed, because they still show whether the SPLIT path
keeps register-level operations underneath it — but that claim is now asserted where it belongs, by
`require_at_least` at SPLIT rank in *both* dispatch tests. A static_assert names the operation;
an instruction count has to be interpreted, and the interpretation is what went wrong here.

Writing those floors found one more thing, immediately: `fma` at a split width is **legitimately**
rank 0 on any target without FMA. With no register `fma` anywhere the split form correctly reports
itself unavailable, and the generic form is `add( mul( a, b ), c )` — whose `mul` and `add` have
register forms and split on their own. Two `mulps` and two `addps`, not a lane loop. The first
version of the floor asserted SPLIT unconditionally and failed at `-msse2`, `-msse4.2` and `-mavx`;
a floor has to ask for what the target can actually give, which is § 8.2 over again.

Three smaller portability fixes the same check needed, all of the silent-zero kind: the stack
pointer is `sp` and not `%rsp`; Mach-O prefixes symbols with `_`; and a frame pointer puts
`%rsp` in a *clean* prologue, so the probe is built with `-fomit-frame-pointer` and zero can mean
zero.

### 9.5 gcc 13 rejected the selection mechanism, and the mechanism was not following its own advice

`test_arm_dispatch.cpp` built everywhere it was tried — clang on x86 at five feature levels and on
AArch64 at six, gcc 15 on AArch64, gcc 15 on the author's Ubuntu — and failed on GitHub's
`ubuntu-latest`, whose default compiler is **gcc 13**:

```
Selection.h:94: error: 'constexpr int asimd::sel::search() [with Op = ops::cmp_gt;
                        Key = Key<float,4,X86Cpu<64,SSE2,SSE>>; int R = 40]'
                        used before its definition
```

on `cmp_gt`, `select` and `permute` at `Key<float,4,…>` — exactly the three cells `main` named
directly inside an `if constexpr`. "Used before its definition", for a specialization whose
definition sits eight lines above the use, is an **instantiation-ordering** complaint: the point of
instantiation gcc picks for `search<Op,Key,MAX_RANK>`, which the variable template `rank` needs to
be initialized, can land before the definition it requires.

And the chain that gets it there is not incidental — it is the design. A split rank's `available`
asks what rank the **halves** reached (`ops/Split.h`), which re-enters `rank`, which
re-enters the search, at a smaller width. On x86 at `-msse2` neither `select` nor `permute` has a
register form at four lanes, so both walk that whole chain; on ARM every one of them is registered,
so none does. That asymmetry is why it reproduced on one architecture and not the other.

**It could not be reproduced here.** Not with gcc 13, gcc 15 or clang on AArch64; not with the
architecture type replaced by one whose backend is absent; not with the discarded `if constexpr`
branch, the function-template point of instantiation, or an architecture whose `permute` lacks a
register form — six attempts, including one with gcc 13.4 installed specifically to try. So the fix
was chosen to be one whose correctness does not depend on knowing the trigger:

**`sel::search` is no longer a `constexpr` function template.** It is a class template, and the
diagnostic gcc emitted is now inexpressible — there is no constexpr function left whose definition
could be used too early. That is not a workaround dressed up: this file's own argument, a few lines
above, is that *a variant is a class specialization and never a function overload, because
specializations are looked up at the point of instantiation rather than of definition*. The search
was the one piece of the mechanism not following it.

Two things had to survive the change, and one of them was not being checked at all:

- **The selected ranks must be identical.** They are — both dispatch grids print the same numbers,
  3 of 63 generic on ARM and 33 of 63 at the ARMv7 floor, on three compilers.
- **The search must stay LAZY.** A rank below the selected one must never be instantiated, or the
  recursion into the halves is walked for nothing on every operation at every width. The class
  template keeps it by computing the "available" flag in a *default template argument*, so `R-1`
  is only ever looked at when `R` is unavailable — and `tests/test_selection.cpp` now proves it,
  with a variant at `SPLIT` whose `available` is `Poison<T>::value` for a `Poison` that is declared
  and never defined. The key it sits on has a `REGISTER` form, so the search must stop above it;
  if it ever looks lower the file stops compiling and names `Poison`. Verified to have teeth by
  moving it one rank up, where it does fail.

The test was also brought in line with `test_x86_dispatch.cpp`, which never hit this: it has always
materialised its ranks through `rank_at_native_width<Op,T>()` rather than naming `sel::rank` inside
a non-template function. Belt and braces — the `Selection.h` change is what makes the error
impossible, that one makes the two files structurally the same.

### 9.6 MSVC ran for the first time, and found a compile error nothing local could reach

`/arch:AVX` failed on the Windows job:

```
SimdVecImpl_Generic.h(611): error C3863: array type 'SimdVecImpl<T_,4,Arch>::Values'
                                         is not assignable
```

**Array-to-pointer decay, in a `requires` test that was asking the wrong question.** The generic
arithmetic form chooses between three branches, and the middle one is guarded by

```cpp
} else if constexpr ( requires { a.data.values OP b.data.values; } ) {
        res.data.values = a.data.values OP b.data.values;
```

which reads as "does `values` support this operator as a whole vector". On MSVC -- and under
`ASIMD_NO_COMPILER_VECTORS` -- `values` is a plain `T[ N ]`, and `a.data.values - b.data.values`
decays to **pointer subtraction**: perfectly valid, and it yields a `ptrdiff_t`. So the test said
yes for `sub`, the branch was taken, and assigning an integer to an array is a hard error. `+`,
`*`, `/` and `&` are all ill-formed on pointers, so `sub` was the only operator affected — which is
why one third of the table was fine and one cell was not.

The fix is one token: ask whether the result can be **assigned back**.

```cpp
} else if constexpr ( requires { res.data.values = a.data.values OP b.data.values; } ) {
```

Pre-existing, and verified so by checking out the tree from before this port. Reproduced locally in
one line with clang: `-mavx -DASIMD_NO_COMPILER_VECTORS` gives
`error: array type 'Values' (aka 'long[4]') is not assignable`, the same defect in clang's words.

**Why no local row reached it, which is the more useful half.** The cell has to be one where a
register IMPL exists but the operation has no register FORM — on x86 that is 256-bit integer
`add`/`sub`: AVX provides the impls, AVX2 provides the instructions. The two `MSVC path` rows in
`run_all_isa.sh` were `-msse2`, where the 128-bit integer add/sub are registered, and
`-march=native`, where the 256-bit ones are. The gap sits exactly between them, and it is exactly
what MSVC calls `/arch:AVX`. There is an `-mavx -DASIMD_NO_COMPILER_VECTORS` row now, labelled as
such; with the old code it fails and with the new code it passes, which is the only test of a new
test row worth running.

All four `/arch:` levels were then walked in the no-vector-extension path — baseline, AVX, AVX2,
AVX512 — and all four compile every test. That is not MSVC, but it is MSVC's *constraint*, which is
the half that can be checked from here. One further MSVC-only risk was removed rather than tested:
the `bcast_lane` split form called a member template with explicit template arguments from inside
its own class, the shape MSVC's two-phase lookup has always been weakest on. It is a free function
template now; nothing is lost, and it is one less thing that can only be found by pushing.

MSVC also reported a `C4244` narrowing warning in `test_ops.cpp`: `to_bits` returns a `PI64`
because a mask can be 64 lanes wide, and the test stored it in an `unsigned`. The value was right
at eight lanes; the type was smaller than the operation hands back.

**And then one runtime failure, at `/arch:AVX2` only. It is closed, without ever having been
reproduced off Windows** -- see the end of this section for what that does and does not establish.

```
    FAIL   tests\test_arm_ops.cpp:157  [SI16x16]  same
```

A store round trip: load sixteen `SI16`, store them back, compare. It fails at `/arch:AVX2` and
passes at `/arch:AVX`, and **nothing in the library differs between the two for that type**:
`SimdVec<SI16,16>` has no register impl on x86 below AVX-512BW, so both levels take the identical
splittable path, same `split_size_0` of 8, same generic recursion down to one lane — checked by
printing the impl's shape at both levels. The same sixteen lanes of the same value are also read by
roughly fifteen other checks in the same grid cell, through `lanes_are`, and every one of them
passes on that build.

So the data is not in question and the dispatch is not in question. What is left is either MSVC's
code generation on the comparison loop, or the one thing that check did which nothing else in the
suite does: reach the value through `SimdVec`'s **member** `store_unaligned( T* )` rather than the
static two-argument form.

It could not be reproduced here — the MSVC *constraint* runs clean at all three levels
(`-msse2`, `-mavx`, `-mavx2 -mfma` with `ASIMD_NO_COMPILER_VECTORS`, built **and run** this time,
which is the gap that let this reach CI in the first place: the MSVC-path rows had only been
compiled). So the check was rewritten to answer the question on the next run rather than to guess:
the static and member spellings of the store, tested apart; the offending lane reported; and the
`bool same = true; … same &= ( … )` reduction dropped, because that shape is a bool reduction over
a counted loop — simultaneously the least informative formulation available and the most likely to
be miscompiled. Two places in the file used it; neither does now.

**The next run answered half of it and exposed a worse problem.** Both spellings failed, which
clears the member forwarding — and *the diagnostic never appeared*:

```
    FAIL   tests\test_arm_ops.cpp:183  [SI16x16]  lanes_match( ... "static form" ... )
    FAIL   tests\test_arm_ops.cpp:184  [SI16x16]  lanes_match( ... "member form" ... )
```

**Both harnesses were filtering it out.** `run_all_isa.sh` printed `^  broken|^  FAIL` and
`run_msvc.ps1` `^$t|FAIL|broken`; the new line began `  [SI16x16]` and matched neither. The
information was produced, on the one compiler that needed it, and thrown away by the script — a
whole CI round trip to learn nothing. Both filters also swallowed `check.h`'s **`XPASS`**, which is
the line that says a known-broken assertion has started passing and should be promoted to a
`CHECK`: a fixed bug would have gone unannounced, indefinitely.

A filter is a place where information goes to die, and this one is the fifth false verdict of this
port. Both now pass `FAIL|broken|XPASS|info`, and the diagnostic is prefixed `info` so it reads
like the verdicts beside it and survives the next person's filter. It dumps **both buffers**, not
just the first bad lane: an untouched upper half points at the split store, scattered differences
point at the data.

Two things were removed from the check at the same time, on the grounds that neither earned its
place and both were exotic on a compiler that cannot be tried here:

- **`alignas( 64 )` on the two output buffers.** They feed `store_unaligned`, which asks nothing of
  the address. What the alignment bought was two more 64-byte-aligned arrays in a nested block of a
  function that already holds several, with `sa` and `sb` still live in the enclosing scope.
  `lanes_are` aligns its own buffer and has never had trouble — it lives alone in its own function.
- **The two independent `auto` parameters** of `lanes_match`, where the two buffers are necessarily
  the same type. One template parameter says so, makes a mismatched call a compile error, and
  removes an abbreviated-template form from a file that goes through MSVC unseen.

What the library does for `SimdVec<SI16,16>` is unchanged and provably identical at `/arch:AVX` and
`/arch:AVX2`.

**It passes.** All four `/arch:` levels build and the three that a hosted runner can execute pass
every assertion -- MSVC is green, which is the last claim in § 6's list to become true.

What that establishes, and what it does not. The library was never implicated and still is not:
the type has no register impl on x86 below AVX-512BW, both levels take the identical splittable
path, and fifteen other assertions read the same lanes on the same build. What is NOT established
is which of the three changes in that round was responsible -- the needless `alignas( 64 )` is the
candidate, since it is the only thing the check did that `lanes_are` does not do from inside its
own function, but the failure was never reproduced here and so nothing was ever isolated. A fix
that cannot be attributed is worth recording as such rather than as a diagnosis.

The transferable part is not about MSVC at all. Three of the five false verdicts in this section
were measurement or reporting faults, not code faults, and the last one cost two CI rounds purely
because a `grep` pattern in a shell script was narrower than the output it filtered. `lanes_are`
now reports the failing lane too -- it backs hundreds of grid assertions and used to say only that
something among them differed.

### 9.7 What the backend costs, where it costs anything

Two honest numbers, both regressions, both understood.

**Integer `div`**: 20 instructions against 12 at `SI32 × 4`, and the same four `sdiv`s either way.
Adding a register impl makes `values` a vector type, and the compiler lowers a vector integer
division by extracting each lane, dividing, and inserting. There is no SIMD integer division on
either architecture, so there is nothing to dispatch to that would fix it; `no_vecext.sh` already
carried a note about this cell on x86, and on ARM the sign is reversed.

**`sum` in `bench_ops.cpp`**: 0.51 → 0.65 ns/op at the native width, and it is **not** `ADDV` being
slow. Removing the `ADDV` form changes nothing — 0.65 either way — and a standalone micro-benchmark
puts `ADDV` and a `faddp` ladder within noise of each other. The disassembly gives the real answer:
the generic form is plain code, so clang **unrolled the benchmark loop eightfold** (35 `fadd.4s`,
28 `fadd`) and hid the latency of the dependency chain through the accumulator. It cannot unroll
through an opaque intrinsic, so the compact loop pays full reduction latency every iteration.

Fewer instructions per call, less freedom for the optimiser around the call. That is the trade every
intrinsic makes, and it is worth writing down because the instruction count — the thing
`no_vecext.sh` measures — says the opposite: 5 against 5 at four lanes, 6 against 8 at eight, 9
against 14 at sixteen. Both measurements are right about different things.

### 9.8 What is left

Three cells on ARM reach nothing better than a lane loop, and they are one cell three times:
`permute` on `double`. The index vector is always `SimdVec<SI32,N>`, and at two lanes that is 64
bits of a 128-bit register — no register impl, so no `idx.data.reg` to build a `TBL` control from.
x86 leaves the identical cell generic for the identical reason. The 4- and 8-lane cases follow from
it: the split form asks the half whether it has anything better, and § 8.2's lesson is that
`available` has to mean it.

A `TBL`-based two-lane form would unlock all three. It is buildable — narrow the index through
`vmovn`, mask, look up — and whether it beats a memory round trip *at two lanes* is unmeasured.
Registering it on the assumption that it does is exactly what § 8.2 was about.

**MSVC on ARM64 is unverified**, the same way MSVC on x86 is: `<arm64_neon.h>` rather than
`<arm_neon.h>`, `__prefetch` rather than `__builtin_prefetch`, and none of the `__ARM_FEATURE_*`
macros defined at all — so `arm_intrin.h` takes what the architecture guarantees (Advanced SIMD and
FMLA) and nothing more, which is the same conservative reading it gives MSVC below `/arch:AVX`.
Reasoned from documented behaviour, not measured. The MSVC *constraint* is measured on both
architectures, through `ASIMD_NO_COMPILER_VECTORS`: the whole suite passes with the compiler's
vector types switched off, on ARM as on x86.
